"""Explicit-view alignment. No geographic references and no mutable map graph.

All transforms map column vectors from block-local coordinates into map_world.
The accepted anchor snapshot is an input, never an optimization variable.
"""
from collections import Counter
from pathlib import Path
import gc
import hashlib
import json
import math
import sqlite3

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .contracts import digest
from .graph import similarity, pack, unpack


class ManualError(ValueError):
    pass


DEFAULTS = {
    "min_pair_inliers": 8, "min_pnp_inliers": 30, "min_pnp_ratio": .20,
    "max_reprojection_px": 4., "max_center_error_m": 2.,
    "max_rotation_error_deg": 5., "min_bridges": 6,
}


def policy(value=None):
    value = value or {}
    if set(value)-set(DEFAULTS):
        raise ManualError("Unknown manual alignment policy")
    result = {**DEFAULTS, **value}
    for key, number in result.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
            raise ManualError("Invalid policy: "+key)
        if isinstance(DEFAULTS[key], int) and not isinstance(number, int):
            raise ManualError("Expected integer: "+key)
    if result["min_pnp_ratio"] > 1 or result["min_bridges"] < 4:
        raise ManualError("Invalid evidence policy")
    return result


def transform(matrix, xyz):
    matrix = similarity(matrix)
    xyz = np.asarray(xyz)
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def rotation(matrix):
    matrix = similarity(matrix)
    return matrix[:3, :3]/np.cbrt(np.linalg.det(matrix[:3, :3]))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4*1024**2), b""):
            h.update(b)
    return h.hexdigest()


def stamps(model):
    return {p.name: file_hash(p) for p in sorted(Path(model).glob("*.bin"))}


def audit(manifest, needed):
    """Integrity and feature-index audit independent of geographical placement."""
    from alignexp import binary
    model = Path(manifest["sparse"]["model_uri"])
    views = binary.images(model)
    names = [v["name"] for v in views.values()]
    if not views or len(set(names)) != len(names):
        raise ManualError("Empty or ambiguous sparse")
    for name in names:
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or any(c.isspace() for c in name):
            raise ManualError("Unsafe image name")
    path = Path(manifest["features"]["database_uri"])
    if not path.is_file():
        raise ManualError("Missing feature database")
    wal = Path(str(path)+"-wal")
    if wal.exists() and wal.stat().st_size:
        raise ManualError("Feature database has an active WAL; use an immutable snapshot")
    checks = {"sparse": stamps(model), "database_sha256": file_hash(path), "images": {}}
    with sqlite3.connect(path.as_uri()+"?mode=ro&immutable=1", uri=True) as db:
        for iid in needed:
            if iid not in views:
                raise ManualError("Selected view is not registered")
            view = views[iid]
            row = db.execute("SELECT name,camera_id FROM images WHERE image_id=?", (iid,)).fetchone()
            if row != (view["name"], view["camera_id"]):
                raise ManualError("Model/database identities disagree")
            k = db.execute("SELECT rows,cols,data FROM keypoints WHERE image_id=?", (iid,)).fetchone()
            d = db.execute("SELECT rows,cols,length(data) FROM descriptors WHERE image_id=?", (iid,)).fetchone()
            if not k or k[0] != view["n2d"] or not d or d != (k[0], 128, k[0]*128):
                raise ManualError("SIFT indexing differs from the sparse")
            if k[1] not in (2, 4, 6) or len(k[2]) != k[0]*k[1]*4:
                raise ManualError("Invalid SIFT keypoints")
            positions = binary.point2d(model, view)
            xy = np.c_[positions["x"], positions["y"]]
            if not np.allclose(np.frombuffer(k[2], dtype="<f4").reshape(k[0], k[1])[:, :2], xy, atol=.001, rtol=0):
                raise ManualError("SIFT observation coordinates differ from the sparse")
            image = (Path(manifest["images"]["root_uri"])/view["name"]).resolve()
            if not image.is_relative_to(Path(manifest["images"]["root_uri"]).resolve()) or not image.is_file():
                raise ManualError("Original image unavailable: "+view["name"])
            checks["images"][view["name"]] = file_hash(image)
    return checks


def propose(source_views, target_views, selected, source_world, target_world,
            *, limit=12, radius=80., angle=110.):
    """Propose support images, never select or replace query views."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 64:
        raise ManualError("Pair budget must be between 1 and 64 per query")
    if not math.isfinite(radius) or not 0 < radius <= 1000 or not math.isfinite(angle) or not 0 < angle <= 180:
        raise ManualError("Invalid search radius/angle")
    rs, rt = rotation(source_world), rotation(target_world)
    candidates = []
    for iid in selected:
        if iid not in source_views:
            raise ManualError("Selected view is not registered")
        s = source_views[iid]
        center = transform(source_world, s["center"])
        direction = rs @ np.asarray(s["R"]).T[:, 2]
        ranked = []
        for tid, t in target_views.items():
            distance = float(np.linalg.norm(center-transform(target_world, t["center"])))
            cosine = np.clip(direction @ (rt @ np.asarray(t["R"]).T[:, 2]), -1, 1)
            degrees = float(np.degrees(np.arccos(cosine)))
            if distance <= radius and degrees <= angle:
                ranked.append((distance+radius*.35*degrees/180, tid, distance, degrees))
        # Temporal diversity is advisory, not silent filtering of query selections.
        ranked.sort()
        for _, tid, distance, degrees in ranked[:limit]:
            candidates.append({"query": iid, "support": tid, "distance_m": distance,
                               "angle_deg": degrees, "reason": "prior_distance_and_direction"})
    return candidates


def normalize(outcome, anchor_matrix, direction, anchor):
    if not outcome["registered"]:
        return {**outcome, "anchor": anchor, "direction": direction}
    ra = rotation(anchor_matrix)
    if direction == "forward":
        x = outcome["center_source"]
        y = transform(anchor_matrix, outcome["center_target"])
        # Rcam_world = Rcam_anchor Ranchor_world.
        desired = np.asarray(outcome["R_target"]) @ ra.T
        own = np.asarray(outcome["R_source"])
    else:
        x = outcome["center_target"]
        y = transform(anchor_matrix, outcome["center_source"])
        own = np.asarray(outcome["R_target"])
        desired = np.asarray(outcome["R_source"]) @ ra.T
    return {**outcome, "anchor": anchor, "direction": direction,
            "local_center": np.asarray(x).tolist(), "world_center": np.asarray(y).tolist(),
            "local_R": own.tolist(), "world_R": desired.tolist()}


def residuals(matrix, rows, settings):
    r = rotation(matrix)
    centers = [np.linalg.norm(transform(matrix, x["local_center"])-x["world_center"]) for x in rows]
    angles = [np.degrees(Rotation.from_matrix(
        np.asarray(x["world_R"]) @ (np.asarray(x["local_R"]) @ r.T).T).magnitude()) for x in rows]
    return np.asarray(centers), np.asarray(angles)


def solve(outcomes, prior, settings=None):
    """Joint seven-DoF solve over raw PnP evidence, all anchor poses constant."""
    settings = policy(settings)
    prior = similarity(prior)
    good = [r for r in outcomes if r["registered"]]
    fit = [r for r in good if r["role"] == "fit"]
    held = [r for r in good if r["role"] == "validation"]
    def physical(row):
        return (row["direction"], row.get("anchor") if row["direction"] == "reverse" else "", row["original_id"])
    if len({physical(r) for r in good}) < settings["min_bridges"] or len(fit) < 3 or not held:
        raise ManualError("Insufficient fitting/validation localizations; selections have not been replaced")
    xyz = np.array([r["local_center"] for r in fit])
    singular = np.linalg.svd(xyz-xyz.mean(axis=0), compute_uv=False)
    if singular[0] < 1e-8 or singular[1] < singular[0]*.01:
        raise ManualError("Degenerate camera trajectory: choose views with a wider, non-collinear baseline")
    counts = Counter(r["anchor"] for r in fit)
    # Bound each anchor's total weight, and repeated evidence of the same physical view.
    repeated = Counter(physical(r) for r in fit)
    weights = np.array([1/math.sqrt(counts[r["anchor"]]*repeated[physical(r)])
                        for r in fit])
    weights *= np.array([math.sqrt(min(1., r.get("inliers", 100)/100)*
                                   min(1., r.get("inlier_ratio", 1.)/.5)) for r in fit])

    def objective(v, rows=fit, w=weights):
        m = unpack(v); rr = rotation(m)
        result = []
        for row, weight in zip(rows, w):
            c = (transform(m, row["local_center"])-row["world_center"])/settings["max_center_error_m"]
            rot = Rotation.from_matrix(np.asarray(row["world_R"]) @
                (np.asarray(row["local_R"]) @ rr.T).T).as_rotvec()/np.radians(settings["max_rotation_error_deg"])
            result.extend(np.r_[c, rot]*weight)
        return result

    opt = least_squares(objective, pack(prior), loss="soft_l1", f_scale=.5, max_nfev=300)
    candidate = unpack(opt.x)
    c, a = residuals(candidate, fit, settings)
    inliers = (c <= settings["max_center_error_m"]) & (a <= settings["max_rotation_error_deg"])
    if inliers.sum() < 3 or inliers.mean() < .6:
        raise ManualError("Conflicting anchor evidence; inspect rejected views or refine the prior")
    kept = [row for row, yes in zip(fit, inliers) if yes]
    kept_weights = weights[inliers]
    opt = least_squares(lambda v: objective(v, kept, kept_weights), pack(candidate),
                        loss="soft_l1", max_nfev=200)
    candidate = unpack(opt.x)
    c, a = residuals(candidate, good, settings)
    diagnostics = [{"anchor": row["anchor"], "direction": row["direction"], "original_id": row["original_id"],
                    "role": row["role"], "center_error_m": float(x), "rotation_error_deg": float(y),
                    "inlier": bool(x <= settings["max_center_error_m"] and y <= settings["max_rotation_error_deg"])}
                   for row, x, y in zip(good, c, a)]
    valid = opt.success and all(r["inlier"] for r in diagnostics if r["role"] == "validation")
    by_anchor = {}
    for anchor in sorted(counts):
        group = [r for r in diagnostics if r["anchor"] == anchor]
        by_anchor[anchor] = {"localized": len(group), "inliers": sum(r["inlier"] for r in group),
                             "median_center_m": float(np.median([r["center_error_m"] for r in group]))}
    sensitivity = {}
    if len(counts) >= 3:
        center = xyz.mean(axis=0)
        for anchor in counts:
            rows = [r for r in kept if r["anchor"] != anchor]
            positions = np.array([r["local_center"] for r in rows])
            if len(rows) < 3 or np.linalg.matrix_rank(positions-positions.mean(axis=0), tol=1e-6) < 2:
                sensitivity[anchor] = {"status": "insufficient_remaining_baseline"}
                continue
            trial = least_squares(lambda v: objective(v, rows, np.ones(len(rows))), pack(candidate),
                                  loss="soft_l1", max_nfev=150)
            alternative = unpack(trial.x)
            sensitivity[anchor] = {
                "status": "diagnostic_only", "center_change_m": float(np.linalg.norm(transform(alternative, center)-transform(candidate, center))),
                "rotation_change_deg": float(np.degrees(Rotation.from_matrix(rotation(alternative)@rotation(candidate).T).magnitude())),
                "scale_change_percent": float((np.cbrt(np.linalg.det(alternative[:3,:3])/np.linalg.det(candidate[:3,:3]))-1)*100)}
    return {"matrix": candidate.tolist(), "validated": bool(valid), "diagnostics": diagnostics,
            "by_anchor": by_anchor, "fit_views": len(fit), "validation_views": len(held),
            "leave_one_anchor_out": sensitivity,
            "initial_matrix": prior.tolist(), "solver": opt.message,
            "validation_scope": "Reserved PnP views; original SfM and reciprocal matches may be correlated",
            "anchors_fixed": True, "geometry_changed": False}


def measure(source, target, selection, scratch, settings=None, threads=4, keep_cross=True):
    """Localize EXACT explicit views in one direction; no minimum per anchor."""
    import pycolmap as pc
    from .colmap_adapter import allocate, feature_database, reconstruction
    from .pairwise import localize
    settings = policy(settings)
    scratch = Path(scratch); scratch.mkdir(parents=True, exist_ok=False)
    manifests = {"source": source, "target": target}
    ns = allocate(manifests)
    needed = {"source": {p["query"] for p in selection["pairs"]},
              "target": {p["support"] for p in selection["pairs"]}}
    # Include unmatched queries so the report preserves the entire user selection.
    needed["source"].update(v["id"] for v in selection["views"])
    integrity = {role: audit(m, needed[role]) for role, m in manifests.items()}
    candidates = [{"original_id": v["id"], "image_id": ns["source"]["images"][v["id"]],
                   "name": ns["source"]["names"][v["id"]], "role": v["role"]} for v in selection["views"]]
    db_path = scratch/"features.db"
    feature_database(manifests, ns, db_path, selected=needed)
    pairs = sorted({(ns["source"]["names"][p["query"]], ns["target"]["names"][p["support"]])
                    for p in selection["pairs"]})
    pair_file = scratch/"pairs.txt"
    pair_file.write_text("".join(f"{a} {b}\n" for a, b in pairs))
    if pairs:
        import subprocess
        with (scratch/"matching.log").open("w") as log:
            subprocess.run(["colmap", "matches_importer", "--database_path", str(db_path),
                "--match_list_path", str(pair_file), "--match_type", "pairs",
                "--FeatureMatching.use_gpu", "0", "--FeatureMatching.num_threads", str(threads)],
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)
    src = reconstruction(source, ns["source"])
    tgt = reconstruction(target, ns["target"])
    with sqlite3.connect(db_path) as db:
        outcomes = localize(src, tgt, db, candidates, settings)
    # Retain fit-only cross observations needed by constrained BA. Original IDs
    # and 2D indices are explicit; no assumptions about numeric ID ranges.
    inverse = {v: k for k, v in ns["target"]["images"].items()}
    for row in outcomes:
        row["reason"] = ("localized" if row["registered"] else
            "insufficient_2d_3d_or_pnp_support")
        if keep_cross and row["registered"] and row["role"] == "fit":
            cross = []
            for index, pid in row["observations"]:
                point = tgt.points3D[pid]
                for e in point.track.elements:
                    if inverse[e.image_id] not in needed["target"]:
                        continue
                    im = tgt.images[e.image_id]
                    cross.append({"query_index": index, "target_image": inverse[e.image_id],
                                  "target_index": int(e.point2D_idx),
                                  "xy": im.point2D(e.point2D_idx).xy.tolist()})
            row["cross_observations"] = cross
    # Verify inputs remained identical before publishing any scientific result.
    if integrity != {role: audit(m, needed[role]) for role, m in manifests.items()}:
        raise ManualError("Input changed during matching")
    (scratch/"integrity.json").write_text(json.dumps(integrity, allow_nan=False))
    db_path.unlink()
    del src, tgt
    gc.collect()
    return outcomes


def constrained_ba(manifest, matrix, outcomes, anchors, destination, *, max_images=48,
                   max_points=3000, max_iterations=40):
    """True reprojection BA of incoming poses/points only, fixed anchor cameras.

    Anchor models are never merged or modified. Only fitting cross-observations
    enter the objective; reserved incoming views are excluded even as context.
    """
    import pycolmap as pc
    from scipy.sparse import lil_matrix
    from alignexp import binary
    if not 3 <= max_images <= 64 or not 1 <= max_points <= 10000 or not 1 <= max_iterations <= 100:
        raise ManualError("BA exceeds bounded budgets")
    rec = pc.Reconstruction(manifest["sparse"]["model_uri"])
    rec.transform(pc.Sim3d(np.cbrt(np.linalg.det(np.asarray(matrix)[:3, :3])),
                          pc.Rotation3d(rotation(matrix)), np.asarray(matrix)[:3, 3]))
    held = {r["original_id"] for r in outcomes if r["direction"] == "forward" and r["role"] == "validation"}
    anchor_held = {(r["anchor"], r["original_id"]) for r in outcomes
                   if r["direction"] == "reverse" and r["role"] == "validation"}
    fits = [r for r in outcomes if r["registered"] and r["role"] == "fit"]
    observations = []
    anchor_cache = {}
    for row in fits:
        aid = row["anchor"]
        if aid not in anchor_cache:
            anchor_cache[aid] = (binary.images(anchors[aid]["sparse"]["model_uri"]),
                                 binary.cameras(anchors[aid]["sparse"]["model_uri"]))
        views, cameras = anchor_cache[aid]
        ar = rotation(anchors[aid]["matrix"]); am = np.asarray(anchors[aid]["matrix"])
        for x in row.get("cross_observations", []):
            if row["direction"] == "forward":
                im = rec.images[row["original_id"]]
                pid = im.point2D(x["query_index"]).point3D_id
                anchor_iid, xy = x["target_image"], x["xy"]
            else:
                if x["target_image"] in held:
                    continue
                im = rec.images[x["target_image"]]
                pid = im.point2D(x["target_index"]).point3D_id
                anchor_iid = row["original_id"]
                # The inverse query observation belongs to the original anchor.
                pts = binary.point2d(anchors[aid]["sparse"]["model_uri"], views[anchor_iid])
                xy = [float(pts["x"][x["query_index"]]), float(pts["y"][x["query_index"]])]
            if pid == pc.INVALID_POINT3D_ID:
                continue
            if (aid, anchor_iid) in anchor_held:
                continue
            v = views[anchor_iid]; cal = cameras[v["camera_id"]]
            rr = np.asarray(v["R"]) @ ar.T
            center = transform(am, v["center"])
            cam = pc.Camera(model=cal["model"], width=cal["width"], height=cal["height"], params=cal["params"])
            observations.append((int(pid), ("anchor", aid, anchor_iid), np.asarray(xy), rr, -rr@center, cam))
    support = Counter(x[0] for x in observations)
    pids = [p for p, _ in support.most_common(max_points)]
    if len(pids) < 50:
        raise ManualError("Insufficient shared points for BA (need at least 50)")
    pset = set(pids)
    incoming = Counter(e.image_id for p in pids for e in rec.points3D[p].track.elements if e.image_id not in held)
    variable = set(i for i, _ in incoming.most_common(max_images))
    if len(variable) < 3:
        raise ManualError("Insufficient incoming fitting cameras for BA")
    observations = [o for o in observations if o[0] in pset]
    for pid in pids:
        for e in rec.points3D[pid].track.elements:
            if e.image_id in held:
                continue
            im = rec.images[e.image_id]; pose = im.cam_from_world()
            observations.append((pid, ("incoming", e.image_id), im.point2D(e.point2D_idx).xy.copy(),
                                 pose.rotation.matrix(), pose.translation.copy(), im.camera))
    # Same physical observation must never receive multiple/new competing points.
    unique = {}; conflicts = set()
    for o in observations:
        key = (o[1], tuple(np.round(o[2], 6)))
        if key in unique and unique[key][0] != o[0]:
            conflicts.add(key)
        else:
            unique[key] = o
    observations = [o for k, o in unique.items() if k not in conflicts]
    if len(observations) > 150000:
        raise ManualError("BA observation budget exceeded; reduce the point budget")
    ids = sorted(variable); ci = {iid: n for n, iid in enumerate(ids)}
    pi = {pid: n for n, pid in enumerate(pids)}
    start = np.r_[np.concatenate([np.r_[Rotation.from_matrix(rec.images[i].cam_from_world().rotation.matrix()).as_rotvec(),
                                      rec.images[i].cam_from_world().translation] for i in ids]),
                  np.concatenate([rec.points3D[p].xyz for p in pids])]
    groups = {}
    for n, o in enumerate(observations):
        groups.setdefault(o[1], []).append(n)
    sparse = lil_matrix((2*len(observations), len(start)), dtype=np.int8)
    for n, (pid, key, *_rest) in enumerate(observations):
        sparse[2*n:2*n+2, 6*len(ids)+3*pi[pid]:6*len(ids)+3*pi[pid]+3] = 1
        if key[0] == "incoming" and key[1] in ci:
            sparse[2*n:2*n+2, 6*ci[key[1]]:6*ci[key[1]]+6] = 1
    def errors(v):
        xyz = v[6*len(ids):].reshape(-1, 3)
        result = np.empty((len(observations), 2))
        for key, indices in groups.items():
            first = observations[indices[0]]
            rr, tt, cam = first[3:]
            if key[0] == "incoming" and key[1] in ci:
                pose = v[6*ci[key[1]]:6*ci[key[1]]+6]
                rr, tt = Rotation.from_rotvec(pose[:3]).as_matrix(), pose[3:]
            camera_xyz = xyz[[pi[observations[j][0]] for j in indices]] @ rr.T + tt
            projection = np.asarray(cam.img_from_cam(camera_xyz))
            delta = projection-np.array([observations[j][2] for j in indices])
            delta[~np.isfinite(delta)] = 1e4
            delta[camera_xyz[:, 2] <= 0] = 1e4
            result[indices] = delta
        return result.ravel()
    before = float(np.mean(np.linalg.norm(errors(start).reshape(-1, 2), axis=1)))
    opt = least_squares(errors, start, jac_sparsity=sparse.tocsr(), loss="soft_l1",
                        f_scale=2., max_nfev=max_iterations, x_scale="jac", ftol=1e-5)
    after = float(np.mean(np.linalg.norm(errors(opt.x).reshape(-1, 2), axis=1)))
    if opt.status < 0 or not np.isfinite(opt.x).all() or after > before*1.05:
        raise ManualError("BA failed or degraded reprojection")
    # Held-out incoming reprojection is never in the optimization objective.
    def held_errors(points):
        values = []
        for iid in held:
            im = rec.images[iid]
            for p in im.points2D:
                if p.point3D_id in pi:
                    q = im.cam_from_world()*points[pi[p.point3D_id]]
                    if q[2] <= 0:
                        values.append(1e4)
                    else:
                        e = np.linalg.norm(np.asarray(im.camera.img_from_cam(q))-p.xy)
                        values.append(float(e) if np.isfinite(e) else 1e4)
        return float(np.mean(values)) if values else None
    old_points = start[6*len(ids):].reshape(-1, 3)
    new_points = opt.x[6*len(ids):].reshape(-1, 3)
    hold_before, hold_after = held_errors(old_points), held_errors(new_points)
    if hold_before is None:
        raise ManualError("BA requires independent held-out incoming reprojections")
    if hold_after > hold_before*1.05+.05:
        raise ManualError("BA degraded reserved-view reprojection")
    # Bound internal deformation relative to the placed input.
    displacement = np.linalg.norm(new_points-old_points, axis=1)
    if np.quantile(displacement, .95) > 2.:
        raise ManualError("BA exceeded 2 m point deformation guard")
    for iid, n in ci.items():
        v = opt.x[6*n:6*n+6]
        pose = pc.Rigid3d(pc.Rotation3d(Rotation.from_rotvec(v[:3]).as_matrix()), v[3:])
        original = rec.images[iid].cam_from_world()
        distance = np.linalg.norm(pose.inverse().translation-original.inverse().translation)
        angle = np.degrees(Rotation.from_matrix(pose.rotation.matrix() @ original.rotation.matrix().T).magnitude())
        if distance > 2 or angle > 5:
            raise ManualError("BA exceeded 2 m / 5 degree camera deformation guards")
        rec.frames[rec.images[iid].frame_id].rig_from_world = pose
    for pid, n in pi.items():
        rec.points3D[pid].xyz = new_points[n]
    destination = Path(destination); destination.mkdir()
    rec.write(destination)
    return {"geometry_changed": True, "anchors_fixed": True, "dense_requires_regeneration": True,
            "before_reprojection_px": before, "after_reprojection_px": after,
            "reserved_before_px": hold_before, "reserved_after_px": hold_after,
            "variable_images": len(ids), "variable_points": len(pids),
            "variable_image_ids": ids,
            "solver": opt.message, "converged": bool(opt.success),
            "validation_scope": "Reserved incoming observations excluded; original SfM still correlated",
            "world_from_model": np.eye(4).tolist()}
