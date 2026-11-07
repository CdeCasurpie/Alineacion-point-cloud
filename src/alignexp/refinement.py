"""Bounded seam BA: optimized intrinsics stay fixed; context anchors the gauge."""
from collections import Counter
import heapq

import numpy as np
import pycolmap as pc

from .audit import reprojection_stats, reciprocal_tracks
from .common import artifact, config, load, record, require_job, save, stage_dir
from .export import model_ply


def run():
    require_job()
    cfg = config(); settings = cfg["refinement"]
    out = stage_dir("refinement")
    merged = artifact("fusion") / "merged_ref009"
    rec = pc.Reconstruction(merged)
    before = reprojection_stats(rec)
    bridges = [r for r in load(artifact("localization")/"registrations.json") if r.get("registered")]
    # E1 holdout poses are not optimized here. COLMAP may still add their fixed-
    # pose observations for variable points, so E3 is not independent validation.
    holdout = {r["image_id"] for r in bridges if r["role"] == "validation"}
    seed = {r["image_id"] for r in bridges if r["role"] == "fit"}
    shared = []
    support = Counter()
    for pid, point in rec.points3D.items():
        image_ids = {e.image_id for e in point.track.elements}
        if any(i < 1000000 for i in image_ids) and any(i >= 1000000 for i in image_ids):
            shared.append((point.error, -point.track.length(), pid))
            support.update(image_ids - holdout)
    if len(shared) < 50:
        raise RuntimeError("Insufficient shared tracks for seam correction")
    window = seed | set(i for i,_ in support.most_common(settings["max_images"]) if i not in holdout)
    window = set(sorted(window, key=lambda i:(i not in seed, -support[i]))[:settings["max_images"]])
    selected = [pid for _,_,pid in heapq.nsmallest(settings["max_points"], shared)
                if sum(e.image_id in window for e in rec.points3D[pid].track.elements) >= 2]
    variable = set(selected)
    seam_before = reprojection_stats(rec,point_ids=variable)
    context_counts = Counter(e.image_id for pid in selected for e in rec.points3D[pid].track.elements
                             if e.image_id not in window and e.image_id not in holdout)
    context = set(i for i,_ in context_counts.most_common(64))
    if len(context) < 2:
        # Preserve a pair of reference frames if no external observed context exists.
        context = set(i for i in window if i>=1000000)
        context = set(sorted(context)[:2]); window -= context
    if len(context) < 2 or not variable:
        raise RuntimeError("Cannot establish seam BA anchors")
    ba = pc.BundleAdjustmentConfig()
    included = window | context
    anchor_poses = {i:rec.images[i].cam_from_world().matrix().copy() for i in context}
    for i in included:
        ba.add_image(i)
        if i in context:
            ba.set_constant_rig_from_world_pose(rec.images[i].frame_id)
        ba.set_constant_cam_intrinsics(rec.images[i].camera_id)
    observed = set()
    for i in included:
        observed.update(int(p.point3D_id) for p in rec.images[i].points2D if p.point3D_id!=pc.INVALID_POINT3D_ID)
    for pid in observed - variable:
        ba.add_constant_point(pid)
    for pid in variable:
        ba.add_variable_point(pid)
    opts = pc.BundleAdjustmentOptions()
    opts.refine_focal_length = False; opts.refine_principal_point = False; opts.refine_extra_params = False
    opts.refine_sensor_from_rig = False; opts.use_gpu = False
    opts.loss_function_type = pc.LossFunctionType.SOFT_L1
    opts.loss_function_scale = 2.0
    opts.solver_options.max_num_iterations = settings["max_iterations"]
    opts.solver_options.num_threads = settings["threads"]
    save(out/"window.json",{"variable_images":sorted(window),"context_images":sorted(context),
                            "e1_holdout_fixed_pose_images":sorted(holdout),"variable_points":len(variable),
                            "constant_points":len(observed-variable),"shared_tracks":len(shared)})
    summary = pc.create_default_bundle_adjuster(opts,ba,rec).solve()
    (out/"solver.txt").write_text(summary.FullReport())
    if not summary.IsSolutionUsable():
        raise RuntimeError("BA did not produce a usable solution")
    for i,pose in anchor_poses.items():
        np.testing.assert_allclose(rec.images[i].cam_from_world().matrix(),pose,atol=1e-10)
    after = reprojection_stats(rec,update_errors=True)
    seam_after = reprojection_stats(rec,point_ids=variable)
    tracks = reciprocal_tracks(rec)
    result = out/"model";result.mkdir();rec.write(result)
    geo = load(artifact("geography")/"geography.json")
    S = np.array(geo["scenes"][cfg["reference"]]["S_metric_from_local"])
    model_ply(result,out/"E3_refined_metric.ply",S)
    save(out/"evaluation.json",{"before":before,"after":after,"tracks":tracks,
                                "seam_before":seam_before,"seam_after":seam_after,
                                "initial_cost":summary.initial_cost,"final_cost":summary.final_cost,
                                "solver_termination":str(summary.termination_type),
                                "interpretation":"training reprojection and structural audit; no ground-truth accuracy claim"})
    if after["mean_error_px"]>before["mean_error_px"]*1.05:
        raise RuntimeError("BA degraded global reprojection beyond tolerance; candidate retained for review")
    record("refinement",out,before=before,after=after,seam_before=seam_before,seam_after=seam_after,
           variable_images=len(window),variable_points=len(variable),solver_termination=str(summary.termination_type))
