import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from shapely.geometry import shape, box, mapping
from shapely.ops import transform
from shapely.affinity import translate
import shapely

from . import binary
from .common import ROOT, config, load, record, require_job, save, stage_dir
from .sim3 import apply, robust


def seconds(value):
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(value.replace(",", ".").split(":"))))


def frame_locations(scene):
    root = ROOT / "inputs" / scene
    tracks = {}
    for path in (root / "telemetry").rglob("*_telemetry.csv"):
        rows = []
        with path.open() as f:
            for r in csv.DictReader(f):
                try:
                    values = [seconds(r["timestamp"]), float(r["longitude"]), float(r["latitude"]), float(r["abs_alt"])]
                except (KeyError, ValueError):
                    continue
                if np.isfinite(values).all() and values[1:3] != [0, 0]:
                    rows.append(values)
        tracks[path.name.removesuffix("_telemetry.csv")] = np.asarray(rows)
    info = load(root / "image_info.json")
    result = {}
    cfg = config()["gps"]
    for name, image in info.items():
        values = tracks.get(image["clip"])
        if values is None or not len(values):
            continue
        t = image["time_s"] + cfg["time_shift_s"]
        times = values[:, 0]
        i = np.searchsorted(times, t)
        if i < len(times) and abs(times[i] - t) < 1e-6:
            result[name] = values[i, 1:]
        elif 0 < i < len(times) and max(t - times[i-1], times[i] - t) <= cfg["max_gap_s"]:
            alpha = (t - times[i-1]) / (times[i] - times[i-1])
            result[name] = values[i-1, 1:] * (1-alpha) + values[i, 1:] * alpha
    return result


def run():
    require_job()
    cfg = config()
    out = stage_dir("geography")
    project = Transformer.from_crs("EPSG:4326", cfg["gis"]["metric_crs"], always_xy=True)
    gps = {s: frame_locations(s) for s in cfg["sources"]}
    all_positions = np.array([p for values in gps.values() for p in values.values()])
    e, n = project.transform(all_positions[:,0], all_positions[:,1])
    origin = np.median(np.c_[e, n, all_positions[:,2]], axis=0)
    clouds, cameras, diagnostics = {}, {}, {}
    for scene in cfg["sources"]:
        model = ROOT / "inputs" / scene / "model"
        views = binary.images(model)
        valid = [v for v in views.values() if v["name"] in gps[scene]]
        source = np.array([v["center"] for v in valid])
        raw = np.array([gps[scene][v["name"]] for v in valid])
        x,y = project.transform(raw[:,0], raw[:,1])
        target = np.c_[x,y,raw[:,2]] - origin
        if np.ptp(target[:,:2], axis=0).max() < cfg["gps"]["min_baseline_m"]:
            raise RuntimeError("GPS trajectory has insufficient baseline")
        S, mask, errors = robust(source, target, cfg["gps"]["robust_threshold_m"], z_weight=0.3)
        if mask.sum() < 6:
            raise RuntimeError("Insufficient geographic alignment consensus")
        pts = binary.points(model)
        xyz = apply(S, pts["xyz"])
        np.savez(out / f"points_{scene}.npz", points=pts, xyz=xyz)
        cameras[scene] = np.array([apply(S, v["center"]) for v in views.values()])
        clouds[scene] = xyz[(pts["length"] >= 3) & (pts["error"] < 2)]
        diagnostics[scene] = {"S_metric_from_local": S.tolist(), "gps_images": len(valid),
                              "gps_inliers": int(mask.sum()), "scale": float(np.cbrt(np.linalg.det(S[:3,:3]))),
                              "median_weighted_error_m": float(np.median(errors)),
                              "p90_weighted_error_m": float(np.quantile(errors,.9)),
                              "temporal_note": "nominal sampled timestamps; no RTK accuracy claimed"}
        save(out / f"cameras_{scene}.json", [{"image_id":v["id"], "name":v["name"],
              "center":apply(S,v["center"]).tolist(), "R_metric_from_camera": (S[:3,:3] / np.cbrt(np.linalg.det(S[:3,:3])) @ v["R"].T).tolist()}
              for v in views.values()])
    save(out / "geography.json", {"crs":cfg["gis"]["metric_crs"], "origin":origin.tolist(),
                                  "vertical_datum":"DJI abs_alt, not independently certified", "scenes":diagnostics})
    all_c = np.concatenate(list(cameras.values()))
    aoi = box(all_c[:,0].min()-100, all_c[:,1].min()-100, all_c[:,0].max()+100, all_c[:,1].max()+100)
    candidates = []
    samples = {}
    for scene,pts in clouds.items():
        _,idx = np.unique(np.floor(pts[:,:2]/2).astype(np.int64),axis=0,return_index=True)
        samples[scene] = pts[idx]
    for feature in load(ROOT / "inputs/gis/block_candidates.geojson")["features"]:
        polygon = translate(transform(project.transform, shape(feature["geometry"])), xoff=-origin[0], yoff=-origin[1])
        if polygon.intersects(aoi):
            counts = {}
            for scene, sample in samples.items():
                counts[scene] = int(shapely.intersects_xy(polygon, sample[:,0],sample[:,1]).sum())
            candidates.append({"properties":feature["properties"],"geometry":mapping(polygon),"support_cells":counts})
    save(out / "region_candidates.json", candidates)
    lots = []
    for feature in load(ROOT / "inputs/gis/lots.geojson")["features"]:
        polygon = translate(transform(project.transform, shape(feature["geometry"])), xoff=-origin[0], yoff=-origin[1])
        if polygon.intersects(aoi):
            lots.append((str(int(feature["properties"]["objectid"])), polygon))
    for with_points in (False, True):
        fig, ax = plt.subplots(figsize=(15,14))
        for _, polygon in lots:
            x,y = polygon.exterior.xy
            ax.plot(x,y,color="gray",linewidth=.45)
        for f in candidates:
            polygon = shape(f["geometry"])
            x,y = polygon.exterior.xy
            ax.plot(x,y,color="black",linewidth=.9)
            center = polygon.representative_point()
            ax.text(center.x,center.y,f["properties"]["id"][-6:],fontsize=7,color="darkgreen")
        for scene,color in (("005","royalblue"),("009","darkorange")):
            c=cameras[scene]
            ax.scatter(c[:,0],c[:,1],s=7,c=color,label=f"camera {scene}")
            if with_points:
                pts=clouds[scene]; step=max(1,len(pts)//15000)
                ax.scatter(pts[::step,0],pts[::step,1],s=.2,c=color,alpha=.15)
        ax.set_xlim(aoi.bounds[0],aoi.bounds[2]); ax.set_ylim(aoi.bounds[1],aoi.bounds[3])
        ax.set_aspect("equal");ax.legend();ax.grid(alpha=.2)
        ax.set_title("005–009: local UTM18S, lot parcels, candidate IDs and GPS-aligned cameras")
        ax.set_xlabel("Easting offset (m)");ax.set_ylabel("Northing offset (m)")
        fig.savefig(out / ("gis_points.png" if with_points else "gis_cameras.png"),dpi=160)
        plt.close(fig)
    record("geography",out,diagnostics=diagnostics,candidates=len(candidates),lots=len(lots))
