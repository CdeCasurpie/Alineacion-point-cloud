import subprocess

import numpy as np
import pycolmap as pc

from . import binary
from .audit import observation_preservation, reciprocal_tracks, reprojection_stats, input_image_multiplicity
from .common import ROOT, artifact, config, load, record, require_job, save, stage_dir
from .export import model_ply, ply
from .sim3 import apply, umeyama


def verify_image_union(rec, source_ids, reference_ids):
    if set(rec.reg_image_ids()) != set(source_ids) | set(reference_ids):
        raise RuntimeError("Merger did not incorporate the exact registered image union")


def run():
    require_job();cfg=config();out=stage_dir("fusion");ids=artifact("identities");local=artifact("localization")
    src=ids/"models"/cfg["source"];ref=ids/"models"/cfg["reference"]
    merged=out/"merged_ref009";merged.mkdir()
    command=["colmap","model_merger","--input_path1",str(src),"--input_path2",str(local/"reference_extended"),
             "--output_path",str(merged),"--max_reproj_error","8"]
    with (out/"merger.log").open("w") as f:
        p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError("Native merger returned an error")
    # Native merger can return success even if it did not merge. Verify union.
    rec=pc.Reconstruction(merged)
    original_src=binary.images(src);original_ref=binary.images(ref)
    verify_image_union(rec,original_src,original_ref)
    audits={s:observation_preservation(ids/"models"/s,merged) for s in cfg["sources"]}
    tracks=reciprocal_tracks(rec)
    # Multiple distinct SIFT features from one image already exist in the
    # originals. Do not confuse them with repeated identical track elements.
    baseline_multiplicity=sum(input_image_multiplicity(ROOT/"inputs"/s/"model") for s in cfg["sources"])
    source_views=[v for i,v in original_src.items() if i not in {r["image_id"] for r in load(local/"registrations.json") if r.get("registered")}]
    S_actual=umeyama(np.array([v["center"] for v in source_views]),np.array([rec.images[v["id"]].projection_center() for v in source_views]))
    S_expected=np.array(load(local/"alignment.json")["S_reference_from_source"])
    geo=load(artifact("geography")/"geography.json")
    S_ref=np.array(geo["scenes"][cfg["reference"]]["S_metric_from_local"])
    centers=np.array([v["center"] for v in source_views])
    discrepancy=np.linalg.norm(apply(S_actual,centers)-apply(S_expected,centers),axis=1)*geo["scenes"][cfg["reference"]]["scale"]
    stats=reprojection_stats(rec,update_errors=True)
    rec.write(merged)
    source_points=binary.points(ROOT/"inputs"/cfg["source"]/"model")
    ref_points=binary.points(ROOT/"inputs"/cfg["reference"]/"model")
    S_src=np.array(geo["scenes"][cfg["source"]]["S_metric_from_local"])
    baseline=np.r_[apply(S_src,source_points["xyz"]),apply(S_ref,ref_points["xyz"])]
    visual=np.r_[apply(S_ref@S_expected,source_points["xyz"]),apply(S_ref,ref_points["xyz"])]
    colors=np.r_[np.tile([35,90,230],(len(source_points),1)),np.tile([245,145,25],(len(ref_points),1))].astype(np.uint8)
    ply(out/"E0_gps_colored.ply",baseline,colors)
    ply(out/"E1_visual_colored.ply",visual,colors)
    model_ply(merged,out/"E2_merged_metric.ply",S_ref)
    summary={"stats":stats,"observation_audit":audits,"tracks":tracks,
             "input_duplicate_image_observations":baseline_multiplicity,
             "S_native_reference_from_source":S_actual.tolist(),
             "native_vs_estimated_center_median_m":float(np.median(discrepancy)),
             "native_vs_estimated_center_max_m":float(discrepancy.max()),
             "coordinate_note":"native model in 009 frame; PLY in local UTM meters with geography origin"}
    save(out/"audit.json",summary)
    if tracks["exact_duplicate_elements"] or tracks["duplicate_image_observations"]>baseline_multiplicity or any(a["loss_fraction"]>0.05 for a in audits.values()) or discrepancy.max()>3:
        raise RuntimeError("Merged candidate needs review: track loss/conflict or alignment disagreement")
    record("fusion",out,**summary)
