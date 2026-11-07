"""Explicit derived pair correction, with dense regeneration requests."""
from collections import Counter
from pathlib import Path
import gc
import heapq
import os

import numpy as np
import pycolmap as pc

from alignexp.audit import reciprocal_tracks, observation_preservation, input_image_multiplicity, reprojection_stats
from alignexp.colmap_io import add_view
from alignexp.export import model_ply
from .artifacts import command, publish, sha, verify_stamps
from .colmap_adapter import allocate, reconstruction
from .contracts import digest, ContractError
from .storage import read, write


def bounded_ba(rec,membership,bridges,settings,threads):
    held={r["image_id"] for r in bridges if r["role"]=="validation"}
    seed={r["image_id"] for r in bridges if r["role"]=="fit"}
    shared=[];support=Counter()
    for pid,p in rec.points3D.items():
        images={e.image_id for e in p.track.elements}
        if len({membership[i] for i in images})>1:
            shared.append((p.error,-p.track.length(),pid));support.update(images-held)
    if len(shared)<50:
        raise RuntimeError("Insufficient shared points for correction")
    ranked=seed|set(i for i,_ in support.most_common(settings["max_images"]) if i not in held)
    window=set(sorted(ranked,key=lambda i:(i not in seed,-support[i],i))[:settings["max_images"]])
    variable={pid for _,_,pid in heapq.nsmallest(settings["max_points"],shared)
              if sum(e.image_id in window for e in rec.points3D[pid].track.elements)>=2}
    context_counts=Counter(e.image_id for pid in variable for e in rec.points3D[pid].track.elements if e.image_id not in window and e.image_id not in held)
    context=set(i for i,_ in context_counts.most_common(64))
    if len(context)<2:
        context=set(sorted(window)[:2]);window-=context
    if len(context)<2 or not variable:
        raise RuntimeError("Insufficient local BA anchors")
    before=reprojection_stats(rec);seam_before=reprojection_stats(rec,point_ids=variable)
    anchors={i:rec.images[i].cam_from_world().matrix().copy() for i in context}
    ba=pc.BundleAdjustmentConfig();observed=set()
    for i in window|context:
        ba.add_image(i);ba.set_constant_cam_intrinsics(rec.images[i].camera_id)
        if i in context:
            ba.set_constant_rig_from_world_pose(rec.images[i].frame_id)
        observed.update(int(p.point3D_id) for p in rec.images[i].points2D if p.point3D_id!=pc.INVALID_POINT3D_ID)
    for pid in observed-variable:
        ba.add_constant_point(pid)
    for pid in variable:
        ba.add_variable_point(pid)
    opts=pc.BundleAdjustmentOptions();opts.use_gpu=False;opts.refine_sensor_from_rig=False
    opts.refine_focal_length=False;opts.refine_principal_point=False;opts.refine_extra_params=False
    opts.loss_function_type=pc.LossFunctionType.SOFT_L1;opts.loss_function_scale=2.
    opts.solver_options.max_num_iterations=settings["max_iterations"];opts.solver_options.num_threads=threads
    result=pc.create_default_bundle_adjuster(opts,ba,rec).solve()
    if not result.IsSolutionUsable():
        raise RuntimeError("Local correction solver returned unusable geometry")
    for i,pose in anchors.items():
        np.testing.assert_allclose(rec.images[i].cam_from_world().matrix(),pose,atol=1e-10)
    after=reprojection_stats(rec,update_errors=True);seam_after=reprojection_stats(rec,point_ids=variable)
    if after["mean_error_px"]>1.05*before["mean_error_px"] or seam_after["mean_error_px"]>1.05*seam_before["mean_error_px"]:
        raise RuntimeError("Correction degraded reprojection quality")
    return {"before":before,"after":after,"seam_before":seam_before,"seam_after":seam_after,
            "variable_images":len(window),"variable_points":len(variable),"termination":str(result.termination_type),
            "validation_scope":"training/seam diagnostics; reserved E1 frames remain fixed but may constrain variable points"},result.FullReport()


def correct(workspace,request,scratch):
    plan=request["plan"];mapping=read(workspace/"maps"/plan["map_revision"]/"map.json")
    if digest(mapping)!=plan["map_fingerprint"]:
        raise RuntimeError("Correction map fingerprint changed")
    measured=read(workspace/"measurements"/plan["measurement_id"]/"measurement.json")
    source,target=measured["source"],measured["target"]
    if source not in mapping["placements"] or target not in mapping["placements"]:
        raise ContractError("Pair is not connected to the map frame")
    for ref in (source,target):
        matched=[read(p) for p in (workspace/"audits").glob("*/summary.json") if read(p)["ref"]==ref]
        if not matched:
            raise RuntimeError("Missing correction input audit")
        verify_stamps(matched[-1]["input_stamps"])
    local=Path(scratch)/"correction";local.mkdir()
    manifests={"source":mapping["manifests"][source],"target":mapping["manifests"][target]}
    ns=allocate(manifests)
    src=reconstruction(manifests["source"],ns["source"]);tgt=reconstruction(manifests["target"],ns["target"])
    original_source=local/"source";original_target=local/"target"
    original_source.mkdir();original_target.mkdir();src.write(original_source);tgt.write(original_target)
    bridges=[r for r in read(workspace/"measurements"/plan["measurement_id"]/"registrations.json") if r["registered"]]
    for r in bridges:
        iid=r["image_id"];image=src.images[iid];cid=image.camera_id
        if cid not in tgt.cameras:
            tgt.add_camera(pc.Camera(src.cameras[cid].todict()))
            rig=pc.Rig(rig_id=cid);rig.add_ref_sensor(tgt.cameras[cid].sensor_id);tgt.add_rig(rig)
        add_view(tgt,iid,image.name,cid,cid,np.array([p.xy for p in image.points2D]),pc.Rigid3d(np.array(r["pose"])))
        for index,pid in r["observations"]:
            tgt.add_observation(pid,pc.TrackElement(iid,index))
    extended=local/"extended";extended.mkdir();tgt.write(extended)
    membership={new:source for new in ns["source"]["images"].values()}
    membership.update((new,target) for new in ns["target"]["images"].values())
    del src,tgt;gc.collect()
    merged=local/"model";merged.mkdir()
    command(["colmap","model_merger","--input_path1",original_source,"--input_path2",extended,"--output_path",merged,"--max_reproj_error","8"],local/"merger.log")
    rec=pc.Reconstruction(merged)
    if set(rec.reg_image_ids())!=set(membership):
        raise RuntimeError("Merger did not incorporate the registered image union")
    preservation={source:observation_preservation(original_source,merged),target:observation_preservation(original_target,merged)}
    tracks=reciprocal_tracks(rec)
    baseline=input_image_multiplicity(original_source)+input_image_multiplicity(original_target)
    if tracks["exact_duplicate_elements"] or tracks["duplicate_image_observations"]>baseline or any(a["loss_fraction"]>.05 for a in preservation.values()):
        raise RuntimeError("Correction merge failed structural/preservation audit")
    quality,solver=bounded_ba(rec,membership,bridges,plan,int(os.environ["SLURM_CPUS_PER_TASK"]))
    rec.write(merged);(local/"solver.txt").write_text(solver)
    model_ply(merged,local/"corrected_metric.ply",np.array(mapping["placements"][target]))
    fingerprint=digest({p.name:sha(p) for p in merged.glob("*.bin")})
    body={"schema_version":"1.0","parent_map":mapping["id"],"measurement":measured["id"],"geometry_fingerprint":fingerprint,
          "model_frame":"target original sparse frame","world_from_model":mapping["placements"][target],
          "membership":{str(i):ref for i,ref in membership.items()},"quality":quality,"preservation":preservation,
          "changed_blocks":[source,target],"regeneration_requests":[{"block":r,"status":"requires_regeneration",
             "geometry":"model/","image_ids":[i for i,ref in membership.items() if ref==r],
             "reason":"derived component BA changes internal geometry; old dense is not labelled corrected",
             "note":"consumer must support the component model and operation-local image identities"} for r in (source,target)],
          "original_map_unchanged":True,"publication":"derived pair component; not an automatic multi-block corrected atlas"}
    write(local/"correction.json",body)
    write(local/"images.json",{role:manifest["images"]["root_uri"] for role,manifest in manifests.items()})
    # Keep only the final component, not three full intermediate model copies.
    import shutil
    for p in (original_source,original_target,extended):
        shutil.rmtree(p)
    key=digest({"plan":plan,"engine":request["engine"]});publish(local,workspace/"corrections"/key)
    return {"correction_revision":key,"artifact":"corrections/"+key+"/correction.json","quality":quality,
            "regeneration_requests":body["regeneration_requests"]}
