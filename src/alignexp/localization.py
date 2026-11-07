from collections import Counter, defaultdict
import os
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pycolmap as pc
from scipy.spatial.transform import Rotation

from .colmap_io import add_view
from .common import artifact, config, load, record, require_job, save, stage_dir
from .sim3 import apply, robust


def run():
    require_job();cfg=config();out=stage_dir("localization");canonical=artifact("identities");matched=artifact("matching");selected=artifact("selection")
    source=pc.Reconstruction(canonical/"models"/cfg["source"])
    target=pc.Reconstruction(canonical/"models"/cfg["reference"])
    dbpath=Path(os.environ["TMPDIR"])/"localization.db"
    shutil.copy2(matched/"database.db",dbpath)
    db=sqlite3.connect(dbpath)
    dbnames=dict(db.execute("SELECT name,image_id FROM images"))
    pairs=load(matched/"pairs.json")
    by_source=defaultdict(list)
    for pair in pairs:
        if pair["inliers"]>=cfg["registration"]["min_pair_inliers"]:
            by_source[dbnames[pair["a"]]].append(pair)
    initial_centers={i:target.images[i].projection_center().copy() for i in target.reg_image_ids()}
    outcomes=[];accepted=[]
    est=pc.AbsolutePoseEstimationOptions()
    est.estimate_focal_length=False
    est.ransac.max_error=cfg["registration"]["max_error_px"]
    est.ransac.max_num_trials=10000
    refine=pc.AbsolutePoseRefinementOptions()
    refine.refine_focal_length=False;refine.refine_extra_params=False
    for candidate in load(selected/"candidates.json"):
        iid=candidate["image_id"];image=source.images[iid]
        associations=defaultdict(Counter)
        for pair in by_source[iid]:
            tid=dbnames[pair["b"]]
            if iid>=tid:
                raise RuntimeError("Unexpected pair orientation")
            row=db.execute("SELECT rows,data FROM two_view_geometries WHERE pair_id=?",(pair["pair_id"],)).fetchone()
            matches=np.frombuffer(row[1],dtype="<u4").reshape(-1,2)
            other=target.images[tid]
            for index,j in matches:
                own=image.point2D(int(index)).point3D_id
                pid=other.point2D(int(j)).point3D_id
                if own!=pc.INVALID_POINT3D_ID and pid!=pc.INVALID_POINT3D_ID:
                    associations[int(index)][int(pid)]+=1
        # One source feature and one target 3D point per PnP correspondence.
        choices=[]
        for index,votes in associations.items():
            ranked=votes.most_common()
            if len(ranked)>1 and ranked[0][1]==ranked[1][1]:
                continue
            choices.append((index,ranked[0][0],ranked[0][1]))
        choices.sort(key=lambda r:r[2],reverse=True)
        used=set();unique=[]
        for index,pid,votes in choices:
            if pid not in used:
                unique.append((index,pid));used.add(pid)
        diagnostic={**candidate,"correspondences_2d3d":len(unique)}
        if len(unique)<cfg["registration"]["min_inliers"]:
            diagnostic.update(registered=False,reason="insufficient_unique_2d3d")
            outcomes.append(diagnostic);continue
        xy=np.array([image.point2D(i).xy for i,p in unique])
        xyz=np.array([target.points3D[p].xyz for i,p in unique])
        original_camera=source.cameras[image.camera_id]
        camera=pc.Camera(original_camera.todict())
        result=pc.estimate_and_refine_absolute_pose(xy,xyz,camera,est,refine)
        if result is None:
            diagnostic.update(registered=False,reason="pnp_failed");outcomes.append(diagnostic);continue
        mask=np.asarray(result["inlier_mask"],dtype=bool)
        n=int(mask.sum());ratio=n/len(unique)
        diagnostic.update(inliers=n,inlier_ratio=ratio)
        if n<cfg["registration"]["min_inliers"] or ratio<cfg["registration"]["min_inlier_ratio"]:
            diagnostic.update(registered=False,reason="weak_pnp");outcomes.append(diagnostic);continue
        pose=result["cam_from_world"]
        cam_xyz=pose*xyz[mask]
        reproj=camera.img_from_cam(cam_xyz)
        errors=np.linalg.norm(np.asarray(reproj)-xy[mask],axis=1)
        if not np.isfinite(errors).all() or np.any(cam_xyz[:,2]<=0):
            diagnostic.update(registered=False,reason="invalid_projection");outcomes.append(diagnostic);continue
        cid=image.camera_id;rid=image.frame.rig_id
        if cid not in target.cameras:
            target.add_camera(pc.Camera(original_camera.todict()))
        if rid not in target.rigs:
            rig=pc.Rig(rig_id=rid);rig.add_ref_sensor(target.cameras[cid].sensor_id);target.add_rig(rig)
        keypoints=np.array([p.xy for p in image.points2D])
        add_view(target,iid,image.name,cid,rid,keypoints,pose)
        observations=[]
        for valid,(index,pid) in zip(mask,unique):
            if valid:
                target.add_observation(pid,pc.TrackElement(iid,index))
                observations.append([index,pid])
        diagnostic.update(registered=True,pose=pose.matrix().tolist(),center_source=image.projection_center().tolist(),
                          center_target=pose.inverse().translation.tolist(),rotation_source=image.cam_from_world().rotation.matrix().tolist(),
                          rotation_target=pose.rotation.matrix().tolist(),median_error_px=float(np.median(errors)),
                          observations=observations)
        accepted.append(diagnostic);outcomes.append(diagnostic)
        print("REGISTERED",image.name,n,"inliers, median error",np.median(errors),flush=True)
    db.close()
    for iid,center in initial_centers.items():
        if not np.array_equal(target.images[iid].projection_center(),center):
            raise RuntimeError("Target original camera moved during localization")
    save(out/"registrations.json",outcomes)
    extended=out/"reference_extended";extended.mkdir();target.write(extended)
    if len(accepted)<cfg["alignment"]["min_shared_views"]:
        raise RuntimeError(f"Only {len(accepted)} bridge views registered; inspect diagnostics")
    fit=[r for r in accepted if r["role"]=="fit"]
    hold=[r for r in accepted if r["role"]=="validation"]
    if len(fit)<3:
        raise RuntimeError("Insufficient independent fitting cameras")
    geography=load(artifact("geography")/"geography.json")
    reference_scale=geography["scenes"][cfg["reference"]]["scale"]
    threshold=cfg["alignment"]["center_threshold_m"]/reference_scale
    x=np.array([r["center_source"] for r in fit]);y=np.array([r["center_target"] for r in fit])
    S,mask,errors=robust(x,y,threshold,iterations=1500)
    validation=[]
    for r in hold:
        center_error=np.linalg.norm(apply(S,r["center_source"])-r["center_target"])*reference_scale
        scale=np.cbrt(np.linalg.det(S[:3,:3]));R=S[:3,:3]/scale
        expected=np.asarray(r["rotation_source"])@R.T
        angle=Rotation.from_matrix(np.asarray(r["rotation_target"])@expected.T).magnitude()*180/np.pi
        validation.append({"name":r["name"],"center_error_m":float(center_error),"rotation_error_deg":float(angle)})
    summary={"S_reference_from_source":S.tolist(),"source":cfg["source"],"reference":cfg["reference"],
             "registered":len(accepted),"fit_views":len(fit),"fit_inliers":int(mask.sum()),
             "fit_median_error_m":float(np.median(errors)*reference_scale),"holdout":validation,
             "note":"holdout cameras not used for this Sim3 fit; later native merger uses registered bridges"}
    save(out/"alignment.json",summary)
    if mask.sum()<3 or mask.mean()<0.6 or not validation or max(r["center_error_m"] for r in validation)>cfg["alignment"]["center_threshold_m"] or max(r["rotation_error_deg"] for r in validation)>cfg["alignment"]["max_rotation_error_deg"]:
        raise RuntimeError("Sim3 failed independent camera validation; no fusion submitted")
    record("localization",out,registered=len(accepted),fit_median_error_m=summary["fit_median_error_m"],
           max_holdout_center_error_m=max(v["center_error_m"] for v in validation))
