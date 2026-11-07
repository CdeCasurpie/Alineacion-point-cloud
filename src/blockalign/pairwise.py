from collections import Counter, defaultdict
import gc
from pathlib import Path
import sqlite3

import numpy as np
import pycolmap as pc
from scipy.spatial.transform import Rotation
from shapely.geometry import shape
import shapely

from alignexp import binary
from alignexp.sim3 import apply, robust
from .artifacts import command, publish, verify_stamps, require_job
from .colmap_adapter import allocate, feature_database, reconstruction
from .contracts import digest
from .storage import read, write, lock


class InsufficientEvidence(RuntimeError):
    pass


def measurement_key(source,target,policy,engine):
    return digest({"source":source["fingerprint"],"target":target["fingerprint"],
                   "source_ref":source["ref"],"target_ref":target["ref"],"policy":policy,"engine":engine})


def support(audit_path,model,region,policy):
    with np.load(Path(audit_path)/"points.npz") as data:
        pts=data["points"];xyz=data["xyz"]
    good=shapely.intersects_xy(region,xyz[:,0],xyz[:,1])&(pts["length"]>=3)&(pts["error"]>=0)&(pts["error"]<2)
    idx=np.flatnonzero(good)
    idx=idx[np.argsort(pts["error"][idx]/np.sqrt(pts["length"][idx]))]
    _,unique=np.unique(np.floor(xyz[idx]/2).astype(np.int64),axis=0,return_index=True)
    idx=idx[np.sort(unique)]
    if len(idx)>policy["max_spatial_points"]:
        idx=idx[np.random.default_rng(policy["seed"]).choice(len(idx),policy["max_spatial_points"],replace=False)]
    scores=Counter();cells=defaultdict(set)
    for i,track in zip(idx,binary.point_tracks(model,pts["offset"][idx],pts["length"][idx])):
        cell=tuple(np.floor(xyz[i,:2]/10).astype(int))
        for iid in set(map(int,track[:,0])):
            scores[iid]+=1;cells[iid].add(cell)
    return scores,cells


def diverse(ranked,views,info,gap,limit):
    result=[]
    for iid in ranked:
        item=info[views[iid]["name"]]
        if any(item["clip"]==info[views[j]["name"]]["clip"] and abs(item["index"]-info[views[j]["name"]]["index"])<gap for j in result):
            continue
        result.append(iid)
        if len(result)>=limit:
            break
    return result


def select(source,target,source_audit,target_audit,policy,namespaces):
    region=shape(target["region"]).buffer(policy["selection_margin_m"])
    scores,source_cells=support(source_audit,source["manifest"]["sparse"]["model_uri"],region,policy)
    _,target_cells=support(target_audit,target["manifest"]["sparse"]["model_uri"],region,policy)
    source_ns=namespaces["source"];target_ns=namespaces["target"]
    source_info=read(Path(source_audit)/"catalog.json");target_info=read(Path(target_audit)/"catalog.json")
    ranked=[i for i,n in scores.most_common() if n>=policy["min_relevant_points"]]
    chosen=diverse(ranked,source_ns["views"],source_info,policy["min_frame_gap"],policy["max_source_views"])
    candidates=[];pairs=[]
    for rank,i in enumerate(chosen):
        expanded={(x+dx,y+dy) for x,y in source_cells[i] for dx in (-1,0,1) for dy in (-1,0,1)}
        target_rank=sorted(target_cells,key=lambda j:(-len(expanded&target_cells[j]),j))
        target_rank=[j for j in target_rank if expanded&target_cells[j]]
        targets=diverse(target_rank,target_ns["views"],target_info,policy["min_frame_gap"],policy["max_target_views"])
        pairs.extend((source_ns["names"][i],target_ns["names"][j]) for j in targets)
        candidates.append({"original_id":i,"image_id":source_ns["images"][i],"name":source_ns["names"][i],
                           "role":"validation" if rank%max(2,round(1/policy["validation_fraction"]))==0 else "fit"})
    if len(pairs)>policy["max_pairs"]:
        # Keep a fair prefix per candidate, rather than silently favoring names.
        pairs=pairs[:policy["max_pairs"]]
    if len(candidates)<policy["min_bridges"] or not pairs:
        raise InsufficientEvidence("Insufficient GIS/track support")
    return candidates,sorted(set(pairs))


def localize(source,target,db,candidates,policy):
    outcomes=[]
    est=pc.AbsolutePoseEstimationOptions();est.estimate_focal_length=False
    est.ransac.max_error=policy["max_reprojection_px"];est.ransac.max_num_trials=10000
    refine=pc.AbsolutePoseRefinementOptions();refine.refine_focal_length=False;refine.refine_extra_params=False
    for c in candidates:
        iid=c["image_id"];image=source.images[iid];votes=defaultdict(Counter)
        # Only this operation's cross pairs exist in the feature database.
        for pid,n,blob in db.execute("SELECT pair_id,rows,data FROM two_view_geometries WHERE rows>=?",(policy["min_pair_inliers"],)):
            a,b=divmod(pid,2147483647)
            if iid not in (a,b):
                continue
            tid=b if iid==a else a
            if tid not in target.images:
                continue
            matches=np.frombuffer(blob,dtype="<u4").reshape(-1,2)
            if iid==b:
                matches=matches[:,::-1]
            other=target.images[tid]
            for index,j in matches:
                own=image.point2D(int(index)).point3D_id;p=other.point2D(int(j)).point3D_id
                if own!=pc.INVALID_POINT3D_ID and p!=pc.INVALID_POINT3D_ID:
                    votes[int(index)][int(p)]+=1
        choices=[]
        for index,v in votes.items():
            ranked=v.most_common()
            if len(ranked)==1 or ranked[0][1]>ranked[1][1]:
                choices.append((index,ranked[0][0],ranked[0][1]))
        choices.sort(key=lambda r:(-r[2],r[0],r[1]));unique=[];used=set()
        for index,p,n in choices:
            if p not in used:
                unique.append((index,p));used.add(p)
        outcome={**c,"correspondences":len(unique),"registered":False}
        if len(unique)<policy["min_pnp_inliers"]:
            outcomes.append(outcome);continue
        xy=np.array([image.point2D(i).xy for i,p in unique]);xyz=np.array([target.points3D[p].xyz for i,p in unique])
        result=pc.estimate_and_refine_absolute_pose(xy,xyz,pc.Camera(image.camera.todict()),est,refine)
        if result is None:
            outcomes.append(outcome);continue
        mask=np.asarray(result["inlier_mask"],dtype=bool);count=int(mask.sum())
        outcome.update(inliers=count,inlier_ratio=count/len(unique))
        if count<policy["min_pnp_inliers"] or count/len(unique)<policy["min_pnp_ratio"]:
            outcomes.append(outcome);continue
        pose=result["cam_from_world"];camera_xyz=pose*xyz[mask]
        if np.any(camera_xyz[:,2]<=0):
            outcomes.append(outcome);continue
        errors=np.linalg.norm(np.asarray(image.camera.img_from_cam(camera_xyz))-xy[mask],axis=1)
        if not np.isfinite(errors).all():
            outcomes.append(outcome);continue
        outcome.update(registered=True,center_source=image.projection_center().tolist(),center_target=pose.inverse().translation.tolist(),
                       R_source=image.cam_from_world().rotation.matrix().tolist(),R_target=pose.rotation.matrix().tolist(),
                       pose=pose.matrix().tolist(),median_reprojection_px=float(np.median(errors)),
                       observations=[[index,p] for good,(index,p) in zip(mask,unique) if good])
        outcomes.append(outcome)
    return outcomes


def fit_measurement(outcomes,reference_scale,policy):
    accepted=[r for r in outcomes if r["registered"]]
    fit=[r for r in accepted if r["role"]=="fit"];validation=[r for r in accepted if r["role"]=="validation"]
    if len(accepted)<policy["min_bridges"] or len(fit)<3 or not validation:
        raise InsufficientEvidence("Insufficient fitting/validation bridge views")
    S,mask,errors=robust(np.array([r["center_source"] for r in fit]),np.array([r["center_target"] for r in fit]),
                         policy["max_center_error_m"]/reference_scale,iterations=1500,seed=policy["seed"])
    rotation=S[:3,:3]/np.cbrt(np.linalg.det(S[:3,:3]))
    holdout=[]
    for r in validation:
        center=np.linalg.norm(apply(S,r["center_source"])-r["center_target"])*reference_scale
        expected=np.asarray(r["R_source"])@rotation.T
        angle=Rotation.from_matrix(np.asarray(r["R_target"])@expected.T).magnitude()*180/np.pi
        holdout.append({"name":r["name"],"center_error_m":float(center),"rotation_error_deg":float(angle)})
    if mask.sum()<3 or mask.mean()<.6 or max(r["center_error_m"] for r in holdout)>policy["max_center_error_m"] or max(r["rotation_error_deg"] for r in holdout)>policy["max_rotation_error_deg"]:
        raise InsufficientEvidence("Similarity failed reserved-camera validation")
    return {"target_from_source":S.tolist(),"registered":len(accepted),"fit_views":len(fit),"fit_inliers":int(mask.sum()),
            "fit_median_error_m":float(np.median(errors)*reference_scale),"holdout":holdout,
            "validation_scope":"reserved cameras for Sim3 only; temporal neighbors may be correlated"}


def align_pair(workspace,source,target,source_audit,target_audit,policy,scratch,engine,threads):
    require_job()
    key=measurement_key(source,target,policy,engine)
    destination=Path(workspace)/"measurements"/key
    with lock(Path(workspace)/"locks"/("pair-"+key)):
        if (destination/"measurement.json").exists():
            return read(destination/"measurement.json"),True
        if (destination/"rejection.json").exists():
            raise InsufficientEvidence("Cached rejection: "+read(destination/"rejection.json")["reason"])
        local=Path(scratch)/("pair-"+key);local.mkdir()
        manifests={"source":source["manifest"],"target":target["manifest"]}
        ns=allocate(manifests)
        candidates,pairs=select(source,target,source_audit,target_audit,policy,ns)
        write(local/"candidates.json",candidates)
        pair_file=local/"pairs.txt";pair_file.write_text("".join(f"{a} {b}\n" for a,b in pairs))
        database_path=local/"features.db"
        feature_database(manifests,ns,database_path)
        command(["colmap","matches_importer","--database_path",database_path,"--match_list_path",pair_file,"--match_type","pairs",
                 "--FeatureMatching.use_gpu","0","--FeatureMatching.num_threads",threads],local/"matching.log")
        source_rec=reconstruction(manifests["source"],ns["source"])
        target_rec=reconstruction(manifests["target"],ns["target"])
        db=sqlite3.connect(database_path)
        outcomes=localize(source_rec,target_rec,db,candidates,policy)
        cross_pairs=db.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows>=?",(policy["min_pair_inliers"],)).fetchone()[0]
        db.close()
        write(local/"registrations.json",outcomes)
        reference_scale=float(np.cbrt(np.linalg.det(np.array(target["projected_from_sparse"])[:3,:3])))
        try:
            result=fit_measurement(outcomes,reference_scale,policy)
        except InsufficientEvidence:
            raise
        finally:
            del source_rec,target_rec;gc.collect()
        result.update(id=key,source=source["ref"],target=target["ref"],status="accepted",pairs=len(pairs),verified_pairs=cross_pairs,
                      input_fingerprints=[source["fingerprint"],target["fingerprint"]],engine=engine,policy=policy)
        write(local/"measurement.json",result)
        database_path.unlink()  # Do not persist duplicate multi-GB feature databases per edge.
        publish(local,destination)
        return result,False
