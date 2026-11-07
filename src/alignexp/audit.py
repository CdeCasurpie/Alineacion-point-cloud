"""Structural and observation-level audit. Counts alone cannot prove a merge."""
import numpy as np
import pycolmap as pc

from . import binary


def observation_preservation(original_path, merged_path):
    before=binary.images(original_path);after=binary.images(merged_path)
    old=lost=new=0
    for iid,image in before.items():
        if iid not in after or image["name"]!=after[iid]["name"] or image["n2d"]!=after[iid]["n2d"]:
            raise RuntimeError("Merged image identity/feature count changed")
        a=binary.point2d(original_path,image);b=binary.point2d(merged_path,after[iid])
        if not np.array_equal(a[["x","y"]],b[["x","y"]]):
            raise RuntimeError("Keypoint coordinates/order changed")
        valid_a=a["point"]>=0;valid_b=b["point"]>=0
        old+=int(valid_a.sum());lost+=int((valid_a & ~valid_b).sum());new+=int((~valid_a & valid_b).sum())
    return {"original_observations":old,"lost_observations":lost,"new_observations":new,
            "loss_fraction":lost/max(old,1)}


def reciprocal_tracks(rec):
    total=0;duplicates=0;exact_duplicates=0
    for pid,point in rec.points3D.items():
        seen=set();elements=set()
        for element in point.track.elements:
            pair=(element.image_id,element.point2D_idx)
            if pair in elements:
                exact_duplicates+=1
            elements.add(pair)
            if element.image_id in seen:
                duplicates+=1
            seen.add(element.image_id)
            if element.image_id not in rec.images or rec.images[element.image_id].point2D(element.point2D_idx).point3D_id!=pid:
                raise RuntimeError("Broken reciprocal track reference")
            total+=1
    return {"observations":total,"duplicate_image_observations":duplicates,"exact_duplicate_elements":exact_duplicates}


def input_image_multiplicity(model):
    points=binary.points(model)
    return sum(len(track)-len(set(map(int,track[:,0]))) for track in
               binary.point_tracks(model,points["offset"],points["length"]))


def reprojection_stats(rec, update_errors=False, point_ids=None):
    ids=np.array(list(rec.points3D.keys()),dtype=np.int64)
    if not len(ids) or ids.max()>10000000:
        raise RuntimeError("Unexpected point namespace")
    sums=np.zeros(ids.max()+1);counts=np.zeros(ids.max()+1,dtype=np.int64)
    all_errors=[]
    for iid in rec.reg_image_ids():
        image=rec.images[iid]
        indices=image.get_observation_point2D_idxs()
        if not len(indices):
            continue
        pids=np.array([image.point2D(int(i)).point3D_id for i in indices],dtype=np.int64)
        if point_ids is not None:
            mask=np.fromiter((int(pid) in point_ids for pid in pids),dtype=bool,count=len(pids))
            pids=pids[mask]
            indices=np.asarray(indices)[mask]
            if not len(pids):
                continue
        xy=np.array([image.point2D(int(i)).xy for i in indices])
        xyz=np.array([rec.points3D[int(pid)].xyz for pid in pids])
        camera_xyz=image.cam_from_world()*xyz
        projected=image.camera.img_from_cam(camera_xyz)
        errors=np.linalg.norm(np.asarray(projected)-xy,axis=1)
        if not np.isfinite(errors).all():
            raise RuntimeError("Nonfinite reprojection")
        np.add.at(sums,pids,errors);np.add.at(counts,pids,1)
        all_errors.append(errors)
    if update_errors:
        for pid in ids:
            rec.points3D[int(pid)].error=float(sums[pid]/counts[pid]) if counts[pid] else -1.
    if not all_errors:
        raise RuntimeError("No observations available for reprojection evaluation")
    errors=np.concatenate(all_errors)
    return {"images":rec.num_reg_images(),"points":rec.num_points3D() if point_ids is None else len(point_ids),"observations":int(len(errors)),
            "mean_error_px":float(errors.mean()),"median_error_px":float(np.median(errors)),
            "p90_error_px":float(np.quantile(errors,.9)),"track_length":float(counts[ids if point_ids is None else list(point_ids)].mean())}
