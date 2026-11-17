"""No production inputs: synthetic cameras/features test the real native path."""
import json
from pathlib import Path
import sqlite3

import numpy as np
import pycolmap as pc
import pytest
from scipy.spatial.transform import Rotation

from alignexp.colmap_io import add_camera_and_rig, add_view
from blockalign.manual import (ManualError, audit, constrained_ba, measure, normalize,
                               propose, solve, stamps, transform)


def make_scene(root, *, incoming=False, noise=False):
    root = Path(root); root.mkdir()
    rng = np.random.default_rng(11)
    xyz = np.c_[rng.uniform(-2, 2, (160, 2)), rng.uniform(0, 2, 160)]
    matrix = np.eye(4)
    if incoming:
        matrix[:3, :3] = 1.3*Rotation.from_euler("z", 12, degrees=True).as_matrix()
        matrix[:3, 3] = [1, -2, .5]
    inv = np.linalg.inv(matrix)
    rr = matrix[:3, :3]/np.cbrt(np.linalg.det(matrix[:3, :3]))
    rec = pc.Reconstruction()
    add_camera_and_rig(rec, 1, 1, "SIMPLE_RADIAL", 1000, 800, [700, 500, 400, 0])
    descriptors = rng.integers(0, 255, (160, 128), dtype=np.uint8)
    # Native SIFT descriptor vectors are positive L2 normalized uint8.
    descriptors = np.round(descriptors/np.linalg.norm(descriptors, axis=1)[:, None]*512).astype(np.uint8)
    handle = pc.Database.open(str(root/"features.db")); handle.close()
    frames = root/"frames"; frames.mkdir()
    from PIL import Image
    with sqlite3.connect(root/"features.db") as db:
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("INSERT INTO cameras VALUES(?,?,?,?,?,?)",
                   (1, 2, 1000, 800, np.array([700,500,400,0], dtype="<f8").tobytes(), 1))
        for i, (x, y) in enumerate([(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(1,1)], 1):
            center_world = np.array([x,y,-8.])
            center = transform(inv, center_world)
            r = rr
            pose = pc.Rigid3d(pc.Rotation3d(r), -r@center)
            camera_xyz = xyz-center_world
            xy = np.c_[camera_xyz[:,0]/camera_xyz[:,2]*700+500,
                       camera_xyz[:,1]/camera_xyz[:,2]*700+400]
            add_view(rec, i, f"{i:03}.jpg", 1, 1, xy, pose)
            image = Image.new("RGB", (1000,800), (50+i,80,100));image.save(frames/f"{i:03}.jpg")
            db.execute("INSERT INTO images(image_id,name,camera_id) VALUES(?,?,?)", (i,f"{i:03}.jpg",1))
            kp = np.c_[xy,np.ones(160),np.zeros(160)].astype("<f4")
            db.execute("INSERT INTO keypoints VALUES(?,?,?,?)", (i,160,4,kp.tobytes()))
            db.execute("INSERT INTO descriptors VALUES(?,?,?,?)", (i,160,128,descriptors.tobytes()))
        db.commit()
    db.close()
    local = transform(inv, xyz)
    if noise:
        local += rng.normal(0,.002,local.shape)
    for p, point in enumerate(local):
        rec.add_point3D(point, pc.Track([pc.TrackElement(i,p) for i in range(1,9)]), np.array([100,120,140],dtype=np.uint8))
    model = root/"model";model.mkdir();rec.write(model)
    manifest = {"sparse":{"model_uri":str(model)}, "features":{"database_uri":str(root/"features.db"),"family":"sift"},
                "images":{"root_uri":str(frames)}}
    return manifest, matrix


def evidence(matrix, n=8):
    rng = np.random.default_rng(3)
    return [{"registered":True, "role":"fit" if i<6 else "validation", "original_id":i,
             "anchor":"A" if i%2 else "B", "direction":"forward",
             "local_center":p.tolist(), "world_center":transform(matrix,p).tolist(),
             "local_R":np.eye(3).tolist(), "world_R":Rotation.from_matrix(matrix[:3,:3]/1.2).inv().as_matrix().tolist()}
            for i,p in enumerate(rng.normal(size=(n,3)))]


def test_joint_fixed_anchors_recovers_scale_rotation_translation():
    m=np.eye(4);m[:3,:3]=1.2*Rotation.from_euler("xyz",[.1,.2,.3]).as_matrix();m[:3,3]=[5,6,7]
    rows=evidence(m);before=json.dumps(rows)
    prior=m.copy();prior[:3,3]+=[.8,-.7,.4]
    result=solve(rows,prior)
    assert result["validated"] and result["anchors_fixed"]
    np.testing.assert_allclose(result["matrix"],m,atol=1e-7)
    assert json.dumps(rows)==before


def test_validation_and_degenerate_selection_are_not_silently_replaced():
    m=np.eye(4);m[:3,:3]*=1.2
    rows=evidence(m)
    rows[-1]["world_center"][0]+=20
    assert not solve(rows,m)["validated"]
    for i,row in enumerate(rows):
        row["local_center"]=[i,0,0]
    with pytest.raises(ManualError,match="Degenerate"):
        solve(rows,m)
    with pytest.raises(ManualError,match="Insufficient"):
        solve(rows[:3],m)


def test_reverse_normalization_and_prior_candidate_query_preservation():
    a=np.eye(4);a[:3,3]=[4,5,6]
    r={"registered":True,"center_source":[1,2,3],"center_target":[7,8,9],
       "R_source":np.eye(3).tolist(),"R_target":np.eye(3).tolist()}
    normalized=normalize(r,a,"reverse","A")
    assert normalized["local_center"]==[7,8,9]
    assert normalized["world_center"]==[5,7,9]
    views={1:{"center":[0,0,0],"R":np.eye(3)},2:{"center":[1,1,0],"R":np.eye(3)}}
    result=propose(views,views,[2],np.eye(4),np.eye(4),limit=1)
    assert result[0]["query"]==2


def test_real_matching_pnp_joint_solve_and_constrained_ba(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/data/barranco/runtime/alignment313/bin:"+__import__("os").environ["PATH"])
    source, truth=make_scene(tmp_path/"source",incoming=True,noise=True)
    target,_=make_scene(tmp_path/"anchor")
    anchor_before=stamps(target["sparse"]["model_uri"]);source_before=stamps(source["sparse"]["model_uri"])
    selected={"views":[{"id":i,"role":"fit" if i<7 else "validation"} for i in range(1,9)],
              "pairs":[{"query":i,"support":i%8+1} for i in range(1,9)]}
    rows=measure(source,target,selected,tmp_path/"measure")
    assert len(rows)==8 and sum(r["registered"] for r in rows)>=6
    outcomes=[normalize(r,np.eye(4),"forward","A") for r in rows]
    result=solve(outcomes,truth)
    assert result["validated"]
    np.testing.assert_allclose(result["matrix"],truth,atol=.01)
    ba=constrained_ba(source,result["matrix"],outcomes,{"A":{**target,"matrix":np.eye(4).tolist()}},
                      tmp_path/"derived",max_iterations=20)
    assert ba["anchors_fixed"] and ba["dense_requires_regeneration"]
    assert ba["after_reprojection_px"]<=ba["before_reprojection_px"]*1.05
    assert stamps(target["sparse"]["model_uri"])==anchor_before
    assert stamps(source["sparse"]["model_uri"])==source_before
    derived=pc.Reconstruction(str(tmp_path/"derived"))
    assert derived.num_reg_images()==8
    # Held-out poses cannot move.
    original=pc.Reconstruction(source["sparse"]["model_uri"])
    original.transform(pc.Sim3d(np.array(result["matrix"])[:3,:]))
    for i in (7,8):
        np.testing.assert_allclose(derived.images[i].cam_from_world().matrix(),original.images[i].cam_from_world().matrix(),atol=1e-12,rtol=0)


def test_audit_rejects_feature_index_mismatch_and_active_wal(tmp_path):
    manifest,_=make_scene(tmp_path/"input")
    with sqlite3.connect(manifest["features"]["database_uri"]) as db:
        db.execute("UPDATE keypoints SET rows=159 WHERE image_id=1");db.commit()
    with pytest.raises(ManualError,match="indexing"):
        audit(manifest,{1})
    wal=Path(manifest["features"]["database_uri"]+"-wal");wal.write_bytes(b"active")
    with pytest.raises(ManualError,match="active WAL"):
        audit(manifest,{1})


def test_native_partial_multianchor_forward_and_reverse(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/data/barranco/runtime/alignment313/bin:"+__import__("os").environ["PATH"])
    incoming,truth=make_scene(tmp_path/"new",incoming=True)
    a,_=make_scene(tmp_path/"a")
    b,b_matrix=make_scene(tmp_path/"b",incoming=True)
    def selected(ids):
        return {"views":[{"id":i,"role":"validation" if i in (7,8) else "fit"} for i in ids],
                "pairs":[{"query":i,"support":i%8+1} for i in ids]}
    forward=measure(incoming,a,selected([1,2,3,7]),tmp_path/"f")
    reverse=measure(b,incoming,selected([4,5,6,8]),tmp_path/"r")
    outcomes=[normalize(r,np.eye(4),"forward","A") for r in forward]
    outcomes += [normalize(r,b_matrix,"reverse","B") for r in reverse]
    result=solve(outcomes,truth)
    assert result["validated"]
    assert set(result["by_anchor"])=={"A","B"}
    assert result["fit_views"]==6
    np.testing.assert_allclose(result["matrix"],truth,atol=.01)
