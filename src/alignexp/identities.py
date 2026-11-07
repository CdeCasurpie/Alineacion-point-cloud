"""One canonical namespace, preserving feature indices and intra-pair direction."""
import os
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pycolmap as pc

from . import binary
from .colmap_io import add_camera_and_rig, add_view
from .common import ROOT, config, record, require_job, save, stage_dir

M = 2147483647


def offsets(scene):
    return (0,0) if scene == "005" else (1000000,1000)


def copy_database(dstpath):
    source = ROOT / "inputs/005/database.db"
    src = sqlite3.connect(source.as_uri()+"?mode=ro&immutable=1",uri=True)
    dst = sqlite3.connect(dstpath)
    src.backup(dst)
    src.close()
    dst.execute("UPDATE images SET name='005/' || name")
    normalized = dst.execute("SELECT f.rig_id,d.data_id,d.sensor_id,d.sensor_type FROM frames f JOIN frame_data d USING(frame_id)").fetchall()
    if len(normalized) != dst.execute("SELECT COUNT(*) FROM images").fetchone()[0] or any(r[3]!=0 for r in normalized):
        raise RuntimeError("Only one camera image per frame is supported")
    dst.execute("DELETE FROM frame_data")
    dst.execute("DELETE FROM frames")
    for rid,iid,cid,kind in normalized:
        dst.execute("INSERT INTO frames(frame_id,rig_id) VALUES(?,?)",(iid,rid))
        dst.execute("INSERT INTO frame_data(frame_id,data_id,sensor_id,sensor_type) VALUES(?,?,?,?)",(iid,iid,cid,kind))
    dst.commit()
    src = sqlite3.connect((ROOT / "inputs/009/database.db").as_uri()+"?mode=ro&immutable=1",uri=True)
    frames = dict(src.execute("SELECT frame_id,data_id FROM frame_data WHERE sensor_type=0"))
    if len(frames) != src.execute("SELECT COUNT(*) FROM frames").fetchone()[0]:
        raise RuntimeError("Only one camera per frame supported")
    for table in ("cameras","rigs","rig_sensors","frames","images","frame_data","pose_priors","keypoints","descriptors","matches","two_view_geometries","pipeline_camera_groups"):
        cols = [r[1] for r in src.execute(f"PRAGMA table_info({table})")]
        if not cols:
            continue
        sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})"
        for row in src.execute(f"SELECT * FROM {table}"):
            values = dict(zip(cols,row))
            for key in ("camera_id","rig_id","ref_sensor_id","sensor_id"):
                if key in values:
                    values[key] += 1000
            if "image_id" in values:
                values["image_id"] += 1000000
            if "frame_id" in values:
                values["frame_id"] = frames[values["frame_id"]] + 1000000
            if "data_id" in values:
                values["data_id"] += 1000000
            if table == "images":
                values["name"] = "009/"+values["name"]
            if "pair_id" in values:
                a,b = divmod(values["pair_id"],M)
                values["pair_id"] = (a+1000000)*M + b+1000000
            dst.execute(sql,[values[c] for c in cols])
        dst.commit()
    src.close()
    if dst.execute("PRAGMA foreign_key_check").fetchone():
        raise RuntimeError("Global namespace FK failure")
    if dst.execute("SELECT COUNT(*) FROM images").fetchone()[0] != 1131:
        raise RuntimeError("Global image count mismatch")
    dst.close()


def remap_model(scene, out):
    source = ROOT / "inputs" / scene
    model = source / "model"
    views = binary.images(model)
    io,co = offsets(scene)
    db = sqlite3.connect((source/"database.db").as_uri()+"?mode=ro&immutable=1",uri=True)
    cameras = binary.cameras(model)
    rigs = db.execute("SELECT rig_id,ref_sensor_id,ref_sensor_type FROM rigs").fetchall()
    image_rig = dict(db.execute("SELECT d.data_id,f.rig_id FROM frame_data d JOIN frames f USING(frame_id) WHERE d.sensor_type=0"))
    if db.execute("SELECT COUNT(*) FROM rig_sensors").fetchone()[0] or any(r[2]!=0 for r in rigs):
        raise RuntimeError("Prototype supports trivial camera rigs only")
    db.close()
    rec = pc.Reconstruction()
    cam_rig = {sensor:rid for rid,sensor,_ in rigs}
    for cid,camera in cameras.items():
        add_camera_and_rig(rec,cid+co,cam_rig[cid]+co,camera["model"],camera["width"],camera["height"],camera["params"])
    for iid,v in views.items():
        p=binary.point2d(model,v)
        add_view(rec,iid+io,scene+"/"+v["name"],v["camera_id"]+co,image_rig[iid]+co,
                 np.c_[p["x"],p["y"]],pc.Rigid3d(np.c_[v["R"],v["t"]]))
    pts=binary.points(model)
    new_ids=np.empty(len(pts),dtype=np.uint64)
    for i,(point,observations) in enumerate(zip(pts,binary.point_tracks(model,pts["offset"],pts["length"]))):
        track=pc.Track([pc.TrackElement(int(iid)+io,int(index)) for iid,index in observations])
        pid=rec.add_point3D(point["xyz"],track,point["color"])
        rec.points3D[pid].error=float(point["error"])
        new_ids[i]=pid
        if i and i%100000==0:
            print(scene,"remapped points",i,flush=True)
    if rec.num_reg_images()!=len(views) or rec.num_points3D()!=len(pts):
        raise RuntimeError("Model remap lost entities")
    out.mkdir(parents=True)
    rec.write(out)
    np.savez(out.parent/f"point_ids_{scene}.npz",old=pts["id"],new=new_ids)
    for cid,camera in cameras.items():
        np.testing.assert_array_equal(rec.cameras[cid+co].params,camera["params"])
    # A pure namespace remap must retain the native camera projection quality.
    from .audit import reprojection_stats
    quality = reprojection_stats(rec)
    if quality["mean_error_px"] > 2:
        raise RuntimeError(f"Remapped model no longer preserves reprojection quality: {quality}")
    save(out.parent/f"remap_quality_{scene}.json",quality)
    return {"images":rec.num_reg_images(),"points":rec.num_points3D(),"image_offset":io,"camera_rig_offset":co}


def run():
    require_job()
    out=stage_dir("identities")
    local=Path(os.environ["TMPDIR"])/"canonical.db"
    copy_database(local)
    shutil.copy2(local,out/"database.db")
    summaries={}
    for scene in config()["sources"]:
        summaries[scene]=remap_model(scene,out/"models"/scene)
    save(out/"identities.json",summaries)
    image_root=out/"images"
    image_root.mkdir()
    for scene in config()["sources"]:
        (image_root/scene).symlink_to(os.path.relpath(ROOT/"inputs"/scene/"images",image_root),target_is_directory=True)
    record("identities",out,scenes=summaries,namespace="005 original IDs; 009 image +1000000, camera/rig +1000")
