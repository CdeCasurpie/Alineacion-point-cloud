"""Operation-local namespaces. Never infer block membership from numeric ranges."""
from pathlib import Path
import sqlite3

import numpy as np
import pycolmap as pc

from alignexp import binary
from alignexp.colmap_io import add_camera_and_rig, add_view


def allocate(manifests):
    result={};next_image=next_camera=1
    for role,manifest in manifests.items():
        views=binary.images(manifest["sparse"]["model_uri"])
        cameras=binary.cameras(manifest["sparse"]["model_uri"])
        images={old:new for new,old in enumerate(sorted(views),next_image)}
        cams={old:new for new,old in enumerate(sorted(cameras),next_camera)}
        next_image+=len(images);next_camera+=len(cams)
        result[role]={"images":images,"cameras":cams,"views":views,"calibrations":cameras,
                      "names":{old:role+"/"+v["name"] for old,v in views.items()}}
    return result


def feature_database(manifests,namespaces,path):
    source=Path(next(iter(manifests.values()))["features"]["database_uri"])
    src=sqlite3.connect(source.as_uri()+"?mode=ro&immutable=1",uri=True)
    dst=sqlite3.connect(path);src.backup(dst);src.close()
    dst.execute("PRAGMA foreign_keys=OFF")
    tables=[r[0] for r in dst.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for name in tables:
        if not name.replace("_","").isalnum():
            raise RuntimeError("Unsafe database table")
        dst.execute(f'DELETE FROM "{name}"')
    for role,manifest in manifests.items():
        ns=namespaces[role]
        for cid,calibration in ns["calibrations"].items():
            new=ns["cameras"][cid]
            dst.execute("INSERT INTO cameras(camera_id,model,width,height,params,prior_focal_length) VALUES(?,?,?,?,?,?)",
                        (new,2,calibration["width"],calibration["height"],calibration["params"].astype("<f8").tobytes(),1))
            dst.execute("INSERT INTO rigs(rig_id,ref_sensor_id,ref_sensor_type) VALUES(?,?,0)",(new,new))
        src=sqlite3.connect(Path(manifest["features"]["database_uri"]).as_uri()+"?mode=ro&immutable=1",uri=True)
        for old,new in ns["images"].items():
            view=ns["views"][old];cid=ns["cameras"][view["camera_id"]]
            dst.execute("INSERT INTO images(image_id,name,camera_id) VALUES(?,?,?)",(new,ns["names"][old],cid))
            dst.execute("INSERT INTO frames(frame_id,rig_id) VALUES(?,?)",(new,cid))
            dst.execute("INSERT INTO frame_data(frame_id,data_id,sensor_id,sensor_type) VALUES(?,?,?,0)",(new,new,cid))
            for table in ("keypoints","descriptors"):
                row=src.execute(f"SELECT rows,cols,data FROM {table} WHERE image_id=?",(old,)).fetchone()
                dst.execute(f"INSERT INTO {table}(image_id,rows,cols,data) VALUES(?,?,?,?)",(new,*row))
        src.close()
    dst.commit()
    if dst.execute("PRAGMA foreign_key_check").fetchone():
        raise RuntimeError("Operation-local database referential failure")
    dst.close()


def reconstruction(manifest,ns):
    model=manifest["sparse"]["model_uri"]
    rec=pc.Reconstruction()
    for cid,camera in ns["calibrations"].items():
        new=ns["cameras"][cid]
        add_camera_and_rig(rec,new,new,camera["model"],camera["width"],camera["height"],camera["params"])
    for old,v in ns["views"].items():
        points=binary.point2d(model,v);new=ns["images"][old];cid=ns["cameras"][v["camera_id"]]
        add_view(rec,new,ns["names"][old],cid,cid,np.c_[points["x"],points["y"]],pc.Rigid3d(np.c_[v["R"],v["t"]]))
    pts=binary.points(model)
    for point,track in zip(pts,binary.point_tracks(model,pts["offset"],pts["length"])):
        pid=rec.add_point3D(point["xyz"],pc.Track([pc.TrackElement(ns["images"][int(i)],int(j)) for i,j in track]),point["color"])
        rec.points3D[pid].error=float(point["error"])
    return rec
