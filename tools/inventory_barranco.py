"""Read-only inventory of the 25 map-labelled captures. Must run in Slurm."""
import argparse
import json
from pathlib import Path
import sqlite3
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from alignexp import binary
from blockalign.artifacts import require_job
from blockalign.contracts import from_colmap_run
from blockalign.storage import read,write


def inspect(run):
    quality=read(run/"quality.json")
    model=run/"colmap/sparse"/Path(quality["best"]["path"]).name
    for name in ("cameras.bin","images.bin","points3D.bin","rigs.bin","frames.bin"):
        if not (model/name).is_file():
            raise ValueError("Missing native model file: "+name)
    cameras=binary.cameras(model);views=binary.images(model)
    with (model/"points3D.bin").open("rb") as f:
        count=struct.unpack("<Q",f.read(8))[0]
    if len(views)<6 or count<1000:
        raise ValueError("Insufficient native reconstruction")
    missing=[v["name"] for v in views.values() if not (run/"frames"/v["name"]).is_file()]
    if missing:
        raise ValueError(f"Broken image references ({len(missing)}), example: {missing[0]}")
    dbpath=run/"colmap/database.db"
    if Path(str(dbpath)+"-wal").exists() and Path(str(dbpath)+"-wal").stat().st_size:
        raise ValueError("Active database WAL: immutable export needed")
    db=sqlite3.connect(dbpath.as_uri()+"?mode=ro&immutable=1",uri=True)
    try:
        if db.execute("SELECT COUNT(*) FROM rig_sensors").fetchone()[0]:
            raise ValueError("Nontrivial rig unsupported by current adapter")
        info={i:(name,cid) for i,name,cid in db.execute("SELECT image_id,name,camera_id FROM images")}
        if any(info.get(i)!=(v["name"],v["camera_id"]) for i,v in views.items()):
            raise ValueError("Model/database image identity mismatch")
        features=dict(db.execute("SELECT image_id,rows FROM descriptors WHERE cols=128"))
        if any(features.get(i)!=v["n2d"] for i,v in views.items()):
            raise ValueError("Descriptors missing or incompatible")
    finally:
        db.close()
    if not (run/"assets.json").exists() or not list((run/"telemetry").rglob("*_telemetry.csv")):
        raise ValueError("Missing frame catalog or geographic telemetry")
    dense=[]
    for name in ("mvs/scene_dense.ply","openmvs/scene_dense.ply","colmap/undistorted/fused.ply"):
        if (run/name).is_file():
            dense.append(str(run/name))
    return {"run":str(run),"model":str(model),"images":len(views),"points":count,
            "cameras":len(cameras),"quality_status":quality.get("status"),
            "reprojection_px":quality["best"].get("reprojection_error"),"dense_candidates":dense}


def main():
    require_job()
    p=argparse.ArgumentParser();p.add_argument("--pipeline",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();pipeline=Path(a.pipeline).resolve();output=Path(a.output).resolve()
    rows=[]
    for i in range(1,26):
        block=f"{i:03d}";folder=pipeline/"outputs"/block
        runs=[];preferred=None;errors=[]
        latest=pipeline/"datasets/_state"/block/"latest.json"
        if latest.exists():
            preferred=(pipeline/read(latest)["run"]).resolve()
        if folder.exists():
            runs=[p for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")]
        valid=[]
        for run in runs:
            try:
                valid.append(inspect(run))
            except Exception as error:
                errors.append({"run":str(run),"reason":str(error)})
        if valid:
            chosen=max(valid,key=lambda r:(Path(r["run"])==preferred,r["quality_status"]=="complete",r["images"],r["points"]))
            manifest=from_colmap_run(block,chosen["run"])
            write(output/"manifests"/(block+".json"),manifest.data)
            row={"block_id":block,"eligible":True,**chosen,"manifest":str(output/"manifests"/(block+".json")),"other_attempts":errors}
        else:
            row={"block_id":block,"eligible":False,"reason":"No usable supported reconstruction","attempts":errors}
        rows.append(row);print(json.dumps(row),flush=True)
    result={"schema_version":"1.0","blocks":rows,"eligible":[r["block_id"] for r in rows if r["eligible"]],
            "images":sum(r.get("images",0) for r in rows),"points":sum(r.get("points",0) for r in rows),
            "scope":"25 labels of mapa_provisional.png; metadata validation only, full audit is separate"}
    write(output/"inventory.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="blocks"},indent=2),flush=True)


if __name__=="__main__":
    main()
