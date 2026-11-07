from pathlib import Path
import csv
import sqlite3

import numpy as np
from pyproj import Transformer
from shapely.geometry import MultiPoint, shape, mapping
from shapely.ops import transform, unary_union

from alignexp import binary
from alignexp.geography import seconds
from alignexp.sim3 import apply, robust
from .artifacts import sha, stamp, verify_stamps, publish, require_job
from .contracts import digest
from .storage import read, write, lock


MODEL_FILES=("cameras.bin","images.bin","points3D.bin","rigs.bin","frames.bin")


def catalog(manifest):
    data=read(manifest["images"]["catalog_uri"])
    if manifest["images"]["catalog_format"]=="image_info_v1":
        return data
    return {i["relative"]:{"clip":Path(a["relative"]).stem,"index":i["index"],"time_s":i["time_s"]}
            for a in data for i in a["images"]}


def geographic_references(manifest,images):
    geo=manifest["geography"]
    if geo.get("references_uri"):
        with Path(geo["references_uri"]).open() as f:
            rows=list(csv.DictReader(f))
        return {r["name"]:np.array([float(r[k]) for k in ("longitude","latitude","altitude")]) for r in rows}
    tracks={}
    for p in Path(geo["telemetry_uri"]).rglob("*_telemetry.csv"):
        rows=[]
        with p.open() as f:
            for r in csv.DictReader(f):
                try:
                    row=[seconds(r["timestamp"]),float(r["longitude"]),float(r["latitude"]),float(r["abs_alt"])]
                except (ValueError,KeyError):
                    continue
                if np.isfinite(row).all() and -180<=row[1]<=180 and -90<=row[2]<=90 and row[1:3]!=[0,0]:
                    rows.append(row)
        values=np.asarray(sorted(rows))
        if len(values):
            _,idx=np.unique(values[:,0],return_index=True);values=values[idx]
        tracks[p.name.removesuffix("_telemetry.csv")]=values
    refs={}
    for name,info in images.items():
        values=tracks.get(info["clip"])
        if values is None or not len(values):
            continue
        t=float(info["time_s"]);times=values[:,0];i=np.searchsorted(times,t)
        if i<len(values) and abs(times[i]-t)<1e-6:
            refs[name]=values[i,1:]
        elif 0<i<len(values) and max(t-times[i-1],times[i]-t)<=1:
            a=(t-times[i-1])/(times[i]-times[i-1]);refs[name]=(1-a)*values[i-1,1:]+a*values[i,1:]
    return refs


def audit_block(workspace,ref,manifest,scratch):
    require_job()
    kernels={p.name:sha(p) for p in (Path(__file__),Path(binary.__file__),Path(binary.__file__).with_name("sim3.py"))}
    key=digest({"ref":ref,"adapter":"colmap313-sift-v2","kernels":kernels,"numpy":np.__version__})
    destination=Path(workspace)/"audits"/key
    with lock(Path(workspace)/"locks"/("audit-"+key)):
        if (destination/"summary.json").exists():
            summary=read(destination/"summary.json");verify_stamps(summary["input_stamps"])
            return summary,destination,True
        local=Path(scratch)/("audit-"+key);local.mkdir()
        model=Path(manifest["sparse"]["model_uri"])
        cameras=binary.cameras(model);views=binary.images(model)
        points=binary.points(model);info=catalog(manifest)
        if not views or not len(points) or len({v["name"] for v in views.values()})!=len(views):
            raise RuntimeError("Empty or ambiguous reconstruction")
        db=sqlite3.connect(Path(manifest["features"]["database_uri"]).as_uri()+"?mode=ro&immutable=1",uri=True)
        if Path(manifest["features"]["database_uri"]+"-wal").exists() and Path(manifest["features"]["database_uri"]+"-wal").stat().st_size:
            raise RuntimeError("Source database has active WAL; create a consistent immutable export first")
        if db.execute("SELECT COUNT(*) FROM rig_sensors").fetchone()[0]:
            raise RuntimeError("Adapter currently supports single-camera reference rigs only")
        db_images=dict((i,(n,c)) for i,n,c in db.execute("SELECT image_id,name,camera_id FROM images"))
        features=dict((i,(n,c,d,l)) for i,n,c,d,l in db.execute(
            "SELECT k.image_id,k.rows,k.cols,d.rows,length(d.data) FROM keypoints k JOIN descriptors d USING(image_id)"))
        image_rig=dict(db.execute("SELECT d.data_id,f.rig_id FROM frame_data d JOIN frames f USING(frame_id) WHERE sensor_type=0"))
        if not set(views)<=set(image_rig):
            raise RuntimeError("Invalid single-image frame mapping")
        checksums={};stamps=[]
        paths=[model/name for name in MODEL_FILES]+[Path(manifest["features"]["database_uri"]),Path(manifest["images"]["catalog_uri"])]
        for optional in ("region_uri","references_uri"):
            if manifest["geography"].get(optional):
                paths.append(Path(manifest["geography"][optional]))
        if manifest["geography"].get("telemetry_uri"):
            paths+=sorted(Path(manifest["geography"]["telemetry_uri"]).rglob("*.csv"))
        for p in paths:
            stamps.append(stamp(p));checksums[str(p)]=sha(p)
        for iid,v in views.items():
            if db_images.get(iid)!=(v["name"],v["camera_id"]) or v["name"] not in info:
                raise RuntimeError("Model/database/catalog identities disagree")
            n,cols,dn,size=features.get(iid,(0,0,0,0))
            if n!=v["n2d"] or dn!=n or cols not in (2,4,6) or size!=128*n:
                raise RuntimeError("Model feature indexing or SIFT shape mismatch")
            image=Path(manifest["images"]["root_uri"])/v["name"]
            if Path(v["name"]).is_absolute() or ".." in Path(v["name"]).parts or any(c.isspace() for c in v["name"]):
                raise RuntimeError("Unsafe image relative path")
            stamps.append(stamp(image));checksums[str(image)]=sha(image)
        db.close()
        refs=geographic_references(manifest,info)
        valid=[v for v in views.values() if v["name"] in refs and np.isfinite(refs[v["name"]]).all()]
        if len(valid)<6:
            raise RuntimeError("Insufficient geographic camera references")
        raw=np.array([refs[v["name"]] for v in valid])
        projection=Transformer.from_crs("EPSG:4326",manifest["geography"]["crs"],always_xy=True)
        e,n=projection.transform(raw[:,0],raw[:,1]);world=np.c_[e,n,raw[:,2]]
        origin=np.median(world,axis=0)
        if np.ptp(world[:,:2],axis=0).max()<15:
            raise RuntimeError("Geographic reference trajectory is degenerate")
        S,mask,errors=robust(np.array([v["center"] for v in valid]),world-origin,8.,z_weight=.3)
        if mask.mean()<.7:
            raise RuntimeError("Unreliable geographic prior")
        S[:3,3]+=origin
        xyz=apply(S,points["xyz"])
        quality=(points["length"]>=3)&(points["error"]>=0)&(points["error"]<2)
        sample=xyz[quality][::max(1,int(quality.sum())//15000)]
        coverage=MultiPoint(sample[:,:2]).convex_hull
        region=coverage
        if manifest["geography"].get("region_uri"):
            data=read(manifest["geography"]["region_uri"])
            selected=[f for f in data["features"] if not manifest["geography"].get("region_ids") or f["properties"].get("id") in manifest["geography"]["region_ids"]]
            if not selected:
                raise RuntimeError("Declared GIS regions not found")
            region=unary_union([transform(projection.transform,shape(f["geometry"])) for f in selected])
        if not region.is_valid or region.area<=0:
            raise RuntimeError("Invalid region")
        np.savez(local/"points.npz",points=points,xyz=xyz)
        write(local/"catalog.json",info)
        sparse_hash=digest({name:checksums[str(model/name)] for name in MODEL_FILES})
        summary={"ref":ref,"block_id":manifest["block_id"],"manifest":manifest,"fingerprint":digest(checksums),
                 "sparse_fingerprint":sparse_hash,"input_stamps":stamps,"images":len(views),"points":len(points),
                 "origin":origin.tolist(),"projected_from_sparse":S.tolist(),"coverage":mapping(coverage),"region":mapping(region),
                 "region_kind":"declared_gis" if manifest["geography"].get("region_uri") else "reconstructed_coverage_not_cadastral",
                 "gps_inliers":int(mask.sum()),"gps_median_weighted_m":float(np.median(errors)),
                 "vertical_datum":"reference altitude; DJI abs_alt when imported from telemetry, not independently certified"}
        write(local/"summary.json",summary);publish(local,destination)
        return summary,destination,False
