import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from .common import artifact, config, record, require_job, stage_dir

M=2147483647


def run():
    require_job();cfg=config();out=stage_dir("matching");ids=artifact("identities");sel=artifact("selection")
    database=Path(os.environ["TMPDIR"])/"matching.db"
    shutil.copy2(ids/"database.db",database)
    command=["colmap","matches_importer","--database_path",str(database),"--match_list_path",str(sel/"pairs.txt"),
             "--match_type","pairs","--FeatureMatching.use_gpu","0","--FeatureMatching.num_threads",str(cfg["matching"]["threads"])]
    start=time.monotonic()
    with (out/"matching.log").open("w") as f:
        p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
    db=sqlite3.connect(database)
    names=dict(db.execute("SELECT name,image_id FROM images"))
    report=[]
    for line in (sel/"pairs.txt").read_text().splitlines():
        a,b=line.split();ia,ib=sorted((names[a],names[b]));pid=ia*M+ib
        raw=db.execute("SELECT rows FROM matches WHERE pair_id=?",(pid,)).fetchone()
        geometry=db.execute("SELECT rows,config FROM two_view_geometries WHERE pair_id=?",(pid,)).fetchone()
        report.append({"a":a,"b":b,"pair_id":pid,"raw":raw[0] if raw else None,
                       "inliers":geometry[0] if geometry else 0,"config":geometry[1] if geometry else None})
    db.close()
    from .common import save
    save(out/"pairs.json",report)
    verified=sum(r["inliers"]>=cfg["registration"]["min_pair_inliers"] for r in report)
    if p.returncode or verified==0 or any(r["raw"] is None for r in report):
        raise RuntimeError("Cross matching failed or produced no verified pairs; see diagnostic log")
    shutil.copy2(database,out/"database.db")
    record("matching",out,pairs=len(report),verified_pairs=verified,
           max_inliers=max(r["inliers"] for r in report),seconds=time.monotonic()-start,engine="CPU indexed SIFT; no re-extraction")
