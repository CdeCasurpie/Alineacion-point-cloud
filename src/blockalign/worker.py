"""Pinned-request Slurm worker. Does not consult mutable 'latest stage' files."""
from pathlib import Path
import argparse
import json
import os
import shutil
import time
import traceback

from .artifacts import require_job, publish, command, sha, verify_stamps
from .audit import audit_block
from .contracts import digest
from .graph import proposed_pairs, solve
from .pairwise import align_pair, InsufficientEvidence, measurement_key
from .storage import read, write, update_run, lock


def add(workspace,request,scratch):
    plan=request["plan"];audits={};paths={};audit_hits=0
    for ref in plan["blocks"]:
        verify_stamps(plan.get("registration_stamps",{}).get(ref,[]))
        audits[ref],paths[ref],hit=audit_block(workspace,ref,plan["manifests"][ref],scratch)
        audit_hits+=hit
        print("AUDIT",ref.split("@")[0],"cached",hit,flush=True)
    if plan["base_map"]:
        base=read(workspace/"maps"/plan["base_map"]/"map.json")
        edges=[e for e in base["measurements"] if e["policy"]==plan["policy"] and e["engine"]==request["pair_engine"]]
    else:
        edges=[]
    requested=plan["pairs"] if plan["pairs"] is not None else proposed_pairs(audits,plan["policy"],plan["reference"])
    existing={frozenset((e["source"],e["target"])) for e in edges}
    reused=len(edges);computed=0;rejected=[]
    for source,target in requested:
        if frozenset((source,target)) in existing:
            continue
        print("PAIR",source.split("@")[0],"->",target.split("@")[0],flush=True)
        try:
            measurement,hit=align_pair(workspace,audits[source],audits[target],paths[source],paths[target],plan["policy"],
                                       scratch,request["pair_engine"],int(os.environ.get("SLURM_CPUS_PER_TASK","1")))
        except InsufficientEvidence as error:
            rejected.append({"source":source,"target":target,"reason":str(error)})
            key=measurement_key(audits[source],audits[target],plan["policy"],request["pair_engine"])
            destination=workspace/"measurements"/key
            local=Path(scratch)/("pair-"+key)
            if local.exists() and not destination.exists():
                (local/"features.db").unlink(missing_ok=True)
                write(local/"rejection.json",rejected[-1])
                publish(local,destination)
            continue
        edges.append(measurement);existing.add(frozenset((source,target)));reused+=hit;computed+=not hit
    solution=solve(audits,edges,plan["reference"],plan["world_origin"])
    if not solution["optimizer"].get("success",True):
        raise RuntimeError("Graph optimizer did not converge: "+str(solution["optimizer"]))
    body={"schema_version":"1.0","blocks":plan["blocks"],"reference":plan["reference"],"policy":plan["policy"],
          "crs":"EPSG:32718","world_frame":"local UTM18S meters, add world_origin for projected coordinates",
          "geometry_variant":"placement_original_blocks","measurements":edges,**solution,
          "status":"partial" if solution["disconnected"] else "complete", "pair_rejections":rejected,
          "manifests":plan["manifests"],"sparse_fingerprints":{r:a["sparse_fingerprint"] for r,a in audits.items()},
          "input_fingerprints":{r:a["fingerprint"] for r,a in audits.items()},
          "metrics":{"audit_cache_hits":audit_hits,"pair_cache_hits":reused,"pairs_computed":computed,"pairs_rejected":len(rejected)}}
    key=digest(body);body["id"]=key
    local=Path(scratch)/"map";local.mkdir();write(local/"map.json",body)
    publish(local,workspace/"maps"/key)
    return {"map_revision":key,"status":body["status"],"metrics":body["metrics"],"disconnected":body["disconnected"]}


def execute(workspace,run_id):
    require_job();workspace=Path(workspace).resolve()
    request=read(workspace/"runs"/run_id/"request.json")
    scratch=Path(os.environ["TMPDIR"])
    with lock(workspace/"locks"/("run-"+run_id)):
        update_run(workspace,run_id,state="running",job_id=os.environ["SLURM_JOB_ID"])
        start=time.monotonic()
        try:
            import pycolmap
            if pycolmap.__version__!="3.13.0":
                raise RuntimeError("Worker requires PyCOLMAP 3.13.0")
            command(["colmap","-h"],scratch/"colmap-version.log")
            if "COLMAP 3.13.0" not in (scratch/"colmap-version.log").read_text():
                raise RuntimeError("Native COLMAP must match PyCOLMAP 3.13.0")
            container=sha(request["identity"]["image"])
            request["engine"]=digest({"code":request["identity"]["code"],"container":container})
            # Export/CLI changes must not invalidate expensive pair measurements.
            source=Path(__file__).resolve().parent
            kernels={str(p.relative_to(source.parent)):sha(p) for p in
                     [source/"pairwise.py",source/"colmap_adapter.py",source/"contracts.py",
                      source.parent/"alignexp/binary.py",source.parent/"alignexp/sim3.py",source.parent/"alignexp/colmap_io.py"]}
            request["pair_engine"]=digest({"kernels":kernels,"container":container,"adapter":"colmap313-sift"})
            if request["plan"]["operation"]=="add":
                result=add(workspace,request,scratch)
            elif request["plan"]["operation"]=="export":
                from .exports import export
                result=export(workspace,request,scratch)
            elif request["plan"]["operation"]=="correction":
                from .correction import correct
                result=correct(workspace,request,scratch)
            elif request["plan"]["operation"]=="campaign":
                from .campaign import run
                result=run(workspace,request,scratch)
            else:
                raise RuntimeError("Unsupported operation")
            result.update(run_id=run_id,job_id=os.environ["SLURM_JOB_ID"],elapsed_seconds=time.monotonic()-start,engine=request["engine"],
                          execution=request.get("execution",{}),hostname=os.uname().nodename)
            result_path=workspace/"runs"/run_id/"result.json"
            write(result_path,result)
            update_run(workspace,run_id,state="succeeded",result=str(result_path.relative_to(workspace)),error=None)
            print(json.dumps(result,indent=2),flush=True)
        except Exception as error:
            # Retain diagnostic files from /local before the job's scratch cleanup.
            diagnostics=workspace/"runs"/run_id/"diagnostics"
            diagnostics.mkdir(exist_ok=True)
            for p in scratch.rglob("*"):
                if p.is_file() and p.suffix in (".json",".log",".txt") and "src" not in p.relative_to(scratch).parts:
                    target=diagnostics/p.relative_to(scratch);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
            update_run(workspace,run_id,state="failed",error=f"{type(error).__name__}: {error}")
            raise


def main():
    p=argparse.ArgumentParser();p.add_argument("workspace");p.add_argument("run_id");a=p.parse_args()
    execute(a.workspace,a.run_id)


if __name__=="__main__":
    main()
