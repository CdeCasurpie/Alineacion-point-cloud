"""Small control-plane API; all array processing is isolated in workers."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import uuid
import importlib.util

from .contracts import BlockManifest, ContractError, Policy, digest
from .storage import database, initialize, lock, read, write


class Workspace:
    def __init__(self,path):
        self.root=Path(path).expanduser().resolve()
        initialize(self.root)

    @classmethod
    def open(cls,path):
        return cls(path)

    def register_block(self,manifest):
        if isinstance(manifest,BlockManifest):
            block=manifest
        elif isinstance(manifest,(str,Path)):
            p=Path(manifest).resolve();block=BlockManifest.parse(read(p),p.parent)
        else:
            block=BlockManifest.parse(manifest)
        from .artifacts import stamp
        model=Path(block.data["sparse"]["model_uri"])
        files=[model/n for n in ("cameras.bin","images.bin","points3D.bin","rigs.bin","frames.bin")]
        files += [Path(block.data["features"]["database_uri"]),Path(block.data["images"]["catalog_uri"])]
        source_stamps=[stamp(p) for p in files if p.exists()]
        with database(self.root) as db:
            db.execute("INSERT OR IGNORE INTO blocks VALUES(?,?,?)",(block.ref,block.data["block_id"],json.dumps(block.data)))
            existing=db.execute("SELECT source_stamps FROM registrations WHERE ref=?",(block.ref,)).fetchone()
            if existing and json.loads(existing[0])!=source_stamps:
                raise ContractError("Registered input metadata changed; publish a new immutable block revision")
            db.execute("INSERT OR IGNORE INTO registrations VALUES(?,?)",(block.ref,json.dumps(source_stamps)))
        db.close()
        return block.ref

    def block(self,ref):
        db=database(self.root)
        row=db.execute("SELECT manifest FROM blocks WHERE ref=?",(ref,)).fetchone();db.close()
        if row is None:
            raise ContractError(f"Unregistered block: {ref}")
        return json.loads(row[0])

    def plan_add(self,blocks,base_map=None,reference=None,policy=None,pairs=None):
        base=read(self.root/"maps"/base_map/"map.json") if base_map else None
        policy=Policy.parse(policy if policy is not None else (base.get("policy") if base else None)).to_dict()
        refs=list(dict.fromkeys((base["blocks"] if base else [])+list(blocks)))
        if not refs:
            raise ContractError("At least one block is required")
        manifests={r:self.block(r) for r in refs}
        db=database(self.root)
        registration_stamps={r:json.loads(row[0]) if (row:=db.execute("SELECT source_stamps FROM registrations WHERE ref=?",(r,)).fetchone()) else [] for r in refs}
        db.close()
        ids=[m["block_id"] for m in manifests.values()]
        if len(set(ids))!=len(ids):
            raise ContractError("One revision per block per map; replacing a revision requires a new branch without the old revision")
        reference=reference or (base["reference"] if base else sorted(refs)[0])
        if reference not in refs or (base and reference!=base["reference"]):
            raise ContractError("Invalid reference or attempted incremental frame change")
        if pairs is not None:
            for pair in pairs:
                if len(pair)!=2 or pair[0]==pair[1] or any(r not in refs for r in pair):
                    raise ContractError("Explicit pairs must contain two distinct registered revisions")
        plan={"schema_version":"1.0","operation":"add","blocks":refs,"manifests":manifests,
              "base_map":base_map,"reference":reference,"policy":policy,"pairs":pairs,
              "registration_stamps":registration_stamps,
              "world_origin":base["world_origin"] if base else None}
        return self._plan(plan)

    def plan_export(self,map_revision,product="atlas",block_assets=None,color_mode="original"):
        if product not in ("atlas","transformed_ply","combined_ply"):
            raise ContractError("Supported exports: atlas, transformed_ply, combined_ply")
        if color_mode not in ("original","by_block"):
            raise ContractError("Supported colors: original or by_block")
        mapping=read(self.root/"maps"/map_revision/"map.json")
        return self._plan({"schema_version":"1.0","operation":"export","map_revision":map_revision,
                           "map_fingerprint":digest(mapping),"product":product,"block_assets":block_assets or {},"color_mode":color_mode})

    def plan_correction(self,map_revision,measurement_id,*,max_images=64,max_points=30000,max_iterations=40):
        """Explicit pair seam correction; never mutates original map or marks old dense as corrected."""
        if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in (max_images,max_points,max_iterations)):
            raise ContractError("Correction budgets must be positive integers")
        if max_images<3 or max_images>128 or max_points>50000 or max_iterations>200:
            raise ContractError("Correction exceeds this release's bounded local BA budgets")
        mapping=read(self.root/"maps"/map_revision/"map.json")
        if measurement_id not in {m["id"] for m in mapping["measurements"]}:
            raise ContractError("Correction requires an accepted measurement in the selected map")
        return self._plan({"schema_version":"1.0","operation":"correction","map_revision":map_revision,
                           "map_fingerprint":digest(mapping),"measurement_id":measurement_id,
                           "max_images":max_images,"max_points":max_points,"max_iterations":max_iterations})

    def plan_campaign(self,blocks,*,reference=None,policy=None,map_hints=None,max_expanded_attempts=12):
        """Measure spatial neighbors and return every verified connected component."""
        base=self.plan_add(blocks,reference=reference,policy=policy)
        if not isinstance(max_expanded_attempts,int) or isinstance(max_expanded_attempts,bool) or not 0<=max_expanded_attempts<=100:
            raise ContractError("Invalid expanded attempt budget")
        hints=map_hints or []
        if any(not isinstance(p,list) or len(p)!=2 or any(not isinstance(i,str) for i in p) for p in hints):
            raise ContractError("Map hints must be pairs of block labels")
        body={k:v for k,v in base.items() if k!="id"}
        body.update(operation="campaign",map_hints=hints,max_expanded_attempts=max_expanded_attempts)
        return self._plan(body)

    def _plan(self,body):
        key=digest(body);body={**body,"id":key}
        with database(self.root) as db:
            db.execute("INSERT OR IGNORE INTO plans VALUES(?,?)",(key,json.dumps(body)))
        db.close()
        write(self.root/"plans"/(key+".json"),body)
        return body

    def submit(self,plan,*,image,cpus=8,memory_gb=24,time="02:00:00",partition="standard"):
        """Idempotent for an active/succeeded request with identical code and runtime."""
        if not isinstance(cpus,int) or not 1<=cpus<=32 or not 1<=memory_gb<=98:
            raise ContractError("Invalid resource budget")
        saved=read(self.root/"plans"/(plan["id"]+".json"))
        if saved!=plan:
            raise ContractError("Plan mutated after creation")
        packages={"blockalign":Path(__file__).resolve().parent,
                  "alignexp":Path(importlib.util.find_spec("alignexp").origin).parent}
        h=hashlib.sha256()
        for name,source in sorted(packages.items()):
            for p in sorted(source.rglob("*")):
                if p.is_file() and p.suffix in (".py",".slurm"):
                    h.update((name+"/"+str(p.relative_to(source))).encode());h.update(p.read_bytes())
        image=Path(image).resolve()
        stat=image.stat()
        identity={"plan":plan["id"],"code":h.hexdigest(),"image":str(image),"image_size":stat.st_size,"image_mtime_ns":stat.st_mtime_ns}
        key=digest(identity)
        with lock(self.root/"locks"/"submission.lock"):
            db=database(self.root)
            existing=db.execute("SELECT id FROM runs WHERE id LIKE ? AND state IN ('submitting','submitted','running','succeeded') ORDER BY rowid DESC LIMIT 1",(key+"%",)).fetchone();db.close()
            if existing:
                return self.status(existing[0])
            run_id=key+"-"+uuid.uuid4().hex[:8]
            run_dir=self.root/"runs"/run_id;run_dir.mkdir(parents=True)
            for name,source in packages.items():
                shutil.copytree(source,run_dir/"src"/name,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
            job_name="ba-"+key[:12]+"-"+run_id[-8:]
            write(run_dir/"request.json",{"run_id":run_id,"plan":saved,"identity":identity,"job_name":job_name,
                                         "execution":{"cpus":cpus,"memory_gb":memory_gb,"time":time,"partition":partition,"gpu":False}})
            runner=packages["blockalign"]/"resources/worker.slurm"
            shutil.copy2(runner,run_dir/"runner.slurm")
            with database(self.root) as db:
                db.execute("INSERT INTO runs(id,plan_id,state) VALUES(?,?,'submitting')",(run_id,plan["id"]))
            db.close()
            command=["sbatch","--parsable","--job-name="+job_name,"--partition="+partition,
                     "--ntasks=1","--cpus-per-task="+str(cpus),"--mem="+str(memory_gb)+"G","--time="+time,
                     "--chdir="+str(self.root),"--output="+str(run_dir/"job.out"),"--error="+str(run_dir/"job.err"),
                     str(run_dir/"runner.slurm"),str(self.root),run_id,str(image)]
            from .storage import update_run
            try:
                result=subprocess.run(command,capture_output=True,text=True,timeout=60)
            except (OSError,subprocess.TimeoutExpired) as error:
                update_run(self.root,run_id,error="Submission acknowledgement uncertain: "+str(error))
                raise
            if result.returncode:
                update_run(self.root,run_id,state="failed",error=result.stderr)
                raise RuntimeError(result.stderr)
            job=result.stdout.strip().split(";")[0]
            if not job.isdigit():
                update_run(self.root,run_id,error="Uncertain acknowledgement: "+result.stdout)
                raise RuntimeError("Scheduler receipt requires reconciliation before resubmission")
            # Preserve a worker's early running/succeeded state on fast jobs.
            with database(self.root) as db:
                db.execute("UPDATE runs SET job_id=?,state=CASE WHEN state='submitting' THEN 'submitted' ELSE state END WHERE id=?",(job,run_id))
            db.close()
            return self.status(run_id)

    def status(self,run_id):
        db=database(self.root);db.row_factory=__import__("sqlite3").Row
        row=db.execute("SELECT * FROM runs WHERE id=?",(run_id,)).fetchone();db.close()
        if row is None:
            raise ContractError("Unknown run")
        return dict(row)

    def result(self,run_id):
        status=self.status(run_id)
        if status["state"]!="succeeded":
            raise ContractError("Result is not published: "+str(status))
        return read(self.root/status["result"])

    def reconcile(self,run_id):
        """Explicit scheduler reconciliation; never infer scientific success from exit code alone."""
        from .storage import update_run
        state=self.status(run_id)
        if state["state"] in ("succeeded","failed"):
            return state
        job=state["job_id"]
        if not job:
            request=read(self.root/"runs"/run_id/"request.json")
            query=subprocess.run(["squeue","--me","-h","-o","%i|%j"],capture_output=True,text=True,check=True)
            matches=[r.split("|")[0].strip() for r in query.stdout.splitlines() if r.split("|")[-1].strip()==request.get("job_name")]
            if len(matches)!=1:
                raise ContractError("Uncertain receipt: no unique active job. Inspect sacct before any resubmission.")
            job=matches[0];update_run(self.root,run_id,job_id=job,state="submitted")
        query=subprocess.run(["sacct","-X","-n","-P","-j",job,"--format=JobID,State,ExitCode"],capture_output=True,text=True,check=True)
        rows=[r.split("|") for r in query.stdout.splitlines() if r.split("|")[0].strip()==job]
        if rows:
            status=rows[0][1].split()[0]
            if status=="COMPLETED":
                result=self.root/"runs"/run_id/"result.json"
                if result.exists() and read(result).get("run_id")==run_id and rows[0][2]=="0:0":
                    update_run(self.root,run_id,state="succeeded",result=str(result.relative_to(self.root)),error=None)
                else:
                    update_run(self.root,run_id,state="failed",error="Scheduler completed without a published verified result")
            elif status in {"FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED","BOOT_FAIL"}:
                update_run(self.root,run_id,state="failed",error="Scheduler terminal state: "+status)
        return self.status(run_id)
