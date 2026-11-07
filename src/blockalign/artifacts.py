"""Worker-side content and publication helpers."""
from pathlib import Path
import hashlib
import os
import shutil
import uuid

from .storage import read, write


def require_job():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Numerical work and dataset audits require a Slurm allocation")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def stamp(path):
    p=Path(path).resolve(strict=True);s=p.stat()
    return {"input":str(Path(path).absolute()),"path":str(p),"size":s.st_size,"mtime_ns":s.st_mtime_ns}


def verify_stamps(stamps):
    for item in stamps:
        if stamp(item["input"])!=item:
            raise RuntimeError("Input changed after audit; register a new immutable revision: "+item["path"])


def publish(local,destination):
    local=Path(local);destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(".pending-"+uuid.uuid4().hex)
    shutil.copytree(local,temporary)
    try:
        os.rename(temporary,destination)
    except OSError:
        shutil.rmtree(temporary)
        if not destination.exists():
            raise


def command(argv,log,*,env=None):
    import subprocess
    with Path(log).open("w") as f:
        completed=subprocess.run(list(map(str,argv)),stdout=f,stderr=subprocess.STDOUT,env=env)
    if completed.returncode:
        raise RuntimeError(f"External command failed ({completed.returncode}): {argv[0]}; see {log}")
