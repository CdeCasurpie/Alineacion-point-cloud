"""Worker-side content and publication helpers."""
from pathlib import Path
import hashlib
import os
import shutil
import uuid

from .storage import read, write


def require_job():
    if os.environ.get("SLURM_JOB_ID"):
        return
    # Explicit local backend for L507. It must run in a real bounded user scope;
    # no fabricated Slurm IDs and no unbounded numerical work on import.
    context = os.environ.get("BLOCKALIGN_LOCAL_CONTEXT")
    if context:
        value = read(context)
        group = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split(':', 2)[2].lstrip('/')
        maximum = (group / 'memory.max').read_text().strip()
        swap = (group / 'memory.swap.max').read_text().strip()
        cpu = (group / 'cpu.max').read_text().split()
        if (value.get('backend') == 'local' and value.get('pid') == os.getpid()
                and maximum != 'max' and int(maximum) <= 40 * 1024**3 and swap == '0'
                and cpu[0] != 'max' and int(cpu[0]) / int(cpu[1]) <= 8):
            return
    raise RuntimeError("Numerical work requires Slurm or an explicit resource-bounded local allocation")


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
    if os.environ.get('BLOCKALIGN_LOCAL_CONTEXT') and local.stat().st_dev==destination.parent.stat().st_dev:
        # Local scratch and workspace share /data: move the derived product
        # atomically instead of temporarily doubling large dense exports.
        os.rename(local,temporary)
    else:
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
