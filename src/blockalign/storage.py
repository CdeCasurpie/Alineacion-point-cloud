import contextlib
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import tempfile


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=".pending-",dir=path.parent)
    try:
        with os.fdopen(fd,"w") as f:
            json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
            f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        Path(temp).unlink(missing_ok=True)


@contextlib.contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        yield


def database(root):
    db=sqlite3.connect(Path(root)/"registry.sqlite",timeout=30)
    # DELETE journal: do not use WAL across the cluster's NFS clients.
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def initialize(root):
    Path(root).mkdir(parents=True,exist_ok=True)
    with database(root) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS blocks(ref TEXT PRIMARY KEY, block_id TEXT NOT NULL, manifest TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS registrations(ref TEXT PRIMARY KEY, source_stamps TEXT NOT NULL,
          FOREIGN KEY(ref) REFERENCES blocks(ref));
        CREATE TABLE IF NOT EXISTS plans(id TEXT PRIMARY KEY, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, state TEXT NOT NULL,
          job_id TEXT, result TEXT, error TEXT, FOREIGN KEY(plan_id) REFERENCES plans(id));
        """)
    db.close()


def update_run(root,run_id,**kwargs):
    allowed={"state","job_id","result","error"}
    if not kwargs or set(kwargs)-allowed:
        raise ValueError("Invalid run update")
    with database(root) as db:
        db.execute("UPDATE runs SET "+",".join(k+"=?" for k in kwargs)+" WHERE id=?",[*kwargs.values(),run_id])
    db.close()
