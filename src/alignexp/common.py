from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("ALIGNEXP_ROOT", Path(__file__).resolve().parents[2])).resolve()


def require_job():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Numerical/data phases must run in a Slurm allocation")


def config():
    return yaml.safe_load(Path(os.environ.get("ALIGNEXP_CONFIG",ROOT / "configs/experiment.yaml")).read_text())


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".json-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_dir(stage):
    out = ROOT / "work" / stage / os.environ["SLURM_JOB_ID"]
    out.mkdir(parents=True, exist_ok=True)
    return out


def record(stage, directory, **summary):
    data = {"stage": stage, "directory": str(Path(directory).relative_to(ROOT)),
            "job_id": os.environ["SLURM_JOB_ID"], "at": datetime.now(timezone.utc).isoformat(),
            "configuration_sha256":sha(Path(os.environ.get("ALIGNEXP_CONFIG",ROOT/"configs/experiment.yaml"))), **summary}
    save(Path(directory) / "stage.json", data)
    save(ROOT / ".state" / f"{stage}.json", data)
    print(json.dumps(data, indent=2), flush=True)


def artifact(stage):
    return ROOT / load(ROOT / ".state" / f"{stage}.json")["directory"]
