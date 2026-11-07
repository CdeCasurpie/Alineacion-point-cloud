import inspect
import subprocess
import sys

import pycolmap

from .common import ROOT, record, require_job, save, stage_dir


def run():
    require_job()
    if pycolmap.__version__ != "3.13.0":
        raise RuntimeError("PyCOLMAP must match native COLMAP 3.13.0")
    native = subprocess.run(["colmap", "-h"], capture_output=True, text=True, check=True)
    if "COLMAP 3.13.0" not in native.stdout + native.stderr:
        raise RuntimeError("Incorrect native COLMAP")
    out = stage_dir("doctor")
    docs = {}
    for name in ("Camera", "Image", "Frame", "Rig", "Rigid3d", "Sim3d", "sensor_t", "data_t",
                 "estimate_and_refine_absolute_pose", "create_default_bundle_adjuster", "BundleAdjustmentConfig"):
        value = getattr(pycolmap, name, None)
        docs[name] = {"doc": getattr(value, "__doc__", None), "init": getattr(getattr(value, "__init__", None), "__doc__", None),
                      "members": [x for x in dir(value) if not x.startswith("_")]}
    for cls, names in ((pycolmap.Reconstruction, ("add_image", "add_frame", "add_rig", "register_frame", "add_point3D", "add_observation", "transform")),
                       (pycolmap.Frame, ("add_data_id", "set_cam_from_world")),
                       (pycolmap.Rig, ("add_ref_sensor", "add_sensor"))):
        for name in names:
            docs[cls.__name__ + "." + name] = getattr(getattr(cls, name, None), "__doc__", None)
    save(out / "api.json", docs)
    test = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True)
    (out / "tests.log").write_text(test.stdout + test.stderr)
    print(test.stdout + test.stderr, flush=True)
    if test.returncode:
        raise RuntimeError("Environment/synthetic contract tests failed")
    record("doctor", out, pycolmap=pycolmap.__version__, native="3.13.0", tests="passed")
