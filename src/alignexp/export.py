from pathlib import Path

import numpy as np

from . import binary
from .sim3 import apply


def ply(path, xyz, colors):
    path=Path(path)
    header=("ply\nformat binary_little_endian 1.0\n"+f"element vertex {len(xyz)}\n"+
            "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    dtype=np.dtype([("xyz","<f4",3),("rgb","u1",3)])
    with path.open("wb") as f:
        f.write(header.encode())
        for start in range(0,len(xyz),100000):
            data=np.empty(min(100000,len(xyz)-start),dtype=dtype)
            data["xyz"]=xyz[start:start+len(data)];data["rgb"]=colors[start:start+len(data)]
            f.write(data.tobytes())


def model_ply(model, destination, S):
    pts=binary.points(model)
    ply(destination,apply(S,pts["xyz"]),pts["color"])
