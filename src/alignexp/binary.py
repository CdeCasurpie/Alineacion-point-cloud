"""Streaming read access to legacy-compatible COLMAP 3.13 image/point files."""
import mmap
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def cameras(model):
    """Read optimized SIMPLE_RADIAL intrinsics from the reconstruction, not DB priors."""
    result = {}
    with (Path(model) / "cameras.bin").open("rb") as f:
        count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(count):
            cid, kind, width, height = struct.unpack("<iiQQ", f.read(24))
            if kind != 2:
                raise ValueError("Prototype camera reader supports SIMPLE_RADIAL only")
            params = np.array(struct.unpack("<dddd", f.read(32)))
            result[cid] = {"model": "SIMPLE_RADIAL", "width": width, "height": height, "params": params}
        if f.read(1):
            raise ValueError("Unexpected cameras.bin trailing data")
    return result


def images(model):
    result = {}
    path = Path(model) / "images.bin"
    length = path.stat().st_size
    with path.open("rb") as f:
        count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(count):
            row = struct.unpack("<i7di", f.read(64))
            name = bytearray()
            while True:
                c = f.read(1)
                if c == b"\0":
                    break
                if not c or len(name) > 65536:
                    raise RuntimeError("Invalid image name")
                name.extend(c)
            n = struct.unpack("<Q", f.read(8))[0]
            offset = f.tell()
            f.seek(24 * n, 1)
            if f.tell() > length:
                raise RuntimeError("Truncated Point2D vector")
            q = row[1:5]
            R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            t = np.asarray(row[5:8])
            result[row[0]] = {"id": row[0], "name": name.decode(), "camera_id": row[8],
                              "R": R, "t": t, "center": -R.T @ t, "n2d": n, "offset": offset}
        if f.tell() != length:
            raise RuntimeError("Unexpected images.bin trailing data")
    return result


def point2d(model, image):
    dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("point", "<i8")])
    with (Path(model) / "images.bin").open("rb") as f:
        f.seek(image["offset"])
        return np.frombuffer(f.read(24 * image["n2d"]), dtype=dtype).copy()


def points(model):
    path = Path(model) / "points3D.bin"
    dtype = np.dtype([("id", "<u8"), ("xyz", "<f8", (3,)), ("color", "u1", (3,)), ("error", "<f8"), ("length", "<u8"), ("offset", "<u8")])
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as b:
        n = struct.unpack_from("<Q", b, 0)[0]
        out = np.empty(n, dtype=dtype)
        at = 8
        for i in range(n):
            row = struct.unpack_from("<QdddBBBdQ", b, at)
            out[i] = (row[0], row[1:4], row[4:7], row[-2], row[-1], at + 51)
            at += 51 + 8 * row[-1]
        if at != len(b):
            raise RuntimeError("Truncated points3D tracks")
    return out


def point_tracks(model, offsets, lengths):
    with (Path(model) / "points3D.bin").open("rb") as f:
        for offset, length in zip(offsets, lengths):
            f.seek(int(offset))
            yield np.frombuffer(f.read(int(length) * 8), dtype="<u4").reshape(-1, 2).copy()
