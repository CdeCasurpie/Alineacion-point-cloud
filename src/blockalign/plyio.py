"""Streaming scalar-vertex PLY transforms. Meshes/list properties fail explicitly."""
from pathlib import Path
import numpy as np
from .graph import similarity

TYPES={"char":"i1","uchar":"u1","int8":"i1","uint8":"u1","short":"<i2","ushort":"<u2","int":"<i4","uint":"<u4",
       "int16":"<i2","uint16":"<u2","int32":"<i4","uint32":"<u4","float":"<f4","double":"<f8","float32":"<f4","float64":"<f8"}


def header(path):
    props=[];count=None;fmt=None;current=None
    with Path(path).open("rb") as f:
        if f.readline().strip()!=b"ply":
            raise ValueError("Not PLY")
        for _ in range(10000):
            line=f.readline()
            if not line:
                raise ValueError("Truncated PLY header")
            row=line.decode("ascii").strip().split()
            if not row:
                continue
            if row[0]=="end_header":
                break
            if row[0]=="format":
                fmt=row[1]
            elif row[0]=="element":
                current=row[1]
                if current=="vertex":
                    count=int(row[2])
                elif int(row[2])!=0:
                    raise ValueError("Only point clouds supported, not nonempty mesh/list elements")
            elif row[0]=="property" and current=="vertex":
                if row[1]=="list" or row[1] not in TYPES:
                    raise ValueError("Unsupported vertex property")
                props.append((row[2],row[1]))
        else:
            raise ValueError("Oversized PLY header")
        offset=f.tell()
    names=[n for n,t in props]
    if count is None or count<=0 or len(set(names))!=len(names) or not {"x","y","z"}<=set(names):
        raise ValueError("Invalid vertex schema")
    if fmt not in ("ascii","binary_little_endian"):
        raise ValueError("Supported PLY: ascii or binary_little_endian")
    if set(names)&{"nx","ny","nz"} and not {"nx","ny","nz"}<=set(names):
        raise ValueError("Normals must provide nx, ny and nz together")
    if any(TYPES[t] not in ("<f4","<f8") for n,t in props if n in ("x","y","z","nx","ny","nz")):
        raise ValueError("Coordinates/normals must use floating-point properties")
    return {"count":count,"format":fmt,"properties":props,"offset":offset}


def dtype(meta):
    return np.dtype([(n,TYPES[t]) for n,t in meta["properties"]])


def chunks(path,meta,limit=100000):
    dt=dtype(meta)
    with Path(path).open("rb") as f:
        f.seek(meta["offset"])
        remaining=meta["count"]
        while remaining:
            n=min(limit,remaining)
            if meta["format"]=="binary_little_endian":
                raw=f.read(n*dt.itemsize)
                if len(raw)!=n*dt.itemsize:
                    raise ValueError("Truncated PLY payload")
                data=np.frombuffer(raw,dtype=dt).copy()
            else:
                data=np.empty(n,dtype=dt)
                for i in range(n):
                    row=f.readline().split()
                    if len(row)!=len(meta["properties"]):
                        raise ValueError("Truncated or malformed ASCII vertices")
                    for value,(name,_) in zip(row,meta["properties"]):
                        data[name][i]=float(value) if dt[name].kind=="f" else int(value)
            remaining-=n
            yield data
        if f.read().strip():
            raise ValueError("Unexpected PLY trailing payload")


def write_header(f,count,props):
    text="ply\nformat binary_little_endian 1.0\n"+f"element vertex {count}\n"
    text+="".join(f"property {t} {n}\n" for n,t in props)+"end_header\n"
    f.write(text.encode())


def transform_clouds(inputs,destination,*,block_ids=None,colors=None):
    """inputs: [(path, matrix), ...]; concatenation requires identical attributes."""
    entries=[(Path(p),similarity(S),header(p)) for p,S in inputs]
    if not entries:
        raise ValueError("No cloud inputs")
    base_props=entries[0][2]["properties"]
    if any(m["properties"]!=base_props for _,_,m in entries):
        raise ValueError("Combined PLY requires matching vertex schemas; export per-block instead")
    if block_ids is not None and (len(block_ids)!=len(entries) or any(not 0<=i<=65535 for i in block_ids)):
        raise ValueError("Expected one uint16 block ID per input")
    if colors is not None and (len(colors)!=len(entries) or any(len(c)!=3 or any(not 0<=int(v)<=255 for v in c) for c in colors)):
        raise ValueError("Expected one RGB color per input")
    props=list(base_props);names={n for n,t in props}
    if block_ids is not None:
        if "block_id" in names:
            raise ValueError("Source already contains block_id; do not silently overwrite provenance")
        props.append(("block_id","ushort"))
    if colors is not None:
        props=[(n,"uchar" if n in ("red","green","blue") else t) for n,t in props]
        props.extend((n,"uchar") for n in ("red","green","blue") if n not in names)
    output_dtype=np.dtype([(n,TYPES[t]) for n,t in props])
    count=sum(m["count"] for _,_,m in entries)
    with Path(destination).open("wb") as f:
        write_header(f,count,props)
        for source_index,(path,S,meta) in enumerate(entries):
            scale=np.cbrt(np.linalg.det(S[:3,:3]));R=S[:3,:3]/scale
            for data in chunks(path,meta):
                xyz=np.c_[data["x"],data["y"],data["z"]]
                xyz=xyz@S[:3,:3].T+S[:3,3]
                if not np.isfinite(xyz).all():
                    raise ValueError("Nonfinite transformed coordinates")
                for i,n in enumerate(("x","y","z")):
                    data[n]=xyz[:,i]
                    if not np.isfinite(data[n]).all():
                        raise ValueError("Transformed coordinates exceed PLY numeric range")
                names=data.dtype.names
                if {"nx","ny","nz"}<=set(names):
                    normals=np.c_[data["nx"],data["ny"],data["nz"]]@R.T
                    if not np.isfinite(normals).all():
                        raise ValueError("Nonfinite cloud normals")
                    length=np.linalg.norm(normals,axis=1)
                    normals/=np.maximum(length[:,None],1e-30)
                    for i,n in enumerate(("nx","ny","nz")):
                        data[n]=normals[:,i]
                if block_ids is not None or colors is not None:
                    output=np.empty(len(data),dtype=output_dtype)
                    for name in data.dtype.names:
                        if colors is None or name not in ("red","green","blue"):
                            output[name]=data[name]
                    if block_ids is not None:
                        output["block_id"]=block_ids[source_index]
                    if colors is not None:
                        for i,name in enumerate(("red","green","blue")):
                            output[name]=colors[source_index][i]
                    data=output
                f.write(data.tobytes())
    return {"vertices":count,"properties":props,"operation":"similarity placement / concatenation; not surface fusion"}
