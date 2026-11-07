"""Metadata-only public contracts: importing the control API never imports numpy."""
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import hashlib
import json
import math
import re


class ContractError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def strict(data, required, optional=()):
    missing=set(required)-set(data);unknown=set(data)-set(required)-set(optional)
    if missing or unknown:
        raise ContractError(f"Invalid fields: missing={sorted(missing)}, unknown={sorted(unknown)}")


@dataclass(frozen=True)
class Policy:
    schema_version: str = "1.0"
    neighbor_limit: int = 4
    neighbor_distance_m: float = 60.
    selection_margin_m: float = 15.
    max_source_views: int = 32
    max_target_views: int = 16
    max_pairs: int = 600
    min_frame_gap: int = 4
    min_relevant_points: int = 30
    max_spatial_points: int = 50000
    min_pair_inliers: int = 8
    min_pnp_inliers: int = 30
    min_pnp_ratio: float = .20
    max_reprojection_px: float = 4.
    min_bridges: int = 6
    validation_fraction: float = .25
    max_center_error_m: float = 2.
    max_rotation_error_deg: float = 5.
    seed: int = 7

    @classmethod
    def parse(cls, data=None):
        data=data or {}
        strict(data,(),[f.name for f in fields(cls)])
        result=cls(**data)
        if result.schema_version!="1.0":
            raise ContractError("Unsupported policy schema")
        integers={f.name for f in fields(cls) if f.type is int}
        for key,value in asdict(result).items():
            if key=="schema_version":
                continue
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ContractError(f"Invalid numeric policy: {key}")
            if key in integers and not isinstance(value,int):
                raise ContractError(f"Expected integer: {key}")
            if value<0 or (key!="seed" and value==0):
                raise ContractError(f"Expected positive policy: {key}")
        if not 0<result.validation_fraction<.5 or not 0<result.min_pnp_ratio<=1:
            raise ContractError("Invalid fractions")
        if result.max_source_views<result.min_bridges or result.max_pairs<result.max_source_views:
            raise ContractError("Inconsistent view/pair budgets")
        return result

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class BlockManifest:
    data: dict

    @classmethod
    def parse(cls,data,base=None):
        data=json.loads(json.dumps(data,allow_nan=False))
        strict(data,("schema_version","block_id","sparse","features","images","geography"),("assets","provenance"))
        if data["schema_version"]!="1.0" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}",data["block_id"]):
            raise ContractError("Unsupported manifest schema or invalid block ID")
        strict(data["sparse"],("model_uri","format","version","frame_id"))
        if data["sparse"]["format"]!="colmap" or data["sparse"]["version"]!="3.13.0":
            raise ContractError("Supported sparse adapter: COLMAP 3.13.0")
        strict(data["features"],("database_uri","family"),("recipe_fingerprint",))
        if data["features"]["family"]!="sift":
            raise ContractError("Supported feature family: SIFT")
        strict(data["images"],("root_uri","catalog_uri","catalog_format"))
        if data["images"]["catalog_format"] not in ("image_info_v1","pipeline_assets_v1"):
            raise ContractError("Unsupported image catalog")
        strict(data["geography"],("crs",),("telemetry_uri","references_uri","region_uri","region_ids"))
        if data["geography"]["crs"]!="EPSG:32718":
            raise ContractError("This release supports metric GIS EPSG:32718")
        if not (data["geography"].get("telemetry_uri") or data["geography"].get("references_uri")):
            raise ContractError("Geographic camera references are required by this adapter")
        base=Path(base or ".").resolve()
        for group,key in (("sparse","model_uri"),("features","database_uri"),("images","root_uri"),("images","catalog_uri"),
                          ("geography","telemetry_uri"),("geography","references_uri"),("geography","region_uri")):
            if key in data[group]:
                p=Path(data[group][key]).expanduser()
                data[group][key]=str((base/p).resolve() if not p.is_absolute() else p.resolve())
        assets=data.setdefault("assets",[])
        if not isinstance(assets,list):
            raise ContractError("assets must be a list")
        ids=set()
        for a in assets:
            strict(a,("asset_id","kind","uri","frame_id","generated_from_sparse","sparse_from_asset"))
            if a["asset_id"] in ids or a["kind"]!="point_cloud_ply":
                raise ContractError("Duplicate asset or unsupported kind (supported: point_cloud_ply)")
            ids.add(a["asset_id"])
            a["uri"]=str((base/Path(a["uri"])).resolve())
            if not isinstance(a["generated_from_sparse"],str) or not a["generated_from_sparse"]:
                raise ContractError("Asset must identify its generating sparse fingerprint")
            m=a["sparse_from_asset"]
            if not isinstance(m,list) or len(m)!=4 or any(not isinstance(row,list) or len(row)!=4 for row in m):
                raise ContractError("Expected 4x4 affine transform")
            if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for row in m for x in row):
                raise ContractError("Nonfinite transform")
            if m[3]!=[0,0,0,1]:
                raise ContractError("Affine last row must be [0,0,0,1]; an arbitrary homogeneous scale is not a physical scale")
        return cls(data)

    @property
    def ref(self):
        return self.data["block_id"]+"@"+digest(self.data)


def from_colmap_run(block_id,run,region_uri=None,region_ids=None,assets=None):
    """Read small metadata only. Source files are audited later inside Slurm."""
    run=Path(run).resolve()
    quality=json.loads((run/"quality.json").read_text())
    component=Path(quality["best"]["path"]).name
    result={"schema_version":"1.0","block_id":block_id,
            "sparse":{"format":"colmap","version":"3.13.0","model_uri":str(run/"colmap/sparse"/component),"frame_id":block_id+"-sparse"},
            "features":{"family":"sift","database_uri":str(run/"colmap/database.db")},
            "images":{"root_uri":str(run/"frames"),"catalog_uri":str(run/"assets.json"),"catalog_format":"pipeline_assets_v1"},
            "geography":{"crs":"EPSG:32718","telemetry_uri":str(run/"telemetry")},
            "assets":assets or [],"provenance":{"run":str(run)}}
    if region_uri:
        result["geography"].update(region_uri=str(Path(region_uri).resolve()),region_ids=region_ids or [])
    return BlockManifest.parse(result)
