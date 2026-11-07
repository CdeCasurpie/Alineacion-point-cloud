from pathlib import Path
import numpy as np

from alignexp import binary
from alignexp.export import ply
from .artifacts import stamp, verify_stamps, sha, publish
from .contracts import digest, ContractError
from .graph import similarity
from .plyio import transform_clouds
from .storage import read, write
from .palette import palette, render_legend


def export(workspace,request,scratch):
    plan=request["plan"];mapping=read(workspace/"maps"/plan["map_revision"]/"map.json")
    if digest(mapping)!=plan["map_fingerprint"]:
        raise RuntimeError("Map changed after export plan creation")
    local=Path(scratch)/"export";local.mkdir()
    atlas=[];inputs=[];input_ids=[];input_colors=[];preview=[]
    colors=palette({r:mapping["manifests"][r] for r in mapping["placements"]})
    color_mode=plan.get("color_mode","original")
    for ref,S in mapping["placements"].items():
        manifest=mapping["manifests"][ref];S=similarity(S)
        choice=plan["block_assets"].get(ref)
        assets=manifest["assets"]
        if choice=="sparse":
            points=binary.points(manifest["sparse"]["model_uri"])
            if color_mode=="by_block":
                sample=points["xyz"][::max(1,len(points)//6000)]
                preview.append((manifest["block_id"],sample@S[:3,:3].T+S[:3,3],colors[ref]["hex"]))
            path=local/(manifest["block_id"]+"-source-sparse.ply");ply(path,points["xyz"],points["color"])
            selected=[{"asset_id":"sparse","kind":"sparse_preview","uri":str(path),"sparse_from_asset":np.eye(4).tolist(),
                       "generated_from_sparse":mapping["sparse_fingerprints"][ref]}]
        else:
            selected=[a for a in assets if choice is None or a["asset_id"]==choice]
            if choice is not None and not selected:
                raise ContractError("Requested asset not found for "+ref)
        for a in selected:
            if a["generated_from_sparse"]!=mapping["sparse_fingerprints"][ref]:
                raise ContractError("Asset needs regeneration or verified lineage; sparse fingerprint mismatch")
            matrix=S@similarity(a["sparse_from_asset"])
            path=Path(a["uri"])
            before=stamp(path)
            entry={"block":ref,"asset_id":a["asset_id"],"kind":a["kind"],"source_uri":str(path),
                   "source_sha256":sha(path),"world_from_asset":matrix.tolist(),"compatibility":"requires_transform",
                   "block_id":colors[ref]["block_id"],"categorical_rgb":colors[ref]["rgb"]}
            if choice=="sparse":
                # Temporary sources are materialized for exports and not advertised as persistent URIs.
                entry["source_uri"]=manifest["sparse"]["model_uri"]
                entry["source_format"]="colmap_sparse_preview"
            if plan["product"]=="transformed_ply":
                name=manifest["block_id"]+"-"+a["asset_id"]+".ply"
                if Path(name).name!=name:
                    raise ContractError("Unsafe asset ID")
                entry["materialized"]=name
                entry["vertices"]=transform_clouds([(path,matrix)],local/name,block_ids=[colors[ref]["block_id"]],
                    colors=[colors[ref]["rgb"]] if color_mode=="by_block" else None)["vertices"]
                entry["materialized_frame"]="map_world"
            elif plan["product"]=="combined_ply":
                inputs.append((path,matrix))
                input_ids.append(colors[ref]["block_id"]);input_colors.append(colors[ref]["rgb"])
            entry["source_stamp"]=before
            atlas.append(entry)
    combined=transform_clouds(inputs,local/"combined.ply",block_ids=input_ids,
        colors=input_colors if color_mode=="by_block" else None) if plan["product"]=="combined_ply" else None
    for entry in atlas:
        verify_stamps([entry["source_stamp"]])
        if entry["kind"]=="sparse_preview":
            entry.pop("source_stamp")
    # Sources are immutable across placement, but verify the audited input stamps again.
    for ref in mapping["blocks"]:
        found=False
        for audit in (workspace/"audits").glob("*/summary.json"):
            summary=read(audit)
            if summary["ref"]==ref and summary["fingerprint"]==mapping["input_fingerprints"][ref]:
                verify_stamps(summary["input_stamps"]);found=True;break
        if not found:
            raise RuntimeError("Missing audit for export input")
    result={"schema_version":"1.0","map_revision":mapping["id"],"product":plan["product"],"atlas":atlas,
             "combined":combined,"world_origin":mapping["world_origin"],"crs":mapping["crs"],
            "disconnected":mapping["disconnected"],"geometry_variant":mapping["geometry_variant"],
            "color_mode":color_mode,"palette":colors,"materialized_frame":"map_world",
            "transform_note":"world_from_asset applies to source_uri, not to already materialized PLY outputs"}
    write(local/"atlas.json",result)
    write(local/"palette.json",colors)
    render_legend(colors,local/"legend.png")
    if preview:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(14,14))
        for label,xyz,color in preview:
            ax.scatter(xyz[:,0],xyz[:,1],s=.4,color=color,alpha=.5)
            center=np.median(xyz[:,:2],axis=0);ax.text(*center,label,fontsize=9,weight="bold")
        ax.set_aspect("equal");ax.grid(alpha=.2);ax.set_title("Sparse blocks in common frame — original block provenance")
        ax.set_xlabel("Easting offset (m)");ax.set_ylabel("Northing offset (m)")
        fig.savefig(local/"overview_xy.png",dpi=180);plt.close(fig)
    for path in local.glob("*-source-sparse.ply"):
        path.unlink()
    key=digest(result);destination=workspace/"exports"/key
    publish(local,destination)
    return {"export_revision":key,"artifact":str((destination/"atlas.json").relative_to(workspace)),
            "assets":len(atlas),"product":plan["product"]}
