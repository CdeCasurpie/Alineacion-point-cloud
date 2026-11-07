from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import shape, mapping
from shapely.ops import unary_union, transform
from pyproj import Transformer

from . import binary
from .common import ROOT, artifact, config, load, record, require_job, save, stage_dir
from .identities import offsets


def support(scene, region, cfg, geo):
    data=np.load(geo/f"points_{scene}.npz")
    pts,xyz=data["points"],data["xyz"]
    mask=shapely.intersects_xy(region,xyz[:,0],xyz[:,1]) & (pts["length"]>=cfg["min_track_length"]) & (pts["error"]<2)
    idx=np.flatnonzero(mask)
    # Representative points per small voxel, then a bounded spatial sample.
    quality=pts["error"][idx]/np.sqrt(pts["length"][idx])
    idx=idx[np.argsort(quality)]
    _,unique=np.unique(np.floor(xyz[idx]/2).astype(np.int64),axis=0,return_index=True)
    idx=idx[np.sort(unique)]
    if len(idx)>cfg["max_spatial_points"]:
        idx=idx[np.random.default_rng(7).choice(len(idx),cfg["max_spatial_points"],replace=False)]
    scores=Counter();cells=defaultdict(set)
    for i,track in zip(idx,binary.point_tracks(ROOT/"inputs"/scene/"model",pts["offset"][idx],pts["length"][idx])):
        cell=tuple(np.floor(xyz[i,:2]/10).astype(int))
        for iid in np.unique(track[:,0]):
            scores[int(iid)]+=1
            cells[int(iid)].add(cell)
    return scores,cells,{"points_in_region":int(mask.sum()),"representatives":len(idx)}


def diverse(ranked, views, info, gap, limit):
    selected=[]
    for iid in ranked:
        image=info[views[iid]["name"]]
        if any(image["clip"]==info[views[old]["name"]]["clip"] and abs(image["index"]-info[views[old]["name"]]["index"])<gap for old in selected):
            continue
        selected.append(iid)
        if len(selected)>=limit:
            break
    return selected


def run():
    require_job();cfg=config();out=stage_dir("selection");geo=artifact("geography")
    catalog=load(geo/"region_candidates.json")
    regions={};region_meta={}
    for scene in cfg["sources"]:
        ids=cfg["gis"]["block_ids"][scene]
        selected=[f for f in catalog if f["properties"]["id"] in ids]
        if len(selected)!=len(ids) or any(not f["properties"]["groupingValid"] for f in selected):
            raise RuntimeError("Unverified or missing geographic grouping")
        polygon=unary_union([shape(f["geometry"]) for f in selected])
        if not polygon.is_valid:
            raise RuntimeError("Invalid selected polygon")
        regions[scene]=polygon
        region_meta[scene]={"geometry":mapping(polygon),"source_group_ids":ids,
                            "lots":sorted({lot for f in selected for lot in f["properties"]["lots"]}),
                            "policy":"conservative parcel core reviewed against provisional capture map; not official block boundary",
                            "selection_buffer_m":cfg["gps"]["selection_margin_m"]}
    save(out/"regions.json",region_meta)
    geography=load(geo/"geography.json")
    origin=np.array(geography["origin"])
    inverse=Transformer.from_crs(cfg["gis"]["metric_crs"],"EPSG:4326",always_xy=True)
    features=[]
    for s,p in regions.items():
        world=transform(lambda x,y,z=None:inverse.transform(np.asarray(x)+origin[0],np.asarray(y)+origin[1]),p)
        features.append({"type":"Feature","properties":{"capture":s,**{k:v for k,v in region_meta[s].items() if k!="geometry"}},"geometry":mapping(world)})
    save(out/"regions_wgs84.geojson",{"type":"FeatureCollection","features":features})
    src,ref=cfg["source"],cfg["reference"]
    region=regions[ref].buffer(cfg["gps"]["selection_margin_m"])
    scores,src_cells,source_stats=support(src,region,cfg["selection"],geo)
    target_scores,target_cells,target_stats=support(ref,region,cfg["selection"],geo)
    views={s:binary.images(ROOT/"inputs"/s/"model") for s in (src,ref)}
    info={s:load(ROOT/"inputs"/s/"image_info.json") for s in (src,ref)}
    ranked=[iid for iid,n in scores.most_common() if n>=cfg["selection"]["min_relevant_points"]]
    chosen=diverse(ranked,views[src],info[src],cfg["selection"]["min_frame_gap"],cfg["selection"]["max_source_views"])
    pairs=[];candidates=[]
    for rank,iid in enumerate(chosen):
        expanded={(x+dx,y+dy) for x,y in src_cells[iid] for dx in (-1,0,1) for dy in (-1,0,1)}
        target_rank=sorted(target_cells,key=lambda j:len(expanded & target_cells[j]),reverse=True)
        target_rank=[j for j in target_rank if len(expanded & target_cells[j])>0]
        targets=diverse(target_rank,views[ref],info[ref],cfg["selection"]["min_frame_gap"],cfg["selection"]["max_target_views"])
        source_name=src+"/"+views[src][iid]["name"]
        candidates.append({"image_id":iid+offsets(src)[0],"original_id":iid,"name":source_name,
                           "score":scores[iid],"role":"validation" if rank%4==0 else "fit",
                           "target_ids":[j+offsets(ref)[0] for j in targets]})
        pairs.extend((source_name,ref+"/"+views[ref][j]["name"]) for j in targets)
    pairs=sorted(set(pairs))[:cfg["selection"]["max_pairs"]]
    if len(candidates)<cfg["alignment"]["min_shared_views"] or not pairs:
        raise RuntimeError("Insufficient geography/track support; inspect ROI and GPS prior")
    save(out/"candidates.json",candidates)
    (out/"pairs.txt").write_text("".join(f"{a} {b}\n" for a,b in pairs))
    fig,ax=plt.subplots(figsize=(12,12))
    for s,c in ((src,"royalblue"),(ref,"darkorange")):
        p=regions[s];x,y=p.exterior.xy;ax.fill(x,y,alpha=.1,color=c);ax.plot(x,y,color=c,label=f"parcel core {s}")
    x,y=region.exterior.xy;ax.plot(x,y,"--",color="darkorange",label="009 selection buffer")
    cameras=load(geo/f"cameras_{src}.json");lookup={v["image_id"]:v for v in cameras}
    all_c=np.array([v["center"] for v in cameras]);ax.scatter(all_c[:,0],all_c[:,1],s=2,c="gray",alpha=.3)
    for candidate in candidates:
        c=lookup[candidate["original_id"]]["center"];ax.scatter(c[0],c[1],s=20,c="red" if candidate["role"]=="validation" else "blue")
    ax.set_aspect("equal");ax.grid(alpha=.2);ax.legend();ax.set_title("005–009 parcel cores and track-selected bridge views")
    fig.savefig(out/"selected_views.png",dpi=150);plt.close(fig)
    record("selection",out,source_stats=source_stats,target_stats=target_stats,
           candidates=len(candidates),pairs=len(pairs),regions=region_meta)
