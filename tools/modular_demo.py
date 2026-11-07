"""Control-plane examples for the frozen 005–009 dataset; no numerical work here."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from blockalign import Workspace, BlockManifest
from blockalign.contracts import from_colmap_run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation",choices=("baseline","incremental","export"))
    p.add_argument("--workspace",required=True);p.add_argument("--image",required=True)
    p.add_argument("--base-map");p.add_argument("--block-id");p.add_argument("--run")
    p.add_argument("--product",choices=("atlas","transformed_ply","combined_ply"),default="combined_ply")
    args=p.parse_args();w=Workspace.open(args.workspace)
    if args.operation=="baseline":
        refs=[]
        for block,region in (("005","MZ-BAR-FD7E0473D4"),("009","MZ-BAR-C440DB7CFC")):
            base=ROOT/"inputs"/block
            refs.append(w.register_block(BlockManifest.parse({"schema_version":"1.0","block_id":block,
                "sparse":{"format":"colmap","version":"3.13.0","model_uri":str(base/"model"),"frame_id":block+"-sparse"},
                "features":{"family":"sift","database_uri":str(base/"database.db")},
                "images":{"root_uri":str(base/"images"),"catalog_uri":str(base/"image_info.json"),"catalog_format":"image_info_v1"},
                "geography":{"crs":"EPSG:32718","telemetry_uri":str(base/"telemetry"),
                             "region_uri":str(ROOT/"inputs/gis/block_candidates.geojson"),"region_ids":[region]},
                "assets":[],"provenance":{"dataset":"frozen validated 005–009 snapshot"}})))
        plan=w.plan_add(refs,reference=refs[1])
    elif args.operation=="incremental":
        if not args.base_map or not args.run or not args.block_id:
            p.error("incremental requires --base-map, --block-id, --run")
        ref=w.register_block(from_colmap_run(args.block_id,args.run))
        plan=w.plan_add([ref],base_map=args.base_map)
    else:
        if not args.base_map:
            p.error("export requires --base-map")
        from blockalign.storage import read
        mapping=read(w.root/"maps"/args.base_map/"map.json")
        plan=w.plan_export(args.base_map,args.product,{r:"sparse" for r in mapping["placements"]})
    print(json.dumps({"plan_id":plan["id"],"run":w.submit(plan,image=args.image)},indent=2))


if __name__=="__main__":
    main()
