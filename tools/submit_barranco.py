"""Metadata-only campaign submission. Inventory and all numerical work use jobs."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from blockalign import Workspace,BlockManifest
from blockalign.storage import read,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory",required=True);p.add_argument("--workspace",required=True);p.add_argument("--image",required=True)
    p.add_argument("--expanded-attempts",type=int,default=12);p.add_argument("--time",default="04:00:00")
    a=p.parse_args();w=Workspace.open(a.workspace);inventory=read(a.inventory)
    refs=[];by_label={}
    for item in inventory["blocks"]:
        if not item["eligible"]:
            continue
        manifest=read(item["manifest"])
        known={"005":"MZ-BAR-FD7E0473D4","009":"MZ-BAR-C440DB7CFC"}
        if item["block_id"] in known:
            manifest["geography"].update(region_uri=str(ROOT/"inputs/gis/block_candidates.geojson"),region_ids=[known[item["block_id"]]])
        ref=w.register_block(BlockManifest.parse(manifest));refs.append(ref);by_label[item["block_id"]]=ref
    hints=read(ROOT/"examples/barranco-map-neighbors.json")["pairs"]
    plan=w.plan_campaign(refs,reference=by_label.get("009"),map_hints=hints,max_expanded_attempts=a.expanded_attempts)
    run=w.submit(plan,image=a.image,cpus=8,memory_gb=24,time=a.time)
    receipt={"inventory":str(Path(a.inventory).resolve()),"blocks":by_label,"plan_id":plan["id"],"run":run}
    write(w.root/"campaign-submission.json",receipt)
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
