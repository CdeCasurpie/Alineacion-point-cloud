import argparse
import json
from pathlib import Path
from .workspace import Workspace
from .contracts import from_colmap_run
from .storage import read


def main():
    p=argparse.ArgumentParser(description="Versioned incremental block alignment; compute runs only in Slurm")
    p.add_argument("--workspace",required=True)
    commands=p.add_subparsers(dest="command",required=True)
    register=commands.add_parser("register");register.add_argument("manifest")
    legacy=commands.add_parser("register-colmap-run");legacy.add_argument("block_id");legacy.add_argument("run")
    legacy.add_argument("--regions");legacy.add_argument("--region-id",action="append",default=[])
    add=commands.add_parser("plan-add");add.add_argument("blocks",nargs="+");add.add_argument("--reference");add.add_argument("--base-map");add.add_argument("--policy")
    submit=commands.add_parser("submit");submit.add_argument("plan_id");submit.add_argument("--image",required=True)
    submit.add_argument("--cpus",type=int,default=8);submit.add_argument("--memory-gb",type=int,default=24)
    submit.add_argument("--time",default="02:00:00");submit.add_argument("--partition",default="standard")
    for name in ("status","result","reconcile"):
        sub=commands.add_parser(name);sub.add_argument("run_id")
    export=commands.add_parser("plan-export");export.add_argument("map_revision")
    export.add_argument("--product",choices=("atlas","transformed_ply","combined_ply"),default="atlas")
    export.add_argument("--sparse-preview",action="store_true")
    export.add_argument("--color-mode",choices=("original","by_block"),default="original")
    correction=commands.add_parser("plan-correction");correction.add_argument("map_revision");correction.add_argument("measurement_id")
    correction.add_argument("--max-iterations",type=int,default=40)
    args=p.parse_args();w=Workspace.open(args.workspace)
    if args.command=="register":
        result={"block_ref":w.register_block(args.manifest)}
    elif args.command=="register-colmap-run":
        result={"block_ref":w.register_block(from_colmap_run(args.block_id,args.run,args.regions,args.region_id))}
    elif args.command=="plan-add":
        result=w.plan_add(args.blocks,base_map=args.base_map,reference=args.reference,policy=read(args.policy) if args.policy else None)
    elif args.command=="submit":
        result=w.submit(read(w.root/"plans"/(args.plan_id+".json")),image=args.image,cpus=args.cpus,
                        memory_gb=args.memory_gb,time=args.time,partition=args.partition)
    elif args.command in ("status","result","reconcile"):
        result=getattr(w,args.command)(args.run_id)
    elif args.command=="plan-correction":
        result=w.plan_correction(args.map_revision,args.measurement_id,max_iterations=args.max_iterations)
    else:
        mapping=read(w.root/"maps"/args.map_revision/"map.json")
        assets={r:"sparse" for r in mapping["placements"]} if args.sparse_preview else None
        result=w.plan_export(args.map_revision,args.product,assets,color_mode=args.color_mode)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
