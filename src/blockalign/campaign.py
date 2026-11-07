"""Multi-block campaign with bounded retries and explicit disconnected outputs."""
from pathlib import Path
import shutil
import time
from shapely.geometry import shape

from .artifacts import publish,verify_stamps
from .audit import audit_block
from .contracts import digest,Policy
from .graph import proposed_pairs,solve
from .pairwise import align_pair,InsufficientEvidence,measurement_key
from .storage import read,write


def components(refs,edges):
    remaining=set(refs);result=[]
    while remaining:
        group={min(remaining)}
        while True:
            previous=len(group)
            for e in edges:
                if e["source"] in group or e["target"] in group:
                    group.update((e["source"],e["target"]))
            if len(group)==previous:
                break
        group&=remaining
        result.append(group);remaining-=group
    return sorted(result,key=lambda g:(-len(g),sorted(g)))


def run(workspace,request,scratch):
    import os
    plan=request["plan"];policy=plan["policy"];audits={};paths={};audit_rejections=[];audit_hits=0
    for ref in plan["blocks"]:
        try:
            verify_stamps(plan.get("registration_stamps",{}).get(ref,[]))
            audits[ref],paths[ref],hit=audit_block(workspace,ref,plan["manifests"][ref],scratch)
            audit_hits+=hit
            print("AUDIT",audits[ref]["block_id"],"cached",hit,flush=True)
        except Exception as error:
            audit_rejections.append({"ref":ref,"reason":f"{type(error).__name__}: {error}"})
            print("AUDIT_REJECTED",ref,str(error),flush=True)
    if not audits:
        raise RuntimeError("No block passed campaign audits: "+str(audit_rejections))
    labels={a["block_id"]:r for r,a in audits.items()}
    reference=plan["reference"] if plan["reference"] in audits else max(audits,key=lambda r:audits[r]["images"])
    origin=audits[reference]["origin"]
    hints=[]
    for a,b in plan["map_hints"]:
        if a in labels and b in labels:
            hints.append((labels[a],labels[b]))
    candidates=list(dict.fromkeys(hints+proposed_pairs(audits,policy,reference)))
    seen=set();ordered=[]
    for a,b in candidates:
        key=frozenset((a,b))
        if key not in seen:
            seen.add(key)
            ordered.append((b,a) if a==reference else (a,b))
    edges=[];attempts=[];pair_hits=0
    checkpoint=workspace/"runs"/request["run_id"]/"campaign-progress.json"

    def attempt(source,target,current_policy,phase):
        nonlocal pair_hits
        start=time.monotonic();entry={"source":source,"target":target,"phase":phase}
        print("PAIR",len(attempts)+1,audits[source]["block_id"],"->",audits[target]["block_id"],phase,flush=True)
        key=measurement_key(audits[source],audits[target],current_policy,request["pair_engine"])
        local=Path(scratch)/("pair-"+key);destination=workspace/"measurements"/key
        try:
            measurement,hit=align_pair(workspace,audits[source],audits[target],paths[source],paths[target],current_policy,
                scratch,request["pair_engine"],int(os.environ["SLURM_CPUS_PER_TASK"]))
            pair_hits+=hit;edges.append(measurement)
            entry.update(status="accepted",measurement=measurement["id"],cached=hit,registered=measurement["registered"])
        except InsufficientEvidence as error:
            entry.update(status="rejected",reason=str(error))
            if local.exists() and not destination.exists():
                (local/"features.db").unlink(missing_ok=True)
                write(local/"rejection.json",entry);publish(local,destination)
        except Exception as error:
            entry.update(status="technical_failure",reason=f"{type(error).__name__}: {error}")
            if local.exists():
                diagnostic=workspace/"runs"/request["run_id"]/"pair-errors"/key
                diagnostic.mkdir(parents=True,exist_ok=True)
                for p in local.glob("*"):
                    if p.suffix in (".json",".log",".txt"):
                        shutil.copy2(p,diagnostic/p.name)
        finally:
            entry["seconds"]=time.monotonic()-start;attempts.append(entry)
            write(checkpoint,{"audited_blocks":sorted(audits),"attempts":attempts,
                "components":[sorted(g) for g in components(audits,edges)]})
            # Each edge has persistent small artifacts; release local models/BDs between edges.
            if local.exists():
                shutil.rmtree(local)
        print("RESULT",entry["status"],round(entry["seconds"],1),flush=True)
        return entry["status"]=="accepted"

    for a,b in ordered:
        attempt(a,b,policy,"neighbors")
    # Reverse only failed cross-component connections, not every failed pair inside a verified component.
    for a,b in ordered:
        groups=components(audits,edges)
        if not any(a in g and b in g for g in groups):
            attempt(b,a,policy,"reverse_bridge")
    expanded=Policy.parse({**policy,"max_source_views":64,"max_target_views":24,"max_pairs":1536,
                           "selection_margin_m":30.,"min_relevant_points":15}).to_dict()
    tried=set();expanded_count=0
    while expanded_count<plan["max_expanded_attempts"]:
        groups=components(audits,edges)
        if len(groups)<=1:
            break
        group_id={r:i for i,g in enumerate(groups) for r in g}
        possible=[]
        refs=sorted(audits)
        for i,a in enumerate(refs):
            for b in refs[i+1:]:
                if group_id[a]==group_id[b] or (a,b) in tried:
                    continue
                distance=shape(audits[a]["coverage"]).distance(shape(audits[b]["coverage"]))
                if distance<=120:
                    possible.append((distance,a,b))
        if not possible:
            break
        _,a,b=min(possible);tried.add((a,b));expanded_count+=1
        if not attempt(a,b,expanded,"expanded_bridge"):
            attempt(b,a,expanded,"expanded_bridge_reverse")

    # Solve each component independently. Graph-rejected links may split a component again.
    pending=components(audits,edges);solutions=[]
    while pending:
        group=pending.pop(0);anchor=reference if reference in group else max(group,key=lambda r:audits[r]["images"])
        group_edges=[e for e in edges if e["source"] in group and e["target"] in group]
        solution=solve({r:audits[r] for r in group},group_edges,anchor,origin)
        if not solution["optimizer"].get("success",True):
            raise RuntimeError("Component graph optimizer did not converge; inspect campaign checkpoint")
        placed=set(solution["placements"])
        if not placed:
            raise RuntimeError("Empty solved component")
        used={e["id"] for e in solution["edge_diagnostics"] if e["used"]}
        keep=[e for e in group_edges if e["id"] in used]
        body={"schema_version":"1.0","blocks":sorted(placed),"reference":anchor,"policy":policy,
              "crs":"EPSG:32718","world_frame":"local UTM18S meters; components are independently anchored",
              "geometry_variant":"placement_original_blocks","measurements":keep,**solution,
              "status":"complete" if len(placed)>1 else "single_block_unaligned","pair_rejections":[],
              "manifests":{r:plan["manifests"][r] for r in placed},
              "input_fingerprints":{r:audits[r]["fingerprint"] for r in placed},
              "sparse_fingerprints":{r:audits[r]["sparse_fingerprint"] for r in placed},
              "metrics":{"audit_cache_hits":audit_hits,"pair_cache_hits":pair_hits,"pairs_computed":len(attempts)-pair_hits}}
        body["disconnected"]=[]
        key=digest(body);body["id"]=key
        local=Path(scratch)/("map-"+key);local.mkdir();write(local/"map.json",body);publish(local,workspace/"maps"/key)
        solutions.append({"map_revision":key,"labels":sorted(audits[r]["block_id"] for r in placed),
                          "blocks":len(placed),"points":sum(audits[r]["points"] for r in placed),"status":body["status"]})
        missing=group-placed
        if missing:
            pending.extend(components(missing,[e for e in group_edges if e["source"] in missing and e["target"] in missing]))
    solutions.sort(key=lambda c:(-c["blocks"],c["labels"]))
    report={"schema_version":"1.0","components":solutions,"attempts":attempts,"audit_rejections":audit_rejections,
            "eligible":sorted(labels),"audit_cache_hits":audit_hits,"pair_cache_hits":pair_hits,
            "scope":"only visual connections join components; no concatenation of GPS-only disconnected blocks"}
    key=digest(report);local=Path(scratch)/"campaign-report";local.mkdir();write(local/"campaign.json",report)
    publish(local,workspace/"campaigns"/key)
    return {"campaign_revision":key,"artifact":"campaigns/"+key+"/campaign.json","components":solutions,
            "attempts":len(attempts),"accepted":sum(a["status"]=="accepted" for a in attempts),
            "technical_failures":sum(a["status"]=="technical_failure" for a in attempts),"audit_rejections":audit_rejections,
            "pair_cache_hits":pair_hits,"largest_map":solutions[0]["map_revision"]}
