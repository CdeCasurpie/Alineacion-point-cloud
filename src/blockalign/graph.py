"""Small robust Sim(3) graph; measurements stay in original block frames."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from shapely.geometry import shape
from shapely.strtree import STRtree


def similarity(matrix):
    m=np.asarray(matrix,dtype=float)
    if m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1],atol=1e-12):
        raise ValueError("Expected finite affine 4x4 with last row [0,0,0,1]")
    s=np.cbrt(np.linalg.det(m[:3,:3]))
    if s<=0 or not np.allclose(m[:3,:3].T@m[:3,:3],s*s*np.eye(3),rtol=1e-6,atol=1e-10):
        raise ValueError("Expected positive uniform scale and proper rotation")
    return m


def pack(m):
    m=similarity(m);s=np.cbrt(np.linalg.det(m[:3,:3]))
    return np.r_[np.log(s),Rotation.from_matrix(m[:3,:3]/s).as_rotvec(),m[:3,3]]


def unpack(v):
    m=np.eye(4);m[:3,:3]=np.exp(v[0])*Rotation.from_rotvec(v[1:4]).as_matrix();m[:3,3]=v[4:7]
    return m


def proposed_pairs(audits,policy,reference):
    refs=sorted(audits);regions=[shape(audits[r]["coverage"]) for r in refs]
    tree=STRtree(regions);pairs=set()
    for i,geometry in enumerate(regions):
        candidates=tree.query(geometry.buffer(policy["neighbor_distance_m"]))
        candidates=sorted((int(j) for j in candidates if j!=i),key=lambda j:(geometry.distance(regions[j]),refs[j]))
        for j in candidates[:policy["neighbor_limit"]]:
            a,b=sorted((refs[i],refs[j]))
            # Stable direction across incremental plans; prefer the map anchor as target.
            pairs.add((b,a) if a==reference else (a,b))
    return sorted(pairs)


def reachable(reference,edges):
    found={reference}
    while True:
        previous=len(found)
        for e in edges:
            if e["source"] in found or e["target"] in found:
                found.update((e["source"],e["target"]))
        if len(found)==previous:
            return found


def edge_error(edge,placements,scale_source):
    predicted=np.linalg.inv(placements[edge["target"]])@placements[edge["source"]]
    error=np.linalg.inv(similarity(edge["target_from_source"]))@predicted
    s=np.cbrt(np.linalg.det(error[:3,:3]))
    return np.r_[error[:3,3]*scale_source,Rotation.from_matrix(error[:3,:3]/s).as_rotvec(),np.log(s)]


def solve(audits,edges,reference,origin=None):
    origin=np.asarray(origin if origin is not None else audits[reference]["origin"],dtype=float)
    offset=np.eye(4);offset[:3,3]=-origin
    priors={r:offset@similarity(a["projected_from_sparse"]) for r,a in audits.items()}
    connected=reachable(reference,edges)
    variables=sorted(connected-{reference})
    scales={r:np.cbrt(np.linalg.det(s[:3,:3])) for r,s in priors.items()}
    active=[e for e in edges if e["source"] in connected and e["target"] in connected]
    if not variables:
        return {"placements":{reference:priors[reference].tolist()},"world_origin":origin.tolist(),
                "disconnected":sorted(set(audits)-connected),"edge_diagnostics":[],"optimizer":{"termination":"single anchor"},"rejected_edges":[]}

    def matrices(x):
        result={reference:priors[reference]}
        result.update((r,unpack(x[7*i:7*i+7])) for i,r in enumerate(variables))
        return result

    def residual(x,selected):
        poses=matrices(x);out=[]
        for e in selected:
            error=edge_error(e,poses,scales[e["source"]])
            out.extend(error/np.r_[np.full(3,.5),np.full(3,np.deg2rad(1)),.03])
        # Weak geographic priors resolve poorly constrained loops. They are not survey truth.
        for r in variables:
            delta=np.linalg.inv(priors[r])@poses[r]
            s=np.cbrt(np.linalg.det(delta[:3,:3]))
            out.extend(np.r_[delta[:3,3]*scales[r]/10,
                              Rotation.from_matrix(delta[:3,:3]/s).as_rotvec()/np.deg2rad(15),np.log(s)/.15])
        return np.asarray(out)

    initial=np.concatenate([pack(priors[r]) for r in variables])
    fit=least_squares(residual,initial,args=(active,),loss="cauchy",max_nfev=200,x_scale="jac")
    poses=matrices(fit.x);bad=[];good=[]
    for e in active:
        error=edge_error(e,poses,scales[e["source"]])
        (bad if np.linalg.norm(error[:3])>3 or np.linalg.norm(error[3:6])>np.deg2rad(5) or abs(error[6])>.08 else good).append(e)
    if bad:
        remaining=reachable(reference,good)
        if remaining==connected:
            fit=least_squares(residual,fit.x,args=(good,),loss="cauchy",max_nfev=200,x_scale="jac")
            poses=matrices(fit.x)
        else:
            # Return an explicit partial component; never silently place it with GPS alone.
            return solve(audits,good,reference,origin)|{"rejected_edges":[e["id"] for e in bad]}
    diagnostics=[]
    for e in active:
        error=edge_error(e,poses,scales[e["source"]])
        diagnostics.append({"id":e["id"],"translation_m":float(np.linalg.norm(error[:3])),
                            "rotation_deg":float(np.rad2deg(np.linalg.norm(error[3:6]))),"log_scale":float(error[6]),
                            "used":e in good})
    return {"placements":{r:s.tolist() for r,s in poses.items()},"world_origin":origin.tolist(),
            "disconnected":sorted(set(audits)-connected),"edge_diagnostics":diagnostics,"rejected_edges":[e["id"] for e in bad],
            "optimizer":{"success":bool(fit.success),"message":str(fit.message),"cost":float(fit.cost),"evaluations":fit.nfev},
            "prior_policy":"weak geographic priors: 10m, 15deg, log-scale 0.15; fixed reference"}
