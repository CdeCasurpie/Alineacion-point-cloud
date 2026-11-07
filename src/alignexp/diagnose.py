from collections import Counter
import numpy as np
import pycolmap as pc
from . import binary
from .common import ROOT, artifact, save, stage_dir, record, require_job


def run():
    require_job();out=stage_dir("diagnose")
    attempts=[p for p in (ROOT/"work/fusion").iterdir() if (p/"merged_ref009/images.bin").is_file()]
    latest=max(attempts,key=lambda p:int(p.name))
    paths={"input005":ROOT/"inputs/005/model","input009":ROOT/"inputs/009/model",
           "merged":latest/"merged_ref009"}
    results={}
    for name,model in paths.items():
        pts=binary.points(model)
        duplicate_images=exact=affected=0;examples=[]
        views=binary.images(model)
        for point,track in zip(pts,binary.point_tracks(model,pts["offset"],pts["length"])):
            n=len(track)-len(np.unique(track[:,0]))
            if n:
                duplicate_images+=n;affected+=1
                exact+=len(track)-len(np.unique(track,axis=0))
                if len(examples)<8:
                    dup=[int(i) for i,n in Counter(track[:,0]).items() if n>1][0]
                    indices=track[track[:,0]==dup,1]
                    p=binary.point2d(model,views[dup])[indices]
                    xy=np.c_[p["x"],p["y"]]
                    examples.append({"point":int(point["id"]),"image":dup,"indices":indices.tolist(),
                                     "max_feature_separation_px":float(np.linalg.norm(xy-xy[0],axis=1).max())})
        results[name]={"extra_observations_same_image":duplicate_images,"exact_duplicates":exact,
                       "affected_points":affected,"examples":examples}
    results["api"]={key:getattr(getattr(pc.Reconstruction,key,None),"__doc__",None)
                    for key in ("delete_observation","delete_point3D","merge_points3D")}
    save(out/"diagnostic.json",results);record("diagnose",out,diagnostic=results)
