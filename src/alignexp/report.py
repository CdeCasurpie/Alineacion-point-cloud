"""Publish a reproducible experimental bundle only after all stage gates pass."""
import shutil
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import shape
import shapely

from . import binary
from .common import ROOT, artifact, config, load, record, require_job, save, stage_dir, sha
from .sim3 import apply


def run():
    require_job();cfg=config();out=stage_dir("report")
    geo=load(artifact("geography")/"geography.json")
    alignment=load(artifact("localization")/"alignment.json")
    fusion=load(artifact("fusion")/"audit.json")
    refinement=load(artifact("refinement")/"evaluation.json")
    termination=[line.strip() for line in (artifact("refinement")/"solver.txt").read_text().splitlines()
                 if line.startswith("Termination:")]
    refinement["solver_termination_report"]=termination[0] if termination else "See solver.txt"
    region=shape(load(artifact("selection")/"regions.json")[cfg["reference"]]["geometry"]).buffer(15)
    source=binary.points(ROOT/"inputs"/cfg["source"]/"model")
    target=binary.points(ROOT/"inputs"/cfg["reference"] /"model")
    src_prior=np.array(geo["scenes"][cfg["source"]]["S_metric_from_local"])
    tgt_prior=np.array(geo["scenes"][cfg["reference"]]["S_metric_from_local"])
    sim=np.array(alignment["S_reference_from_source"])
    B=apply(tgt_prior,target["xyz"])
    roi=shapely.intersects_xy(region,B[:,0],B[:,1]) & (target["length"]>=3)
    B=B[roi];tree=cKDTree(B)
    fig,axes=plt.subplots(1,2,figsize=(18,9))
    distances={}
    for name,S,ax in (("E0 GPS",src_prior,axes[0]),("E1 visual",tgt_prior@sim,axes[1])):
        A=apply(S,source["xyz"])
        mask=shapely.intersects_xy(region,A[:,0],A[:,1]) & (source["length"]>=3)
        A=A[mask]
        if not len(A):
            raise RuntimeError("No overlapping region points to evaluate")
        A=A[::max(1,len(A)//30000)]
        d=tree.query(A,workers=1)[0]
        distances[name]={"median_source_to_reference_m":float(np.median(d)),"p90_source_to_reference_m":float(np.quantile(d,.9)),
                         "source_samples":len(A),"reference_points":len(B),
                         "interpretation":"ROI nearest-neighbor diagnostic; sparse sampling/occlusion bias, not survey accuracy"}
        ax.scatter(B[::max(1,len(B)//25000),0],B[::max(1,len(B)//25000),1],s=.2,c="darkorange",alpha=.3,label="009")
        ax.scatter(A[:,0],A[:,1],s=.2,c="royalblue",alpha=.3,label="005")
        ax.set_title(name);ax.set_aspect("equal");ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/"alignment_comparison.png",dpi=170);plt.close(fig)
    stages={s:load(ROOT/".state"/f"{s}.json") for s in ("snapshot","doctor","geography","identities","selection","matching","localization","fusion","refinement")}
    result={"schema":1,"pair":"005→009","stages":stages,"alignment":alignment,"fusion":fusion,
            "refinement":refinement,"roi_distances":distances,
            "geographic_origin":geo["origin"],"crs":geo["crs"],
            "limits":["No surveyed ground truth", "DJI vertical datum unverified",
                      "ROI is parcel core plus margin, not certified municipal boundary",
                      "Camera holdout applies to E1 fit; native merger later uses all accepted bridges"]}
    manifest=load(ROOT/"inputs/manifest.json")
    for item in manifest["files"]:
        if sha(ROOT/"inputs"/item["path"])!=item["sha256"]:
            raise RuntimeError(f"Immutable input changed: {item['path']}")
    result["input_verification"]={"sha256_files_verified":len(manifest["files"]),"unchanged":True}
    save(out/"evaluation.json",result)
    lines=["# Experimento 005 → 009", "", "## Resultados", "",
           f"- Cámaras puente registradas: {alignment['registered']}",
           f"- Error mediano de centros usados en ajuste E1: {alignment['fit_median_error_m']:.4f} m",
           f"- Imágenes del modelo fusionado: {fusion['stats']['images']}",
           f"- Puntos del modelo fusionado: {fusion['stats']['points']}",
           f"- Reproyección media antes/después de BA: {refinement['before']['mean_error_px']:.4f} / {refinement['after']['mean_error_px']:.4f} px",
           f"- Reproyección de tracks compartidos antes/después: {refinement['seam_before']['mean_error_px']:.4f} / {refinement['seam_after']['mean_error_px']:.4f} px",
           f"- Estado del solver: {refinement['solver_termination_report']}",
           "", "## Artefactos", ""]
    for stage in ("geography","selection","localization","fusion","refinement"):
        lines.append(f"- {stage}: `{stages[stage]['directory']}`")
    lines += ["", "## Límites", ""]+[f"- {s}" for s in result["limits"]]
    (out/"report.md").write_text("\n".join(lines)+"\n")
    destination=ROOT/"results"/f"prototype-{out.name}"
    shutil.copytree(out,destination)
    shutil.copytree(artifact("refinement")/"model",destination/"model_colmap_ref009")
    shutil.copy2(artifact("refinement")/"solver.txt",destination/"solver.txt")
    (destination/"images").mkdir()
    for scene in cfg["sources"]:
        (destination/"images"/scene).symlink_to(
            os.path.relpath(ROOT/"inputs"/scene/"images",destination/"images"),target_is_directory=True)
    shutil.copy2(artifact("refinement")/"E3_refined_metric.ply",destination/"sparse_metric.ply")
    for name in ("E0_gps_colored.ply","E1_visual_colored.ply","E2_merged_metric.ply"):
        shutil.copy2(artifact("fusion")/name,destination/name)
    save(destination/"coordinates.json",{"model_frame":"009 original reconstruction local frame",
                                         "ply_frame":"local UTM18S offsets in meters",
                                         "projected_origin":geo["origin"],"crs":geo["crs"],
                                         "S_metric_from_model":tgt_prior.tolist(),
                                         "note":"Add origin to metric PLY for projected XYZ; height datum is DJI abs_alt"})
    (ROOT/"results"/"LATEST.md").write_text(
        f"# Resultado vigente del prototipo\n\nDirectorio: `{destination.name}/`\n\n"
        "- `sparse_metric.ply`: nube fusionada y refinada, metros locales.\n"
        "- `model_colmap_ref009/`: modelo nativo con cámaras y tracks, marco original de 009.\n"
        "- `images/`: raíz de imágenes COLMAP, con enlaces internos al snapshot de este proyecto.\n"
        "- `coordinates.json`: conversión explícita entre marcos.\n"
        "- `report.md`, `evaluation.json`, `alignment_comparison.png`: evaluación.\n")
    record("report",destination,report="report.md",evaluation="evaluation.json")
