from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from .common import ROOT, config, load, record, require_job, save, sha


def run():
    require_job()
    cfg = config()
    final = ROOT / "inputs"
    if (final / "manifest.json").exists():
        raise RuntimeError("Snapshot already exists; do not overwrite immutable inputs")
    temp = Path(__import__("os").environ["TMPDIR"]) / "snapshot"
    temp.mkdir()
    inventory = []

    def copy(src, rel):
        dst = temp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(src).resolve(strict=True), dst)
        inventory.append({"path": str(rel), "source": str(src), "bytes": dst.stat().st_size, "sha256": sha(dst)})

    scenes = {}
    for scene, spec in cfg["sources"].items():
        source = Path(spec["run"])
        quality = load(source / "quality.json")
        if quality["best"]["registered_images"] != spec["images"] or quality["best"]["points"] != spec["points"]:
            raise RuntimeError("Source version changed")
        copy(source / "colmap/sparse.ply", Path(scene) / "sparse.ply")
        model = source / "colmap/sparse" / spec["component"]
        for name in ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin"):
            copy(model / name, Path(scene) / "model" / name)
        for name in ("assets.json", "run.json", "quality.json", "config_resolved.yaml", "metrics/metrics.json"):
            copy(source / name, Path(scene) / "provenance" / name)
        assets = load(source / "assets.json")
        image_info = {}
        for a in assets:
            for image in a["images"]:
                relative = image["relative"]
                copy(source / "frames" / relative, Path(scene) / "images" / relative)
                image_info[relative] = {"clip": Path(a["relative"]).stem, "index": image["index"],
                                        "time_s": image["time_s"], "asset": a["id"]}
        if len(image_info) != spec["images"]:
            raise RuntimeError("Missing/duplicated image identities")
        save(temp / scene / "image_info.json", image_info)
        for path in (source / "telemetry").rglob("*"):
            if path.is_file():
                copy(path, Path(scene) / "telemetry" / path.relative_to(source / "telemetry"))
        dbpath = temp / scene / "database.db"
        src = sqlite3.connect((source / "colmap/database.db").as_uri() + "?mode=ro", uri=True)
        dst = sqlite3.connect(dbpath)
        try:
            src.backup(dst)
            dst.commit()
            if dst.execute("SELECT COUNT(*) FROM images").fetchone()[0] != spec["images"]:
                raise RuntimeError("Database image count differs")
        finally:
            src.close()
            dst.close()
        inventory.append({"path": f"{scene}/database.db", "source": str(source / "colmap/database.db"),
                          "bytes": dbpath.stat().st_size, "sha256": sha(dbpath)})
        scenes[scene] = {"source_run": str(source), "source_component": spec["component"],
                         "images": spec["images"], "points": spec["points"]}
    for name in ("lots", "block_candidates", "map"):
        src = Path(cfg["gis"][name])
        copy(src, Path("gis") / (name + src.suffix))
    manifest = {"schema": 1, "scenes": scenes, "files": inventory,
                "bytes": sum(f["bytes"] for f in inventory), "symlinks": 0}
    save(temp / "manifest.json", manifest)
    final.parent.mkdir(exist_ok=True)
    staging = final.with_name("inputs.partial")
    if staging.exists() or final.exists():
        raise RuntimeError("Existing snapshot directory requires review")
    shutil.copytree(temp, staging)
    for item in inventory:
        if sha(staging / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Copied artifact checksum mismatch: {item['path']}")
    if any(p.is_symlink() for p in staging.rglob("*")):
        raise RuntimeError("Snapshot contains external symlinks")
    staging.rename(final)
    record("snapshot", final, bytes=manifest["bytes"], scenes=scenes)
    for path in final.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
