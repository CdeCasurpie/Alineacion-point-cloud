import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from blockalign import BlockManifest, Workspace, Policy, ContractError
from blockalign.contracts import digest
from blockalign.storage import write, read
from blockalign.graph import similarity, solve
from blockalign.plyio import header, chunks, transform_clouds


def manifest(block="arbitrary-block"):
    return {"schema_version":"1.0","block_id":block,
            "sparse":{"format":"colmap","version":"3.13.0","model_uri":"model","frame_id":"local"},
            "features":{"family":"sift","database_uri":"database.db"},
            "images":{"root_uri":"images","catalog_uri":"catalog.json","catalog_format":"image_info_v1"},
            "geography":{"crs":"EPSG:32718","telemetry_uri":"telemetry"}}


def test_metadata_api_does_not_import_array_or_sfm_runtimes():
    p=subprocess.run([sys.executable,"-B","-c","import blockalign,sys; assert 'numpy' not in sys.modules; assert 'pycolmap' not in sys.modules"],capture_output=True,text=True)
    assert p.returncode==0,p.stderr


def test_strict_policy_and_manifest(tmp_path):
    with pytest.raises(ContractError):
        Policy.parse({"ignored_setting":True})
    with pytest.raises(ContractError):
        Policy.parse({"max_source_views":2})
    with pytest.raises(ContractError):
        Policy.parse({"max_pairs":True})
    block=BlockManifest.parse(manifest(),tmp_path)
    assert block.ref.startswith("arbitrary-block@")
    assert block.data["sparse"]["model_uri"]==str(tmp_path/"model")
    assert BlockManifest.parse(block.data).ref==block.ref


def test_pair_cache_never_returns_measurement_bound_to_other_node_revisions():
    from blockalign.pairwise import measurement_key
    a={"fingerprint":"same-input-content","ref":"a@first"}
    b={"fingerprint":"other-input-content","ref":"b@first"}
    c={**a,"ref":"a@second"}
    assert measurement_key(a,b,{},"engine")!=measurement_key(c,b,{},"engine")


def test_selected_homogeneous_matrix_is_not_silently_physical_scale():
    wrong=np.eye(4)*.05
    with pytest.raises(ValueError):
        similarity(wrong)
    good=np.eye(4);good[:3,:3]*=.05
    np.testing.assert_allclose(similarity(good),good)
    with pytest.raises(ValueError):
        similarity(np.diag([-1,1,1,1]))


def test_registry_and_plans_are_stable_and_pin_revisions(tmp_path):
    w=Workspace.open(tmp_path/"workspace")
    a=w.register_block(BlockManifest.parse(manifest("north"),tmp_path))
    b=w.register_block(BlockManifest.parse(manifest("south"),tmp_path))
    p=w.plan_add([a,b],reference=b)
    assert p==w.plan_add([a,b],reference=b)
    assert w.register_block(w.block(a))==a
    changed=w.block(a);changed["sparse"]["model_uri"]+="-new"
    second=w.register_block(changed)
    with pytest.raises(ContractError):
        w.plan_add([a,second])
    assert read(w.root/"plans"/(p["id"]+".json"))==p


def test_registration_pins_available_source_metadata(tmp_path):
    (tmp_path/"model").mkdir()
    source=tmp_path/"model/cameras.bin";source.write_bytes(b"registered")
    w=Workspace.open(tmp_path/"workspace");block=BlockManifest.parse(manifest(),tmp_path)
    ref=w.register_block(block)
    assert w.plan_add([ref])["registration_stamps"][ref]
    source.write_bytes(b"different revision")
    with pytest.raises(ContractError,match="metadata changed"):
        w.register_block(block)


def test_correction_requires_accepted_edge_and_bounded_budget(tmp_path):
    w=Workspace.open(tmp_path/"workspace")
    write(w.root/"maps/example/map.json",{"measurements":[{"id":"accepted-edge"}]})
    plan=w.plan_correction("example","accepted-edge")
    assert plan["operation"]=="correction" and plan["max_iterations"]==40
    with pytest.raises(ContractError):
        w.plan_correction("example","unknown-edge")
    with pytest.raises(ContractError):
        w.plan_correction("example","accepted-edge",max_images=10000)


def test_submit_rejects_mutated_plan_before_scheduler(tmp_path):
    w=Workspace.open(tmp_path/"workspace")
    a=w.register_block(BlockManifest.parse(manifest(),tmp_path))
    p=w.plan_add([a]);p["policy"]["max_pairs"]+=1
    with pytest.raises(ContractError,match="mutated"):
        w.submit(p,image="missing.sif")


def test_scheduler_submission_is_idempotent_and_code_is_snapshotted(tmp_path,monkeypatch):
    import blockalign.workspace as api
    w=Workspace.open(tmp_path/"workspace")
    ref=w.register_block(BlockManifest.parse(manifest(),tmp_path))
    plan=w.plan_add([ref]);image=tmp_path/"runtime.sif";image.write_bytes(b"fixture-runtime")
    calls=[]
    def submit(cmd,**kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd,0,"4321\n","")
    monkeypatch.setattr(api.subprocess,"run",submit)
    first=w.submit(plan,image=image);second=w.submit(plan,image=image)
    assert len(calls)==1 and first["id"]==second["id"]
    assert first["job_id"]=="4321" and first["state"]=="submitted"
    assert (w.root/"runs"/first["id"]/"src/blockalign/worker.py").exists()
    assert (w.root/"runs"/first["id"]/"runner.slurm").exists()


def test_changed_input_or_repointed_symlink_is_rejected(tmp_path):
    from blockalign.artifacts import stamp, verify_stamps
    a=tmp_path/"a";b=tmp_path/"b";alias=tmp_path/"image.jpg"
    a.write_bytes(b"old");b.write_bytes(b"new");alias.symlink_to(a)
    recorded=stamp(alias)
    verify_stamps([recorded])
    alias.unlink();alias.symlink_to(b)
    with pytest.raises(RuntimeError,match="Input changed"):
        verify_stamps([recorded])


def test_incompatible_dense_lineage_is_not_published(tmp_path):
    from blockalign.exports import export
    root=tmp_path/"workspace";scratch=tmp_path/"scratch";scratch.mkdir()
    ref="north@revision"
    m=manifest("north");m["assets"]=[{"asset_id":"dense","kind":"point_cloud_ply","uri":"not-read.ply",
        "generated_from_sparse":"wrong-sparse","sparse_from_asset":np.eye(4).tolist()}]
    body={"id":"map","blocks":[ref],"placements":{ref:np.eye(4).tolist()},"manifests":{ref:m},
          "sparse_fingerprints":{ref:"expected-sparse"}}
    write(root/"maps/map/map.json",body)
    with pytest.raises(ContractError,match="fingerprint mismatch"):
        export(root,{"plan":{"map_revision":"map","map_fingerprint":digest(body),"block_assets":{},"product":"transformed_ply"}},scratch)
    assert not (root/"exports").exists()


def test_graph_25_blocks_loop_outlier_and_disconnected():
    rng=np.random.default_rng(5);truth={};audits={};edges=[]
    for i in range(25):
        ref=str(i);s=np.eye(4);s[:3,:3]=(1+i*.002)*Rotation.from_euler("z",i*.01).as_matrix()
        s[:3,3]=[i%5*25,i//5*25,0]
        truth[ref]=s;prior=s.copy()
        if i:
            prior[:3,3]+=rng.normal(0,.25,3)
        audits[ref]={"origin":[0,0,0],"projected_from_sparse":prior.tolist()}
    for i in range(25):
        for j in (i+1 if i%5<4 else -1,i+5 if i<20 else -1):
            if j>=0:
                edges.append({"id":f"{i}-{j}","source":str(i),"target":str(j),"target_from_source":(np.linalg.inv(truth[str(j)])@truth[str(i)]).tolist()})
    bad=np.linalg.inv(truth["24"])@truth["0"];bad[:3,3]+=[40,30,10]
    edges.append({"id":"bad","source":"0","target":"24","target_from_source":bad.tolist()})
    audits["island"]={"origin":[0,0,0],"projected_from_sparse":np.eye(4).tolist()}
    result=solve(audits,edges,"0")
    assert result["disconnected"]==["island"]
    assert "bad" in result["rejected_edges"]
    for ref,s in truth.items():
        np.testing.assert_allclose(np.array(result["placements"][ref])[:3,3],s[:3,3],atol=.2)


def test_streaming_dense_transform_preserves_normals_attributes_and_count(tmp_path):
    source=tmp_path/"dense.ply"
    source.write_text("ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nend_header\n1 0 0 1 0 0 77\n0 1 0 0 1 0 99\n")
    S=np.eye(4);S[:3,:3]=2*Rotation.from_euler("z",90,degrees=True).as_matrix();S[:3,3]=[10,0,0]
    output=tmp_path/"placed.ply"
    stats=transform_clouds([(source,S)],output)
    assert stats["vertices"]==2
    data=next(chunks(output,header(output)))
    np.testing.assert_allclose(data["x"],[10,8],atol=1e-5)
    np.testing.assert_allclose(data["y"],[2,0],atol=1e-5)
    np.testing.assert_allclose(data["ny"],[1,0],atol=1e-5)
    np.testing.assert_array_equal(data["red"],[77,99])


def test_ply_mesh_and_truncated_payload_are_rejected(tmp_path):
    mesh=tmp_path/"mesh.ply";mesh.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n3 0 0 0\n")
    with pytest.raises(ValueError):
        header(mesh)
    truncated=tmp_path/"short.ply";truncated.write_text("ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
    with pytest.raises(ValueError):
        list(chunks(truncated,header(truncated)))


def test_categorical_export_preserves_block_provenance_and_distinct_colors(tmp_path):
    from blockalign.palette import palette
    source=tmp_path/"source.ply"
    source.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n1 2 3 10 20 30\n")
    colors=palette({"first":{"block_id":"005"},"second":{"block_id":"009"}})
    output=tmp_path/"colored.ply"
    transform_clouds([(source,np.eye(4)),(source,np.eye(4))],output,block_ids=[5,9],colors=[colors["first"]["rgb"],colors["second"]["rgb"]])
    data=next(chunks(output,header(output)))
    np.testing.assert_array_equal(data["block_id"],[5,9])
    assert list(data[0][["red","green","blue"]])==colors["first"]["rgb"]
    assert colors["first"]["rgb"]!=colors["second"]["rgb"]
    assert output.stat().st_size-header(output)["offset"]==2*(15+2)


def test_palette_is_stable_when_more_numbered_blocks_arrive():
    from blockalign.palette import palette
    first=palette({"a":{"block_id":"005"}})
    later=palette({"a":{"block_id":"005"},"b":{"block_id":"009"}})
    assert first["a"]==later["a"]


def test_campaign_components_do_not_join_geographically_close_but_unverified_blocks():
    from blockalign.campaign import components
    edges=[{"source":"a","target":"b"},{"source":"c","target":"d"}]
    result=components(["a","b","c","d","island"],edges)
    assert {frozenset(g) for g in result}=={frozenset(["a","b"]),frozenset(["c","d"]),frozenset(["island"])}
