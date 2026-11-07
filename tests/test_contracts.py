import numpy as np
import pytest
from pyproj import Transformer
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon
import shapely
import struct

from alignexp.sim3 import apply, robust, umeyama


def test_camera_reader_preserves_refined_intrinsics(tmp_path):
    from alignexp.binary import cameras
    expected = np.array([2831.234, 1920., 1080., -0.0123])
    (tmp_path / "cameras.bin").write_bytes(struct.pack("<QiiQQdddd",1,7,2,3840,2160,*expected))
    parsed = cameras(tmp_path)
    np.testing.assert_array_equal(parsed[7]["params"], expected)
    assert parsed[7]["width"] == 3840


def test_similarity_recovers_positive_scale_rotation_translation_and_inverse():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 3))
    S = np.eye(4)
    S[:3, :3] = 2.5 * Rotation.from_euler("xyz", [20, -15, 7], degrees=True).as_matrix()
    S[:3, 3] = [10, -5, 3]
    y = apply(S, x)
    recovered = umeyama(x, y)
    np.testing.assert_allclose(recovered, S, atol=1e-10)
    np.testing.assert_allclose(apply(np.linalg.inv(recovered), y), x, atol=1e-10)


def test_similarity_rejects_collinear_but_accepts_planar():
    x = np.c_[np.arange(10), np.zeros((10, 2))]
    with pytest.raises(ValueError):
        umeyama(x, x)
    x = np.c_[np.random.default_rng(1).normal(size=(20, 2)), np.zeros(20)]
    np.testing.assert_allclose(umeyama(x, x), np.eye(4), atol=1e-10)


def test_robust_alignment_rejects_outliers():
    x = np.random.default_rng(3).normal(size=(40, 3))
    y = 3 * x + [2, 4, 6]
    y[-8:] += 100
    S, mask, errors = robust(x, y, 0.05, iterations=200)
    assert mask.sum() == 32
    np.testing.assert_allclose(apply(S, x[:32]), y[:32], atol=1e-7)


def test_robust_recovers_globally_consistent_set_without_dominant_triplet():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(11, 3))
    S = np.eye(4)
    S[:3, :3] = 2.3 * Rotation.from_euler("xyz", [12, -40, 65], degrees=True).as_matrix()
    S[:3, 3] = [7, 2, -3]
    noise = rng.normal(scale=1e-3, size=(11, 3))
    y = apply(S, x) + noise
    recovered, mask, errors = robust(x, y, 0.05, iterations=400)
    assert mask.sum() == 11, mask
    np.testing.assert_allclose(recovered, S, atol=0.2)
    assert errors.max() < 0.05


def test_gis_axis_order_and_metric_buffer_boundary():
    forward = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)
    back = Transformer.from_crs("EPSG:32718", "EPSG:4326", always_xy=True)
    e, n = forward.transform(-77.022, -12.136)
    assert 279000 < e < 282000 and 8650000 < n < 8660000
    np.testing.assert_allclose(back.transform(e, n), [-77.022, -12.136])
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert shapely.intersects_xy(poly, 0, 5)  # Include exact boundary points.
    assert poly.buffer(2).covers(__import__("shapely").geometry.Point(-1, 5))


def test_colmap_trivial_rig_registration_and_observations():
    import pycolmap as pc
    from alignexp.colmap_io import add_camera_and_rig, add_view
    rec = pc.Reconstruction()
    add_camera_and_rig(rec,1,1,"SIMPLE_RADIAL",640,480,[500,320,240,0])
    add_view(rec,1,"a.jpg",1,1,[[320,240]],pc.Rigid3d(np.c_[np.eye(3),[0,0,0]]))
    add_view(rec,2,"b.jpg",1,1,[[220,240]],pc.Rigid3d(np.c_[np.eye(3),[-1,0,0]]))
    assert rec.num_reg_images()==2
    point = rec.add_point3D([0,0,5],pc.Track([pc.TrackElement(1,0),pc.TrackElement(2,0)]),np.array([10,20,30],dtype=np.uint8))
    assert rec.images[1].point2D(0).point3D_id==point
    np.testing.assert_allclose(rec.images[2].projection_center(),[1,0,0])


def test_pnp_projection_and_camera_copy_contract():
    import pycolmap as pc
    camera=pc.Camera(model="SIMPLE_RADIAL",width=640,height=480,params=np.array([500,320,240,0.]))
    copied=pc.Camera(camera.todict())
    points=np.random.default_rng(3).uniform([-1,-1,4],[1,1,8],size=(100,3))
    pose=pc.Rigid3d(np.c_[np.eye(3),[.2,-.1,.3]])
    xy=copied.img_from_cam(pose*points)
    estimation=pc.AbsolutePoseEstimationOptions();estimation.estimate_focal_length=False;estimation.ransac.max_error=2
    refinement=pc.AbsolutePoseRefinementOptions();refinement.refine_focal_length=False;refinement.refine_extra_params=False
    result=pc.estimate_and_refine_absolute_pose(xy,points,copied,estimation,refinement)
    assert result is not None and result["num_inliers"]>=95
    np.testing.assert_allclose(result["cam_from_world"].matrix(),pose.matrix(),atol=1e-6)
    np.testing.assert_array_equal(copied.params,camera.params)


def test_bounded_bundle_adjustment_fixes_context_and_intrinsics():
    import pycolmap as pc
    from alignexp.colmap_io import add_camera_and_rig, add_view
    rec = pc.Reconstruction()
    add_camera_and_rig(rec,1,1,"SIMPLE_RADIAL",640,480,[500,320,240,0])
    xyz = np.random.default_rng(8).uniform([-1,-1,4],[1,1,8],size=(30,3))
    for iid,tx in ((1,0),(2,-1),(3,1)):
        true = pc.Rigid3d(np.c_[np.eye(3),[tx,0,0]])
        xy = rec.cameras[1].img_from_cam(true*xyz)
        pose = pc.Rigid3d(np.c_[np.eye(3),[tx+(0.03 if iid==3 else 0),0,0]])
        add_view(rec,iid,f"{iid}.jpg",1,1,xy,pose)
    ids = [rec.add_point3D(p,pc.Track([pc.TrackElement(i,j) for i in (1,2,3)])) for j,p in enumerate(xyz)]
    ba = pc.BundleAdjustmentConfig()
    for iid in (1,2,3):
        ba.add_image(iid)
    for iid in (1,2):
        ba.set_constant_rig_from_world_pose(iid)
    for pid in ids:
        ba.add_constant_point(pid)
    ba.set_constant_cam_intrinsics(1)
    opts = pc.BundleAdjustmentOptions()
    opts.refine_focal_length=False;opts.refine_principal_point=False;opts.refine_extra_params=False
    opts.use_gpu=False;opts.refine_sensor_from_rig=False
    opts.loss_function_type=pc.LossFunctionType.SOFT_L1
    opts.solver_options.max_num_iterations=15;opts.solver_options.num_threads=1
    summary=pc.create_default_bundle_adjuster(opts,ba,rec).solve()
    assert summary.IsSolutionUsable() and summary.final_cost < summary.initial_cost
    np.testing.assert_allclose(rec.images[1].cam_from_world().translation,[0,0,0],atol=1e-12)
    np.testing.assert_allclose(rec.images[3].cam_from_world().translation,[1,0,0],atol=1e-5)


def test_native_merge_false_success_is_rejected(tmp_path):
    import subprocess
    import pycolmap as pc
    from alignexp.colmap_io import add_camera_and_rig, add_view
    from alignexp.fusion import verify_image_union
    paths=[]
    for tag,first in (("source",1),("reference",100)):
        rec=pc.Reconstruction()
        add_camera_and_rig(rec,first,first,"SIMPLE_RADIAL",640,480,[500,320,240,0])
        add_view(rec,first,tag+"a.jpg",first,first,[[320,240]],pc.Rigid3d())
        add_view(rec,first+1,tag+"b.jpg",first,first,[[220,240]],pc.Rigid3d(np.c_[np.eye(3),[-1,0,0]]))
        rec.add_point3D([0,0,5],pc.Track([pc.TrackElement(first,0),pc.TrackElement(first+1,0)]))
        path=tmp_path/tag;path.mkdir();rec.write(path);paths.append(path)
    output=tmp_path/"merged";output.mkdir()
    process=subprocess.run(["colmap","model_merger","--input_path1",str(paths[0]),"--input_path2",str(paths[1]),
                            "--output_path",str(output)],capture_output=True,text=True)
    assert process.returncode==0
    rec=pc.Reconstruction(output)
    with pytest.raises(RuntimeError,match="exact registered image union"):
        verify_image_union(rec,{1,2},{100,101})


def test_distinct_same_image_features_are_not_exact_duplicate_track_elements():
    import pycolmap as pc
    from alignexp.colmap_io import add_camera_and_rig, add_view
    from alignexp.audit import reciprocal_tracks
    rec=pc.Reconstruction()
    add_camera_and_rig(rec,1,1,"SIMPLE_RADIAL",640,480,[500,320,240,0])
    add_view(rec,1,"a.jpg",1,1,[[320,240],[320.1,240.1]],pc.Rigid3d())
    add_view(rec,2,"b.jpg",1,1,[[220,240]],pc.Rigid3d(np.c_[np.eye(3),[-1,0,0]]))
    rec.add_point3D([0,0,5],pc.Track([pc.TrackElement(1,0),pc.TrackElement(1,1),pc.TrackElement(2,0)]))
    result=reciprocal_tracks(rec)
    assert result["duplicate_image_observations"]==1
    assert result["exact_duplicate_elements"]==0
