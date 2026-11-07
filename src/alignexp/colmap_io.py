import numpy as np
import pycolmap as pc


def add_view(rec, iid, name, cid, rid, xy, pose):
    """Single-camera rig contract; pose and observations are separate operations."""
    image = pc.Image(name=name, keypoints=np.asarray(xy, dtype=float), camera_id=int(cid), image_id=int(iid))
    image.frame_id = int(iid)
    frame = pc.Frame(frame_id=int(iid), rig_id=int(rid))
    frame.add_data_id(image.data_id)
    rec.add_frame(frame)
    rec.add_image(image)
    rec.frames[int(iid)].set_cam_from_world(int(cid), pose)
    rec.register_frame(int(iid))  # Avoid the faulty ObservationManager binding in 3.13.


def add_camera_and_rig(rec, cid, rid, model, width, height, params):
    camera = pc.Camera(camera_id=int(cid), model=model, width=int(width), height=int(height), params=np.asarray(params,dtype=float))
    rec.add_camera(camera)
    rig = pc.Rig(rig_id=int(rid))
    rig.add_ref_sensor(camera.sensor_id)
    rec.add_rig(rig)


def sim3_object(S):
    return pc.Sim3d(np.asarray(S)[:3,:])
