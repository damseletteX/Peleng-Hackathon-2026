import numpy as np

from mv3d.calibration.camera_model import load_camera_from_csv
from mv3d.detection.detector import Detection
from mv3d.geometry.box3d import _projected_aabb, fit_actor_origin, ground_seed


def test_fit_puts_the_nose_on_the_projected_box():
    camera = load_camera_from_csv(
        "configs/calibration/object_transform_CineCameraActor1.csv",
        "cam1",
        frame=80,
        focal_mm=35.0,
    )
    origin = np.array([-16000.0, -10000.0, 300.0])
    heading = np.array([0.85, 0.53])
    heading = heading / np.hypot(*heading)
    aabb = _projected_aabb(camera, origin, heading)
    assert aabb is not None
    detection = Detection(
        bbox=(float(aabb[0]), float(aabb[1]), float(aabb[2]), float(aabb[3])),
        score=0.9,
        label="truck",
        label_id=8,
    )
    seed = origin + np.array([600.0, -400.0, 0.0])
    fitted = fit_actor_origin(camera, detection, heading, 300.0, seed)
    assert fitted is not None
    assert float(np.hypot(fitted[0] - origin[0], fitted[1] - origin[1])) < 150.0


def test_ground_seed_follows_the_bottom_of_the_box():
    camera = load_camera_from_csv(
        "configs/calibration/object_transform_CineCameraActor1.csv",
        "cam1",
        frame=80,
        focal_mm=35.0,
    )
    detection = Detection(bbox=(600.0, 400.0, 900.0, 500.0), score=0.8, label="truck", label_id=8)
    seed = ground_seed(camera, detection, 300.0)
    assert seed is not None
    assert seed[2] == 300.0


def test_draw_truck_paints_the_wireframe():
    camera = load_camera_from_csv(
        "configs/calibration/object_transform_CineCameraActor1.csv",
        "cam1",
        frame=80,
        focal_mm=35.0,
    )
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    from mv3d.geometry.box3d import draw_truck

    draw_truck(canvas, camera, np.array([-16000.0, -10000.0, 300.0]), np.array([0.85, 0.53]))
    assert int(canvas.max()) > 0
