import numpy as np

from mv3d.calibration.camera_model import load_camera_from_csv
from mv3d.detection.detector import Detection
from mv3d.tracking.motion import TrajectoryFilter
from mv3d.tracking.pipeline import View, _draw_heading, _views_for_fusion


def _view(camera, pixel, bbox, weight):
    return View(
        camera_id=camera.camera_id,
        camera=camera,
        detection=Detection(bbox=bbox, score=weight, label="truck", label_id=8),
        track_id=1,
        pixel=np.asarray(pixel, dtype=np.float64),
        weight=weight,
        sharpness=1.0,
        visibility=1.0,
    )


def test_a_view_away_from_the_truck_is_not_fused():
    camera = load_camera_from_csv(
        "configs/calibration/object_transform_CineCameraActor1.csv",
        "cam1",
        frame=80,
    )
    truck = np.array([-16000.0, -10000.0, 300.0])
    seen = camera.project(truck)
    good = _view(
        camera,
        seen,
        (seen[0] - 40, seen[1] - 30, seen[0] + 40, seen[1] + 30),
        0.4,
    )
    bad = _view(camera, np.array([80.0, 80.0]), (40, 40, 140, 160), 0.95)
    chosen, locked = _views_for_fusion([bad, good], 300.0, truck)
    assert chosen == [good]
    assert locked == "cam1"


def test_disagreeing_views_do_not_start_a_joint_track():
    camera = load_camera_from_csv(
        "configs/calibration/object_transform_CineCameraActor1.csv",
        "cam1",
        frame=80,
    )
    heavy = _view(camera, np.array([200.0, 400.0]), (100, 300, 300, 500), 0.9)
    light = _view(camera, np.array([1100.0, 200.0]), (1000, 100, 1200, 300), 0.2)
    chosen, _locked = _views_for_fusion([light, heavy], 300.0, None)
    assert len(chosen) == 1
    assert chosen[0].weight == 0.9


def test_draw_heading_uses_the_motion_vector():
    motion = TrajectoryFilter()
    motion.x = np.array([0.0, 0.0, 0.0, 400.0, 0.0, 0.0])
    fallback = np.array([0.0, 1.0])
    body = _draw_heading(motion, fallback)
    assert body is not None
    assert np.allclose(body, [1.0, 0.0])


def test_draw_heading_falls_back_when_the_truck_is_slow():
    motion = TrajectoryFilter()
    motion.x = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
    fallback = np.array([0.0, 1.0])
    body = _draw_heading(motion, fallback)
    assert body is not None
    assert np.allclose(body, fallback)
