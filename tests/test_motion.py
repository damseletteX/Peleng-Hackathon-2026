import numpy as np

from mv3d.calibration.camera_model import Camera, CameraIntrinsics, load_camera_from_csv
from mv3d.geometry.triangulate import triangulate_points
from mv3d.tracking.motion import TrajectoryFilter


def test_low_weight_moves_the_trajectory_less():
    measurement = np.array([100.0, 0.0, 300.0])
    trusted = TrajectoryFilter()
    trusted.step(np.array([0.0, 0.0, 300.0]), 1.0, 0.1)
    trusted_next = trusted.step(measurement, 1.0, 0.1)

    cautious = TrajectoryFilter()
    cautious.step(np.array([0.0, 0.0, 300.0]), 1.0, 0.1)
    cautious_next = cautious.step(measurement, 0.02, 0.1)

    assert trusted_next is not None and cautious_next is not None
    assert trusted_next[0] > cautious_next[0]


def test_sideways_jump_stays_near_the_driven_path():
    motion = TrajectoryFilter()
    motion.step(np.array([0.0, 0.0, 300.0]), 1.0, 0.1)
    for index in range(1, 12):
        motion.step(np.array([index * 50.0, 0.0, 300.0]), 1.0, 0.1)
    before = motion.position
    assert before is not None
    after = motion.step(np.array([before[0] + 50.0, 500.0, 300.0]), 1.0, 0.1)
    assert after is not None
    assert abs(after[1]) < 250.0


def test_unseen_speed_keeps_going_along_the_path():
    motion = TrajectoryFilter()
    motion.step(np.zeros(3), 1.0, 0.1)
    for index in range(1, 15):
        motion.step(np.array([index * 80.0, 0.0, 0.0]), 1.0, 0.1)
    speed = float(np.hypot(motion.x[3], motion.x[4]))
    before = motion.position
    after = motion.step(None, 0.0, 1.0)
    assert before is not None and after is not None
    assert after[0] > before[0]
    assert abs(float(np.hypot(motion.x[3], motion.x[4])) - speed) < 80.0


def test_missing_measurement_keeps_a_position():
    motion = TrajectoryFilter()
    first = motion.step(np.array([10.0, 20.0, 300.0]), 1.0, 0.1)
    second = motion.step(None, 0.0, 0.1)
    assert first is not None and second is not None
    assert np.allclose(second[:3], first[:3])


def test_disagreement_downweights_the_bad_camera():
    cam_a = Camera(
        camera_id="a",
        intrinsics=CameraIntrinsics(1280, 720, 35.0),
        center=np.array([-24701.9592, 6623.5717, 1200.0]),
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=-70.0,
    )
    cam_b = load_camera_from_csv(
        "configs/calibration/object_transform_CineCameraActor2.csv",
        "cam2",
        frame=150,
    )
    point = np.array([-13889.5241, -7666.0552, 300.0])
    good = cam_a.project(point)
    bad = cam_b.project(point) + np.array([80.0, -60.0])
    equal = triangulate_points([cam_a, cam_b], [good, bad])
    weighted = triangulate_points([cam_a, cam_b], [good, bad], [1.0, 0.01])
    assert np.linalg.norm(weighted - point) < np.linalg.norm(equal - point)
