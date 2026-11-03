import numpy as np

from mv3d.calibration.camera_model import (
    Camera,
    CameraIntrinsics,
    load_camera_from_csv,
    make_project_intrinsics,
    unreal_rotation_to_opencv_R,
)
from mv3d.geometry.triangulate import triangulate_points


def test_intrinsics_shape_and_principal_point():
    cameras = make_project_intrinsics()
    k = cameras["cam1"].K
    assert k.shape == (3, 3)
    assert np.isclose(k[0, 2], 640.0)
    assert np.isclose(k[1, 2], 360.0)


def test_unreal_yaw_only_matches_manual_basis():
    yaw = np.deg2rad(-70.0)
    expected = np.vstack(
        [
            np.array([-np.sin(yaw), np.cos(yaw), 0.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([np.cos(yaw), np.sin(yaw), 0.0]),
        ]
    )
    got = unreal_rotation_to_opencv_R(0.0, 0.0, -70.0)
    assert np.allclose(got, expected, atol=1e-9)


def test_cam1_projection_matches_known_camera_frame():
    camera = load_camera_from_csv(
        "configs/calibration/object_transform_CineCameraActor1.csv",
        "cam1",
        frame=150,
    )
    truck = np.array([-13889.5241, -7666.0552, 300.0])
    uv = camera.project(truck)
    assert (camera.R @ (truck - camera.center))[2] > 0
    assert np.allclose(uv, np.array([1220.5469, 459.0878]), atol=0.05)


def test_triangulation_recovers_synthetic_point():
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
    recovered = triangulate_points([cam_a, cam_b], [cam_a.project(point), cam_b.project(point)])
    assert np.linalg.norm(recovered - point) < 5.0
