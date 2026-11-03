from __future__ import annotations

import numpy as np

from mv3d.calibration.camera_model import Camera
from mv3d.detection.detector import Detection
from mv3d.detection.keypoints import bbox_center, bbox_ground_point
from mv3d.geometry.triangulate import intersect_ray_plane, triangulate_points


def localize_detection(
    cameras: list[Camera],
    detections: list[Detection | None],
    plane_z: float = 300.0,
    use_ground_point: bool = False,
) -> np.ndarray | None:
    """3D-координаты объекта по 2D-детекции и калибровке.

    Две и более камеры — DLT-триангуляция центров боксов.
    Одна камера — пересечение луча с плоскостью Z = plane_z (высота объекта в Unreal, см).
    """
    valid = [
        (camera, detection)
        for camera, detection in zip(cameras, detections, strict=True)
        if detection is not None
    ]
    if not valid:
        return None

    pick = bbox_ground_point if use_ground_point else bbox_center
    pixels = [pick(detection) for _, detection in valid]
    used_cameras = [camera for camera, _ in valid]

    if len(used_cameras) >= 2:
        return triangulate_points(used_cameras, pixels)

    origin, direction = used_cameras[0].ray_from_pixel(pixels[0])
    return intersect_ray_plane(origin, direction, plane_z=plane_z)
