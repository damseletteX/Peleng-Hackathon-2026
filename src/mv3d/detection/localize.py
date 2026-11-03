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
    weights: list[float] | None = None,
) -> np.ndarray | None:
    """3D-координаты объекта по 2D-детекции и калибровке.

    Точка измерения — центр бокса: в калибровке актор лежит на Z = plane_z.
    Две и более камеры — взвешенный DLT. Одна камера — луч × эта плоскость.
    Нулевой вес выключает камеру. Если положительный вес только у одной,
    используется луч, а не триангуляция.
    """
    valid: list[tuple[Camera, Detection, float]] = []
    for index, (camera, detection) in enumerate(zip(cameras, detections, strict=True)):
        if detection is None:
            continue
        weight = 1.0 if weights is None else float(weights[index])
        valid.append((camera, detection, weight))
    if not valid:
        return None

    positive = [item for item in valid if item[2] > 0]
    if not positive:
        positive = [(camera, detection, 1.0) for camera, detection, _ in valid]

    pick = bbox_ground_point if use_ground_point else bbox_center
    pixels = [pick(detection) for _, detection, _ in positive]
    used_cameras = [camera for camera, _, _ in positive]
    used_weights = [weight for _, _, weight in positive]

    if len(used_cameras) >= 2:
        return triangulate_points(used_cameras, pixels, used_weights)

    origin, direction = used_cameras[0].ray_from_pixel(pixels[0])
    return intersect_ray_plane(origin, direction, plane_z=plane_z)
