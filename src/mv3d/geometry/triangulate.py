from __future__ import annotations

import numpy as np

from mv3d.calibration.camera_model import Camera


def triangulate_points(cameras: list[Camera], pixels: list[np.ndarray]) -> np.ndarray:
    """Линейная триангуляция DLT по нескольким видам."""
    if len(cameras) != len(pixels) or len(cameras) < 2:
        raise ValueError("Для триангуляции нужны минимум две пары камера-точка.")

    rows: list[np.ndarray] = []
    for camera, uv in zip(cameras, pixels, strict=True):
        u, v = np.asarray(uv, dtype=np.float64)
        p = camera.P
        rows.append(u * p[2] - p[0])
        rows.append(v * p[2] - p[1])

    design = np.stack(rows, axis=0)
    _, _, vh = np.linalg.svd(design)
    point = vh[-1]
    point = point / point[3]
    return point[:3]


def intersect_ray_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_z: float = 300.0,
) -> np.ndarray | None:
    """Пересечение луча с горизонтальной плоскостью Z = plane_z (монокулярный 3D)."""
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if abs(direction[2]) < 1e-9:
        return None
    t = (plane_z - origin[2]) / direction[2]
    if t <= 0:
        return None
    return origin + t * direction
