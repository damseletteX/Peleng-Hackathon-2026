from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mv3d.calibration.camera_model import Camera

# Пиксели: дальше этой ошибки вес камеры убывает как scale / error.
_HUBER_PX = 12.0


def triangulate_points(
    cameras: list[Camera],
    pixels: list[np.ndarray],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Взвешенная триангуляция DLT.

    Строки камеры умножаются на корень её веса. После первого решения
    вес дополнительно падает, если луч плохо сходится с остальными.
    """
    if len(cameras) != len(pixels) or len(cameras) < 2:
        raise ValueError("Для триангуляции нужны минимум две пары камера-точка.")

    base = np.ones(len(cameras), dtype=np.float64) if weights is None else np.asarray(
        weights, dtype=np.float64
    )
    if base.shape != (len(cameras),):
        raise ValueError("Число весов должно совпадать с числом камер.")
    base = np.clip(base, 0.0, None)
    if float(base.sum()) <= 0:
        base = np.ones(len(cameras), dtype=np.float64)

    rows: list[np.ndarray] = []
    for camera, uv in zip(cameras, pixels, strict=True):
        u, v = np.asarray(uv, dtype=np.float64)
        projection = camera.P
        rows.append(u * projection[2] - projection[0])
        rows.append(v * projection[2] - projection[1])
    design = np.stack(rows, axis=0)

    camera_weights = base.copy()
    point = _solve_weighted(design, camera_weights)
    for _ in range(2):
        adjusted: list[float] = []
        for camera, uv, weight in zip(cameras, pixels, base, strict=True):
            projected = camera.project(point)
            error = float(np.linalg.norm(projected - np.asarray(uv, dtype=np.float64)))
            huber = 1.0 if error <= _HUBER_PX else _HUBER_PX / error
            adjusted.append(float(weight) * huber)
        camera_weights = np.asarray(adjusted, dtype=np.float64)
        if float(camera_weights.sum()) <= 0:
            camera_weights = base.copy()
        point = _solve_weighted(design, camera_weights)
    return point


def _solve_weighted(design: np.ndarray, camera_weights: np.ndarray) -> np.ndarray:
    row_weights = np.repeat(np.clip(camera_weights, 0.0, None), 2)
    scaled = design * np.sqrt(row_weights)[:, None]
    _, _, vh = np.linalg.svd(scaled)
    point = vh[-1]
    scale = point[3] if abs(point[3]) > 1e-12 else 1.0
    return point[:3] / scale


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
