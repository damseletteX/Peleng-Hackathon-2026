from __future__ import annotations

import numpy as np

from mv3d.detection.detector import Detection

# Где в боксе сидит точка эталона на видимых кадрах cam1:
# 96.4% ширины к носу, 59.8% высоты сверху. Луч через это место
# на плоскости Z эталона даёт около метра ошибки, пока грузовик в кадре.
_ACTOR_FRONT = 0.964
_ACTOR_DOWN = 0.598


def bbox_center(detection: Detection) -> np.ndarray:
    x1, y1, x2, y2 = detection.bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)


def actor_pixel(
    detection: Detection,
    motion_uv: np.ndarray | None = None,
) -> np.ndarray:
    """Пиксель начала координат актора внутри бокса."""
    x1, y1, x2, y2 = detection.bbox
    width = x2 - x1
    height = y2 - y1
    down = y1 + _ACTOR_DOWN * height
    front_from_left = x1 + _ACTOR_FRONT * width
    front_from_right = x2 - _ACTOR_FRONT * width
    if motion_uv is None:
        return np.array([front_from_left, down], dtype=np.float64)
    du, dv = (float(v) for v in np.asarray(motion_uv, dtype=np.float64)[:2])
    if abs(du) < 0.2 and abs(dv) < 0.2:
        return np.array([front_from_left, down], dtype=np.float64)
    if abs(du) >= abs(dv):
        u = front_from_left if du > 0 else front_from_right
        return np.array([u, down], dtype=np.float64)
    v = y1 + _ACTOR_FRONT * height if dv > 0 else y2 - _ACTOR_FRONT * height
    return np.array([(x1 + x2) / 2.0, v], dtype=np.float64)


def reference_pixel(
    detection: Detection,
    motion_uv: np.ndarray | None = None,
) -> np.ndarray:
    return actor_pixel(detection, motion_uv)


def bbox_ground_point(detection: Detection) -> np.ndarray:
    """Нижняя середина бокса — ближе к контакту колёс с землёй."""
    x1, _, x2, y2 = detection.bbox
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float64)
