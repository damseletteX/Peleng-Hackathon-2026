from __future__ import annotations

import numpy as np

from mv3d.detection.detector import Detection


def bbox_center(detection: Detection) -> np.ndarray:
    x1, y1, x2, y2 = detection.bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)


def reference_pixel(
    detection: Detection,
    motion_uv: np.ndarray | None = None,
) -> np.ndarray:
    """Точка актора на боксе.

    Пока направление неизвестно, это центр. Когда грузовик уже едет,
    берётся середина передней грани: в этой сцене начало координат
    актора сидит на носу, а не в центре кузова.
    """
    x1, y1, x2, y2 = detection.bbox
    midpoint = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)
    if motion_uv is None:
        return midpoint
    du, dv = (float(v) for v in np.asarray(motion_uv, dtype=np.float64)[:2])
    if abs(du) < 0.2 and abs(dv) < 0.2:
        return midpoint
    if abs(du) >= abs(dv):
        return np.array([x2 if du > 0 else x1, midpoint[1]], dtype=np.float64)
    return np.array([midpoint[0], y2 if dv > 0 else y1], dtype=np.float64)


def bbox_ground_point(detection: Detection) -> np.ndarray:
    """Нижняя середина бокса — ближе к контакту колёс с землёй."""
    x1, _, x2, y2 = detection.bbox
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float64)
