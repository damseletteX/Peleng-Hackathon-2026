from __future__ import annotations

import numpy as np

from mv3d.detection.detector import Detection


def bbox_center(detection: Detection) -> np.ndarray:
    x1, y1, x2, y2 = detection.bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)


def bbox_ground_point(detection: Detection) -> np.ndarray:
    """Нижняя середина бокса — ближе к контакту колёс с землёй."""
    x1, _, x2, y2 = detection.bbox
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float64)
