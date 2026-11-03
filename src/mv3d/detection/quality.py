<<<<<<< HEAD
Detection(
    bbox=(x1, y1, x2, y2),
    score=0.93,
    label="truck",
    label_id=8,
    quality_score=0.82,
)
=======
from __future__ import annotations

import cv2
import numpy as np

from mv3d.detection.detector import Detection

# Дисперсия Лапласиана, при которой резкость даёт вес 0.5.
_SHARPNESS_SCALE = 80.0
_WEIGHT_FLOOR = 0.02


def visibility_weight(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    margin: float = 12.0,
) -> float:
    """1, если бокс целиком внутри кадра, и меньше, если он обрезан границей.

    Обрезанный бокс смещает центр в видимый кусок, поэтому такой вид
    почти не должен тянуть 3D-точку.
    """
    x1, y1, x2, y2 = bbox
    edge = min(x1, y1, (width - 1) - x2, (height - 1) - y2)
    if edge >= margin:
        return 1.0
    if margin <= 0:
        return _WEIGHT_FLOOR
    return float(np.clip(edge / margin, _WEIGHT_FLOOR, 1.0))


def sharpness_weight(
    image_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> float:
    """Вес резкости кропа бокса. Размытый или смазанный грузовик весит меньше."""
    height, width = image_bgr.shape[:2]
    x1 = int(np.floor(max(0.0, bbox[0])))
    y1 = int(np.floor(max(0.0, bbox[1])))
    x2 = int(np.ceil(min(float(width), bbox[2])))
    y2 = int(np.ceil(min(float(height), bbox[3])))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return _WEIGHT_FLOOR

    crop = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if not np.isfinite(variance) or variance <= 0:
        return _WEIGHT_FLOOR
    return float(np.clip(variance / (variance + _SHARPNESS_SCALE), _WEIGHT_FLOOR, 1.0))


def observation_weight(
    image_bgr: np.ndarray,
    detection: Detection,
    margin: float = 12.0,
) -> tuple[float, float, float]:
    """Вес вида: уверенность × резкость кропа × доля бокса внутри кадра.

    Возвращает общий вес и две составляющие, чтобы их было видно в CSV.
    Остаточный вес по рассогласованию лучей считается в триангуляции.
    """
    height, width = image_bgr.shape[:2]
    visibility = visibility_weight(detection.bbox, width, height, margin)
    sharpness = sharpness_weight(image_bgr, detection.bbox)
    score = float(np.clip(detection.score, 0.0, 1.0))
    return score * sharpness * visibility, sharpness, visibility
>>>>>>> a845abdf803650b21ea2d9221a222c5ddafbff9a
