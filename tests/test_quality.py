import numpy as np

from mv3d.detection.detector import Detection
from mv3d.detection.quality import observation_weight, sharpness_weight, visibility_weight


def _truck(bbox, score=0.8) -> Detection:
    return Detection(bbox=bbox, score=score, label="truck", label_id=8)


def test_full_box_is_fully_visible():
    weight = visibility_weight((100.0, 80.0, 400.0, 500.0), 1280, 720)
    assert weight == 1.0


def test_border_box_is_downweighted():
    weight = visibility_weight((0.0, 80.0, 40.0, 200.0), 1280, 720)
    assert weight < 0.1


def test_blurry_crop_weighs_less_than_textured_crop():
    detection = _truck((10.0, 10.0, 90.0, 90.0))
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    noise = np.random.default_rng(0).integers(0, 255, blank.shape, dtype=np.uint8)
    assert sharpness_weight(noise, detection.bbox) > sharpness_weight(blank, detection.bbox)


def test_observation_weight_includes_score():
    image = np.random.default_rng(1).integers(0, 255, (120, 120, 3), dtype=np.uint8)
    low = _truck((20.0, 20.0, 100.0, 100.0), score=0.2)
    high = _truck((20.0, 20.0, 100.0, 100.0), score=0.8)
    low_weight, _, _ = observation_weight(image, low)
    high_weight, _, _ = observation_weight(image, high)
    assert high_weight > low_weight
