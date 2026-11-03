from mv3d.detection.detector import PREFERRED_LABELS, VEHICLE_LABELS, Detection
from mv3d.detection.keypoints import bbox_center
from mv3d.detection.localize import localize_detection


def test_vehicle_label_ids_are_coco():
    assert VEHICLE_LABELS[8] == "truck"
    assert "truck" in PREFERRED_LABELS


def test_bbox_center():
    detection = Detection(bbox=(10.0, 20.0, 50.0, 80.0), score=0.9, label="truck", label_id=8)
    center = bbox_center(detection)
    assert list(center) == [30.0, 50.0]


def test_localize_returns_none_without_detections():
    assert localize_detection([], []) is None
