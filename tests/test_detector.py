from mv3d.detection.detector import PREFERRED_LABELS, VEHICLE_LABELS, Detection
from mv3d.detection.keypoints import bbox_center
from mv3d.detection.localize import localize_detection


def test_vehicle_label_ids_are_coco():
    assert VEHICLE_LABELS[8] == "truck"
    assert "truck" in PREFERRED_LABELS


def test_full_frame_and_duplicate_boxes_are_dropped():
    from mv3d.detection.detector import select_truck_boxes

    image_area = 1280 * 720
    full = Detection(bbox=(0.0, 0.0, 1270.0, 710.0), score=0.9, label="truck", label_id=8)
    strong = Detection(bbox=(380.0, 510.0, 540.0, 630.0), score=0.5, label="truck", label_id=8)
    duplicate = Detection(bbox=(382.0, 512.0, 542.0, 628.0), score=0.2, label="truck", label_id=8)
    other = Detection(bbox=(900.0, 100.0, 980.0, 180.0), score=0.4, label="truck", label_id=8)
    kept = select_truck_boxes([full, duplicate, other, strong], image_area)
    assert [item.score for item in kept] == [0.5, 0.4]


def test_reference_pixel_follows_motion():
    from mv3d.detection.keypoints import reference_pixel

    detection = Detection(bbox=(100.0, 200.0, 300.0, 400.0), score=0.9, label="truck", label_id=8)
    assert list(reference_pixel(detection, [2.0, 0.1])) == [300.0, 300.0]
    assert list(reference_pixel(detection, [-2.0, 0.1])) == [100.0, 300.0]


def test_bbox_center():
    detection = Detection(bbox=(10.0, 20.0, 50.0, 80.0), score=0.9, label="truck", label_id=8)
    center = bbox_center(detection)
    assert list(center) == [30.0, 50.0]


def test_localize_returns_none_without_detections():
    assert localize_detection([], []) is None
