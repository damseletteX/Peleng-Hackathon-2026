import numpy as np

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


def test_detect_batch_filters_each_frame_from_one_call():
    from mv3d.detection.detector import VehicleDetector

    calls: list[tuple[int, float]] = []

    class Raw:
        def __init__(self, xyxy, confidence, names):
            self.xyxy = np.array(xyxy, dtype=np.float32)
            self.confidence = np.array(confidence, dtype=np.float64)
            self.class_id = np.array([8] * len(xyxy))
            self.data = {"class_name": np.array(names)}

    catalog = [
        Raw(
            [[10, 10, 40, 40], [0, 0, 90, 90]],
            [0.9, 0.8],
            ["truck", "truck"],
        ),
        Raw([], [], []),
        Raw([[1, 1, 5, 5]], [0.2], ["car"]),
    ]

    class Model:
        def predict(self, images, threshold, include_source_image=True):
            calls.append((len(images), threshold))
            assert include_source_image is False
            return catalog[: len(images)]

    detector = VehicleDetector.__new__(VehicleDetector)
    detector.score_threshold = 0.15
    detector.model = Model()
    images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    found = detector.detect_batch(images, score_threshold=0.05)

    assert calls == [(3, 0.05)]
    assert len(found) == 3
    assert [item.score for item in found[0]] == [0.9]
    assert found[1] == []
    assert found[2] == []
    assert detector.detect(images[0], score_threshold=0.05) == found[0]
