from __future__ import annotations

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from mv3d.detection.detector import Detection


def make_truck_tracker(frame_rate: float) -> ByteTrackTracker:
    """ByteTrack с буфером 2 с, чтобы бокс не обрывался на краю кадра.

    Слабые детекции не открывают новый ID: они только продолжают трек.
    """
    return ByteTrackTracker(
        lost_track_buffer=60,
        frame_rate=frame_rate,
        track_activation_threshold=0.25,
        minimum_consecutive_frames=1,
        minimum_iou_threshold=0.05,
        high_conf_det_threshold=0.35,
    )


def detections_to_supervision(detections: list[Detection]) -> sv.Detections:
    if not detections:
        empty = sv.Detections.empty()
        empty.confidence = np.zeros((0,), dtype=np.float32)
        empty.data["class_name"] = np.array([], dtype=str)
        return empty
    tracked = sv.Detections(
        xyxy=np.array([item.bbox for item in detections], dtype=np.float32),
        confidence=np.array([item.score for item in detections], dtype=np.float32),
        class_id=np.array([item.label_id for item in detections], dtype=int),
    )
    tracked.data["class_name"] = np.array([item.label for item in detections], dtype=str)
    return tracked


def detection_from_supervision(detections: sv.Detections, index: int) -> Detection:
    x1, y1, x2, y2 = (float(v) for v in detections.xyxy[index])
    score = 0.0 if detections.confidence is None else float(detections.confidence[index])
    label_id = 8 if detections.class_id is None else int(detections.class_id[index])
    names = detections.data.get("class_name") if detections.data else None
    label = "truck" if names is None else str(names[index])
    return Detection(bbox=(x1, y1, x2, y2), score=score, label=label, label_id=label_id)


def track_id_at(detections: sv.Detections, index: int) -> int:
    if detections.tracker_id is None:
        return -1
    return int(detections.tracker_id[index])


def coasting_boxes(tracker: ByteTrackTracker) -> list[tuple[int, np.ndarray]]:
    """Предсказанные боксы треков, у которых на этом кадре нет детекции."""
    boxes: list[tuple[int, np.ndarray]] = []
    for track in tracker.tracks:
        if track.tracker_id < 0 or track.time_since_update <= 0:
            continue
        boxes.append((int(track.tracker_id), np.asarray(track.get_state_bbox(), dtype=np.float64)))
    return boxes
