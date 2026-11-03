from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# COCO id грузовика. Предобученный nano отдаёт этот id, дообученный — имя класса.
TRUCK_LABEL_ID = 8
VEHICLE_LABELS = {TRUCK_LABEL_ID: "truck"}
PREFERRED_LABELS = ("truck",)
FINETUNED_WEIGHTS = Path("weights/rfdetr-nano-truck.pth")


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    score: float
    label: str
    label_id: int

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_truck_boxes(
    detections: list[Detection],
    image_area: float,
    max_box_frac: float = 0.25,
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Убрать бокс на весь кадр и оставить один бокс на один и тот же грузовик."""
    compact = [
        item
        for item in detections
        if item.label == "truck" and item.area <= max_box_frac * image_area
    ]
    compact.sort(key=lambda item: item.score, reverse=True)
    kept: list[Detection] = []
    for item in compact:
        if all(_iou(item, other) < iou_threshold for other in kept):
            kept.append(item)
    return kept


def _iou(left: Detection, right: Detection) -> float:
    ax1, ay1, ax2, ay2 = left.bbox
    bx1, by1, bx2, by2 = right.bbox
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = left.area + right.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


class VehicleDetector:
    """RF-DETR Nano, класс truck.

    Если лежит дообученный чекпоинт, берётся он. Иначе — предобученные веса COCO.
    """

    def __init__(
        self,
        score_threshold: float = 0.15,
        *,
        pretrained: bool = False,
        weights: str | Path | None = None,
    ) -> None:
        import rfdetr

        self.score_threshold = score_threshold
        self.weights_path = _resolve_weights(pretrained=pretrained, weights=weights)
        if self.weights_path is None:
            self.model = rfdetr.RFDETRNano()
        else:
            self.model = rfdetr.from_checkpoint(self.weights_path, trust_checkpoint=True)

    def detect(
        self,
        image_bgr: np.ndarray,
        score_threshold: float | None = None,
    ) -> list[Detection]:
        threshold = self.score_threshold if score_threshold is None else score_threshold
        detections = self._detect_rfdetr(image_bgr, threshold)
        height, width = image_bgr.shape[:2]
        return select_truck_boxes(detections, float(width * height))

    def _detect_rfdetr(self, image_bgr: np.ndarray, threshold: float) -> list[Detection]:
        import cv2
        from rfdetr.assets.coco_classes import COCO_CLASSES

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        raw = self.model.predict(rgb, threshold=threshold)
        if raw.xyxy is None or len(raw.xyxy) == 0:
            return []

        names = raw.data.get("class_name") if raw.data else None
        if names is None:
            names = [COCO_CLASSES.get(int(class_id), "") for class_id in raw.class_id]
        confidences = raw.confidence
        if confidences is None:
            confidences = np.ones(len(raw.xyxy), dtype=np.float64)

        detections: list[Detection] = []
        for box, score, class_id, name in zip(
            raw.xyxy, confidences, raw.class_id, names, strict=True
        ):
            if str(name) != "truck":
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    score=float(score),
                    label="truck",
                    label_id=int(class_id) if int(class_id) > 0 else TRUCK_LABEL_ID,
                )
            )
        return detections

    def detect_best(self, image_bgr: np.ndarray) -> Detection | None:
        detections = self.detect(image_bgr)
        if not detections:
            return None
        return max(detections, key=lambda item: item.score)


def _resolve_weights(*, pretrained: bool, weights: str | Path | None) -> Path | None:
    if weights is not None:
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"Нет весов: {path}")
        return path
    if pretrained:
        return None
    if FINETUNED_WEIGHTS.is_file():
        return FINETUNED_WEIGHTS
    return None
