from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COCO 1-based id -> имя. Цель хакатона — грузовик.
VEHICLE_LABELS = {
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

PREFERRED_LABELS = ("truck", "bus", "car")


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


class VehicleDetector:
    """2D-детектор транспорта: torchvision Faster R-CNN ResNet50 (BSD-3, COCO).

    Код и веса можно свободно использовать в своих проектах.
    3D-положение считается отдельно по центру бокса и калибровке камер.
    """

    def __init__(
        self,
        score_threshold: float = 0.35,
        device: str | None = None,
        max_box_frac: float = 0.25,
    ) -> None:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )

        self.score_threshold = score_threshold
        self.max_box_frac = max_box_frac
        self.device = torch.device(device or _default_device())
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        self._transforms = weights.transforms()
        self._torch = torch

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        from PIL import Image

        height, width = image_bgr.shape[:2]
        image_area = float(width * height)
        rgb = image_bgr[:, :, ::-1]
        tensor = self._transforms(Image.fromarray(rgb)).to(self.device)
        with self._torch.inference_mode():
            output = self.model([tensor])[0]

        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()

        detections: list[Detection] = []
        for box, score, label_id in zip(boxes, scores, labels, strict=True):
            label_id_int = int(label_id)
            if float(score) < self.score_threshold:
                continue
            name = VEHICLE_LABELS.get(label_id_int)
            if name is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            detection = Detection(
                bbox=(x1, y1, x2, y2),
                score=float(score),
                label=name,
                label_id=label_id_int,
            )
            if detection.area > self.max_box_frac * image_area:
                continue
            detections.append(detection)
        return detections

    def detect_best(self, image_bgr: np.ndarray) -> Detection | None:
        detections = self.detect(image_bgr)
        if not detections:
            return None
        rank = {name: index for index, name in enumerate(PREFERRED_LABELS)}
        return min(
            detections,
            key=lambda item: (rank.get(item.label, len(rank)), -item.score),
        )


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
