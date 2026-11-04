from __future__ import annotations

import numpy as np

from mv3d.detection.detector import Detection

# Один кадр случайного бокса не удерживаем. Два подряд уже трек.
_START_HITS = 2
# Внутри кадра без детекции бокс живёт недолго: это уже не объект у края.
_INSIDE_HOLD = 8
# У края держим дольше: в кадре может оставаться узкая полоска.
_EDGE_HOLD = 90


class BoxKeeper:
    """Один бокс на камеру, без мигания на краю.

    Обрезок грузовика не сжимает уже найденный бокс. Пока от бокса
    виден хотя бы край, он остаётся в кадре.
    """

    def __init__(self) -> None:
        self.box: np.ndarray | None = None
        self.velocity = np.zeros(4, dtype=np.float64)
        self.miss = 0
        self.last_live: Detection | None = None
        self._pending: np.ndarray | None = None
        self._pending_hits = 0
        self._size: tuple[int, int] | None = None

    def wants_faint(self) -> bool:
        """Нужен повторный проход с низким порогом: объект на краю или только что пропал."""
        if self.box is None or self._size is None:
            return False
        width, height = self._size
        return self.miss > 0 or _touches_border(self.box, width, height)

    def update(
        self,
        detections: list[Detection],
        width: int,
        height: int,
    ) -> np.ndarray | None:
        self._size = (width, height)
        self.last_live = None
        match = self._match(detections)
        if self.box is None:
            return self._start(match, width, height)
        if match is not None and _is_sliver(match.bbox, self.box, width, height):
            self.box = self.box + self.velocity
            self.miss = 0
        elif match is not None:
            self._adopt(match)
        else:
            self.box = self.box + self.velocity
            self.miss += 1
        if self.box is None or _fully_outside(self.box, width, height):
            self.box = None
            return None
        limit = _EDGE_HOLD if _touches_border(self.box, width, height) else _INSIDE_HOLD
        if self.miss > limit:
            self.box = None
            return None
        return self.box.copy()

    def _start(self, match: Detection | None, width: int, height: int) -> np.ndarray | None:
        if match is None or match.score < 0.15:
            self._pending = None
            self._pending_hits = 0
            return None
        box = np.asarray(match.bbox, dtype=np.float64)
        if self._pending is not None and _iou(self._pending, box) > 0.3:
            self._pending_hits += 1
        else:
            self._pending = box
            self._pending_hits = 1
        if self._pending_hits >= _START_HITS:
            if self._pending is not None:
                self.velocity = box - self._pending
            self._adopt(match)
        if _fully_outside(box, width, height):
            return None
        return box.copy()

    def _adopt(self, detection: Detection) -> None:
        new = np.asarray(detection.bbox, dtype=np.float64)
        if self.box is not None:
            new = _clip_blurred_top(new, self.box)
            delta = new - self.box
            self.velocity = 0.6 * self.velocity + 0.4 * delta
            self.box = 0.75 * new + 0.25 * (self.box + self.velocity)
        else:
            self.box = new
        self.miss = 0
        self.last_live = Detection(
            bbox=(float(self.box[0]), float(self.box[1]), float(self.box[2]), float(self.box[3])),
            score=detection.score,
            label=detection.label,
            label_id=detection.label_id,
        )
        self._pending = None
        self._pending_hits = 0

    def _match(self, detections: list[Detection]) -> Detection | None:
        if not detections:
            return None
        if self.box is None:
            return max(detections, key=lambda item: item.score)
        predicted = self.box + self.velocity
        matched = [
            item
            for item in detections
            if _iou(predicted, item.bbox) > 0.05 or _center_inside(predicted, item.bbox)
        ]
        if not matched:
            return None
        return max(matched, key=lambda item: item.score)


def _clip_blurred_top(new: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Срезать верх, если он вытянулся один, без роста ширины.

    Подъезд увеличивает ширину и высоту вместе. Смаз поднимает только верх,
    пока низ стоит на месте. Высоту при этом не запоминаем: иначе бокс
    не успевает вырасти, когда грузовик действительно приближается.
    """
    prev_w = float(previous[2] - previous[0])
    prev_h = float(previous[3] - previous[1])
    if prev_w < 8.0 or prev_h < 8.0:
        return new
    width_ratio = float(new[2] - new[0]) / prev_w
    if width_ratio > 1.08:
        return new
    new_h = float(new[3] - new[1])
    top_up = float(previous[1] - new[1])
    bottom_shift = abs(float(new[3] - previous[3]))
    if new_h <= prev_h * 1.2 or top_up <= bottom_shift:
        return new
    allowed = prev_h * max(width_ratio, 1.0) * 1.05
    out = new.copy()
    out[1] = float(new[3]) - allowed
    return out


def _is_sliver(
    bbox: tuple[float, float, float, float],
    previous: np.ndarray,
    width: int,
    height: int,
) -> bool:
    box = np.asarray(bbox, dtype=np.float64)
    if not _touches_border(box, width, height):
        return False
    return _area(box) < 0.55 * _area(previous)


def _touches_border(box: np.ndarray, width: int, height: int, margin: float = 8.0) -> bool:
    return bool(
        box[0] <= margin
        or box[1] <= margin
        or box[2] >= width - 1 - margin
        or box[3] >= height - 1 - margin
    )


def _fully_outside(box: np.ndarray, width: int, height: int) -> bool:
    return bool(box[2] <= 0 or box[3] <= 0 or box[0] >= width or box[1] >= height)


def _area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def _center_inside(outer: np.ndarray, inner: tuple[float, float, float, float], pad: float = 40.0) -> bool:
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return bool(
        outer[0] - pad <= cx <= outer[2] + pad and outer[1] - pad <= cy <= outer[3] + pad
    )


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    ix1, iy1 = max(float(left[0]), float(right[0])), max(float(left[1]), float(right[1]))
    ix2, iy2 = min(float(left[2]), float(right[2])), min(float(left[3]), float(right[3]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _area(left) + _area(np.asarray(right)) - intersection
    if union <= 0:
        return 0.0
    return intersection / union
