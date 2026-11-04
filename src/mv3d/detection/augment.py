"""Геометрия кадра и боксов: поворот, отражение, кроп, цвет.

Те же преобразования пишутся в обучающую выборку и повторяются на лету
через TRUCK_TRAIN_AUG, пока RF-DETR Nano дообучается.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

Box = tuple[float, float, float, float]

# Онлайн-аугментации RF-DETR (Albumentations). Кроп здесь задаётся zoom
# через Affine и RandomSizedCrop: рамка пересчитывается вместе с кадром.
TRUCK_TRAIN_AUG: dict = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.25},
    "Rotate": {"limit": 18, "p": 0.7, "border_mode": cv2.BORDER_REFLECT_101},
    "Affine": {
        "scale": (0.8, 1.2),
        "translate_percent": (-0.08, 0.08),
        "rotate": (-8, 8),
        "shear": (-5, 5),
        "p": 0.55,
        "border_mode": cv2.BORDER_REFLECT_101,
    },
    "RandomSizedCrop": {
        "min_max_height": (480, 720),
        "height": 720,
        "width": 1280,
        "w2h_ratio": 1280 / 720,
        "p": 0.45,
    },
    "RandomBrightnessContrast": {
        "brightness_limit": 0.25,
        "contrast_limit": 0.25,
        "p": 0.6,
    },
    "HueSaturationValue": {
        "hue_shift_limit": 8,
        "sat_shift_limit": 20,
        "val_shift_limit": 20,
        "p": 0.4,
    },
    "GaussianBlur": {"blur_limit": 3, "p": 0.25},
    "GaussNoise": {"std_range": (0.01, 0.04), "p": 0.25},
    "ToGray": {"num_output_channels": 3, "p": 0.2},
    "InvertImg": {"p": 0.15},
    "ToSepia": {"p": 0.15},
    "ChannelShuffle": {"p": 0.15},
    "Solarize": {"threshold_range": (0.45, 0.6), "p": 0.15},
    "Equalize": {"p": 0.2},
    "CLAHE": {"clip_limit": 2.0, "p": 0.25},
    "RandomGamma": {"gamma_limit": (80, 130), "p": 0.3},
    "RGBShift": {
        "r_shift_limit": 15,
        "g_shift_limit": 15,
        "b_shift_limit": 15,
        "p": 0.3,
    },
}


def horizontal_flip(image: np.ndarray, boxes: list[Box]) -> tuple[np.ndarray, list[Box]]:
    width = image.shape[1]
    flipped = np.ascontiguousarray(image[:, ::-1])
    mirrored = [(width - x2, y1, width - x1, y2) for x1, y1, x2, y2 in boxes]
    return flipped, _clip(mirrored, image.shape[1], image.shape[0])


def vertical_flip(image: np.ndarray, boxes: list[Box]) -> tuple[np.ndarray, list[Box]]:
    height = image.shape[0]
    flipped = np.ascontiguousarray(image[::-1, :])
    mirrored = [(x1, height - y2, x2, height - y1) for x1, y1, x2, y2 in boxes]
    return flipped, _clip(mirrored, image.shape[1], image.shape[0])


def rotate(image: np.ndarray, boxes: list[Box], degrees: float) -> tuple[np.ndarray, list[Box]]:
    """Поворот вокруг центра. Холст расширяется, рамка строится по новым углам."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = round(height * sine + width * cosine)
    new_height = round(height * cosine + width * sine)
    matrix[0, 2] += (new_width - width) / 2.0
    matrix[1, 2] += (new_height - height) / 2.0
    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    moved: list[Box] = []
    for box in boxes:
        corners = _corners(box)
        ones = np.ones((4, 1), dtype=np.float64)
        transformed = np.hstack([corners, ones]) @ matrix.T
        xs, ys = transformed[:, 0], transformed[:, 1]
        moved.append((float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())))
    return rotated, _clip(moved, new_width, new_height)


def crop(
    image: np.ndarray,
    boxes: list[Box],
    origin: tuple[int, int],
    size: tuple[int, int],
    min_visibility: float = 0.35,
) -> tuple[np.ndarray, list[Box]]:
    """Прямоугольный кроп. Бокс остаётся, если внутри окна видна заметная часть."""
    x0, y0 = origin
    crop_width, crop_height = size
    height, width = image.shape[:2]
    x0 = int(np.clip(x0, 0, max(0, width - 1)))
    y0 = int(np.clip(y0, 0, max(0, height - 1)))
    x1 = int(np.clip(x0 + crop_width, x0 + 1, width))
    y1 = int(np.clip(y0 + crop_height, y0 + 1, height))
    cropped = image[y0:y1, x0:x1]
    kept: list[Box] = []
    for box in boxes:
        visible = _intersection(box, (float(x0), float(y0), float(x1), float(y1)))
        if visible is None:
            continue
        area = _area(box)
        if area <= 0 or _area(visible) / area < min_visibility:
            continue
        kept.append((visible[0] - x0, visible[1] - y0, visible[2] - x0, visible[3] - y0))
    return cropped, _clip(kept, cropped.shape[1], cropped.shape[0])


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Чёрно-белый кадр в трёх каналах. Рамка не меняется."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def invert_colors(image: np.ndarray) -> np.ndarray:
    """Негатив: каждый канал становится 255 минус значение."""
    return cv2.bitwise_not(image)


def to_sepia(image: np.ndarray) -> np.ndarray:
    """Тёплый сепия-тон. Матрица задана в BGR."""
    matrix = np.array(
        [
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189],
        ],
        dtype=np.float32,
    )
    toned = cv2.transform(image, matrix)
    return np.clip(toned, 0, 255).astype(np.uint8)


def shuffle_channels(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Случайный порядок каналов BGR, не исходный."""
    order = [0, 1, 2]
    while order == [0, 1, 2]:
        rng.shuffle(order)
    return np.ascontiguousarray(image[:, :, order])


def solarize(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Яркие пиксели инвертируются, тёмные остаются."""
    result = image.copy()
    mask = image >= threshold
    result[mask] = 255 - image[mask]
    return result


def photometric(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Яркость, контраст и лёгкое размытие. Рамки не двигаются."""
    alpha = rng.uniform(0.8, 1.25)
    beta = rng.uniform(-20, 20)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if rng.random() < 0.4:
        kernel = rng.choice((3, 5))
        adjusted = cv2.GaussianBlur(adjusted, (kernel, kernel), 0)
    return adjusted


def expand_views(
    image: np.ndarray,
    boxes: list[Box],
    rng: random.Random,
) -> list[tuple[str, np.ndarray, list[Box]]]:
    """Исходник плюс поворот, оба отражения, кроп и цветовой сдвиг."""
    views: list[tuple[str, np.ndarray, list[Box]]] = [("orig", image, list(boxes))]
    views.append(("hflip", *horizontal_flip(image, boxes)))
    views.append(("vflip", *vertical_flip(image, boxes)))

    angle = rng.choice((-18.0, -12.0, -8.0, 8.0, 12.0, 18.0))
    views.append((f"rot{int(angle):+d}", *rotate(image, boxes, angle)))

    flipped, flipped_boxes = horizontal_flip(image, boxes)
    views.append(("hflip_rot", *rotate(flipped, flipped_boxes, rng.choice((-10.0, 10.0)))))

    window = _crop_window(image.shape[1], image.shape[0], boxes, rng)
    if window is not None:
        origin, size = window
        cropped, cropped_boxes = crop(image, boxes, origin, size)
        if cropped_boxes:
            views.append(("crop", cropped, cropped_boxes))

    views.append(("color", photometric(image, rng), list(boxes)))
    views.append(("gray", to_grayscale(image), list(boxes)))
    views.append(("invert", invert_colors(image), list(boxes)))
    views.append(("sepia", to_sepia(image), list(boxes)))
    views.append(("channels", shuffle_channels(image, rng), list(boxes)))
    views.append(("solar", solarize(image), list(boxes)))
    return [(name, view, view_boxes) for name, view, view_boxes in views if view_boxes]


def _crop_window(
    width: int,
    height: int,
    boxes: list[Box],
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if width < 32 or height < 32 or not boxes:
        return None
    scale = rng.uniform(0.72, 0.92)
    crop_width = max(32, int(width * scale))
    crop_height = max(32, int(height * scale))
    focus = max(boxes, key=_area)
    center_x = (focus[0] + focus[2]) / 2.0
    center_y = (focus[1] + focus[3]) / 2.0
    jitter_x = rng.uniform(-0.08, 0.08) * width
    jitter_y = rng.uniform(-0.08, 0.08) * height
    x0 = int(np.clip(center_x + jitter_x - crop_width / 2.0, 0, width - crop_width))
    y0 = int(np.clip(center_y + jitter_y - crop_height / 2.0, 0, height - crop_height))
    return (x0, y0), (crop_width, crop_height)


def _corners(box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(left: Box, right: Box) -> Box | None:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _clip(boxes: list[Box], width: int, height: int, min_side: float = 4.0) -> list[Box]:
    kept: list[Box] = []
    for x1, y1, x2, y2 in boxes:
        left, right = sorted((float(x1), float(x2)))
        top, bottom = sorted((float(y1), float(y2)))
        left, top = max(0.0, left), max(0.0, top)
        right, bottom = min(float(width), right), min(float(height), bottom)
        if right - left >= min_side and bottom - top >= min_side:
            kept.append((left, top, right, bottom))
    return kept
