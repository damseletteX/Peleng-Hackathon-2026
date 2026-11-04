from __future__ import annotations

import cv2
import numpy as np

from mv3d.calibration.camera_model import Camera
from mv3d.detection.detector import Detection
from mv3d.detection.keypoints import bbox_ground_point
from mv3d.geometry.triangulate import intersect_ray_plane

# Кузов в сантиметрах. Нос — начало координат актора, как в эталоне.
_LENGTH_CM = 2200.0
_WIDTH_CM = 320.0
_HEIGHT_CM = 550.0
_BOTTOM_Z_CM = 50.0


def ground_seed(camera: Camera, detection: Detection, plane_z: float) -> np.ndarray | None:
    """Луч через низ бокса пересекает землю, затем поднимается на высоту актора."""
    origin, direction = camera.ray_from_pixel(bbox_ground_point(detection))
    point = intersect_ray_plane(origin, direction, plane_z=0.0)
    if point is None:
        point = intersect_ray_plane(origin, direction, plane_z=plane_z)
    if point is None:
        return None
    out = point.copy()
    out[2] = plane_z
    return out


def fit_actor_origin(
    camera: Camera,
    detection: Detection,
    heading_xy: np.ndarray | None,
    plane_z: float,
    seed: np.ndarray | None = None,
) -> np.ndarray | None:
    """Параллелепипед, чья проекция совпадает с боксом. Точка — нос, не центр кузова."""
    start = seed if seed is not None else ground_seed(camera, detection, plane_z)
    if start is None:
        return None
    if heading_xy is None:
        return start
    heading = np.asarray(heading_xy, dtype=np.float64)[:2]
    norm = float(np.hypot(heading[0], heading[1]))
    if norm < 1e-6:
        return start
    heading = heading / norm
    target = np.asarray(detection.bbox, dtype=np.float64)
    best: tuple[float, float, float, float, float] | None = None
    for sign in (1.0, -1.0):
        direction = heading * sign
        for step, count in ((400.0, 7), (80.0, 7), (16.0, 7)):
            cx, cy = (start[0], start[1]) if best is None else (best[1], best[2])
            half = step * (count // 2)
            for dx in np.linspace(-half, half, count):
                for dy in np.linspace(-half, half, count):
                    origin = np.array([cx + dx, cy + dy, plane_z], dtype=np.float64)
                    aabb = _projected_aabb(camera, origin, direction)
                    if aabb is None:
                        continue
                    score = _iou(aabb, target)
                    if best is None or score > best[0]:
                        best = (score, float(origin[0]), float(origin[1]), float(direction[0]), float(direction[1]))
    if best is None:
        return start
    return np.array([best[1], best[2], plane_z], dtype=np.float64)


def heading_from_box(camera: Camera, detection: Detection) -> np.ndarray | None:
    """Направление кузова по нижней грани бокса на земле."""
    x1, _, x2, y2 = detection.bbox
    left = _ground_pixel(camera, np.array([(x1 + x2) / 2.0 - (x2 - x1) / 2.0, y2]))
    right = _ground_pixel(camera, np.array([(x1 + x2) / 2.0 + (x2 - x1) / 2.0, y2]))
    if left is None or right is None:
        return None
    delta = right[:2] - left[:2]
    span = float(np.hypot(delta[0], delta[1]))
    if span < 80.0:
        return None
    direction = delta / span
    if abs(span - _WIDTH_CM) < abs(span - _LENGTH_CM):
        direction = np.array([-direction[1], direction[0]], dtype=np.float64)
    return direction


def _ground_pixel(camera: Camera, uv: np.ndarray) -> np.ndarray | None:
    origin, direction = camera.ray_from_pixel(uv)
    return intersect_ray_plane(origin, direction, plane_z=0.0)


def draw_truck(
    canvas: np.ndarray,
    camera: Camera,
    origin: np.ndarray,
    heading_xy: np.ndarray,
    color: tuple[int, int, int] = (0, 210, 255),
) -> None:
    """Рёбра параллелепипеда в кадре. Передняя грань ярче."""
    heading = np.asarray(heading_xy, dtype=np.float64)[:2]
    norm = float(np.hypot(heading[0], heading[1]))
    if norm < 1e-6:
        return
    heading = heading / norm
    points = _corners(origin, heading)
    ahead = (camera.R @ (points - camera.center).T).T[:, 2]
    uv = camera.project(points)
    pixels = np.round(uv).astype(int)
    edges = (
        (0, 2), (2, 6), (6, 4), (4, 0),
        (1, 3), (3, 7), (7, 5), (5, 1),
        (0, 1), (2, 3), (4, 5), (6, 7),
    )
    nose = {(0, 2), (2, 3), (3, 1), (1, 0)}
    height, width = canvas.shape[:2]
    for i, j in edges:
        if ahead[i] <= 1.0 or ahead[j] <= 1.0:
            continue
        p1 = (int(np.clip(pixels[i, 0], 0, width - 1)), int(np.clip(pixels[i, 1], 0, height - 1)))
        p2 = (int(np.clip(pixels[j, 0], 0, width - 1)), int(np.clip(pixels[j, 1], 0, height - 1)))
        shade = (0, 255, 180) if (i, j) in nose or (j, i) in nose else color
        cv2.line(canvas, p1, p2, shade, 2)


def _corners(origin: np.ndarray, heading: np.ndarray) -> np.ndarray:
    forward = np.array([heading[0], heading[1], 0.0], dtype=np.float64)
    left = np.array([-heading[1], heading[0], 0.0], dtype=np.float64)
    points = []
    for along in (0.0, -_LENGTH_CM):
        for side in (-_WIDTH_CM / 2.0, _WIDTH_CM / 2.0):
            for height in (_BOTTOM_Z_CM, _BOTTOM_Z_CM + _HEIGHT_CM):
                point = origin.copy()
                point[0] += along * forward[0] + side * left[0]
                point[1] += along * forward[1] + side * left[1]
                point[2] = height
                points.append(point)
    return np.stack(points, axis=0)


def _projected_aabb(camera: Camera, origin: np.ndarray, heading: np.ndarray) -> np.ndarray | None:
    points = _corners(origin, heading)
    ahead = (camera.R @ (points - camera.center).T).T[:, 2]
    if np.any(ahead <= 1.0):
        return None
    uv = camera.project(points)
    return np.array(
        [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()],
        dtype=np.float64,
    )


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    ix1 = max(float(left[0]), float(right[0]))
    iy1 = max(float(left[1]), float(right[1]))
    ix2 = min(float(left[2]), float(right[2]))
    iy2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    if union <= 0:
        return 0.0
    return intersection / union
