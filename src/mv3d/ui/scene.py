from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_CAMERA_COLORS = {
    "cam1": (90, 220, 120),
    "cam2": (230, 180, 70),
    "cam3": (80, 150, 255),
    "cam4": (220, 110, 210),
    "cam5": (70, 210, 230),
}
_FUSED = (0, 255, 204)
_BG = (18, 18, 18)


def load_tracks(folder: Path) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]]]:
    """Общий track.csv и отдельные camN.csv. Пустые кадры пропускаются."""
    fused = _read_points(folder / "track.csv")
    cameras: dict[str, list[np.ndarray]] = {}
    for path in sorted(folder.glob("cam*.csv")):
        points = _read_points(path)
        if points:
            cameras[path.stem] = points
    return fused, cameras


def render_space(
    fused: list[np.ndarray],
    cameras: dict[str, list[np.ndarray]],
    *,
    width: int = 1100,
    height: int = 720,
    cursor: int | None = None,
) -> np.ndarray:
    """Изометрия: пол, следы камер и общий путь грузовика."""
    canvas = np.full((height, width, 3), _BG, dtype=np.uint8)
    usable = [point for point in fused if point is not None]
    if len(usable) < 2:
        cv2.putText(
            canvas,
            "too few points",
            (40, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (160, 160, 160),
            2,
            cv2.LINE_AA,
        )
        return canvas

    cloud = np.stack(usable, axis=0)
    center = np.median(cloud[:, :2], axis=0)
    span_xy = float(max(np.ptp(cloud[:, 0]), np.ptp(cloud[:, 1]), 500.0))
    gate = max(span_xy * 1.4, 2500.0)
    shown: dict[str, np.ndarray] = {}
    for name, points in cameras.items():
        stack = np.stack(points, axis=0)
        distance = np.hypot(stack[:, 0] - center[0], stack[:, 1] - center[1])
        keep = stack[distance <= gate]
        if len(keep) >= 2:
            shown[name] = keep

    floor = float(np.min(cloud[:, 2]))
    lo = cloud.min(axis=0)
    hi = cloud.max(axis=0)
    for stack in shown.values():
        lo = np.minimum(lo, stack.min(axis=0))
        hi = np.maximum(hi, stack.max(axis=0))
    lo[2] = min(lo[2], floor)
    pad = np.maximum((hi - lo) * 0.08, np.array([200.0, 200.0, 40.0]))
    lo = lo - pad
    hi = hi + pad
    scale = float(np.max(hi - lo))

    def to_screen(points: np.ndarray) -> np.ndarray:
        unit = (points - lo) / scale
        x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]
        u = (x - y) * np.cos(np.deg2rad(30))
        v = -((x + y) * np.sin(np.deg2rad(30)) - z)
        return np.stack([u, v], axis=1)

    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
        dtype=np.float64,
    )
    projected = to_screen(corners)
    umin, vmin = projected.min(axis=0)
    umax, vmax = projected.max(axis=0)

    def pixel(points: np.ndarray) -> np.ndarray:
        uv = to_screen(np.atleast_2d(points))
        x = 70 + (uv[:, 0] - umin) / max(umax - umin, 1e-6) * (width - 150)
        y = 40 + (vmax - uv[:, 1]) / max(vmax - vmin, 1e-6) * (height - 90)
        return np.stack([x, y], axis=1)

    _draw_floor(canvas, pixel, lo, hi, floor)
    for name, stack in shown.items():
        color = _CAMERA_COLORS.get(name, (140, 140, 140))
        _polyline(canvas, pixel(stack), color, 1)
    shadow = cloud.copy()
    shadow[:, 2] = floor
    _polyline(canvas, pixel(shadow), (40, 70, 40), 2)
    _polyline(canvas, pixel(cloud), (20, 90, 70), 7)
    _polyline(canvas, pixel(cloud), _FUSED, 2)

    index = len(cloud) - 1 if cursor is None else int(np.clip(cursor, 0, len(cloud) - 1))
    _draw_truck(canvas, pixel, cloud, index)
    _legend(canvas, ["track", *shown.keys()], width)
    return canvas


def _read_points(path: Path) -> list[np.ndarray]:
    if not path.is_file():
        return []
    points: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        parts = [item.strip() for item in line.split(";")]
        if len(parts) < 4 or not all(parts[1:4]):
            continue
        try:
            points.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
        except ValueError:
            continue
    return points


def _draw_floor(canvas, pixel, lo, hi, floor) -> None:
    steps = 8
    color = (42, 42, 42)
    xs = np.linspace(lo[0], hi[0], steps)
    ys = np.linspace(lo[1], hi[1], steps)
    for x in xs:
        line = np.array([[x, lo[1], floor], [x, hi[1], floor]])
        _polyline(canvas, pixel(line), color, 1)
    for y in ys:
        line = np.array([[lo[0], y, floor], [hi[0], y, floor]])
        _polyline(canvas, pixel(line), color, 1)
    cv2.putText(canvas, "X", tuple(pixel(np.array([[hi[0], lo[1], floor]]))[0].astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 160, 120), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Y", tuple(pixel(np.array([[lo[0], hi[1], floor]]))[0].astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 160, 120), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Z", tuple(pixel(np.array([[lo[0], lo[1], hi[2]]]))[0].astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 160, 120), 1, cv2.LINE_AA)


def _draw_truck(canvas, pixel, cloud: np.ndarray, index: int) -> None:
    point = cloud[index]
    before = cloud[max(0, index - 1)]
    heading = point[:2] - before[:2]
    norm = float(np.hypot(heading[0], heading[1]))
    if norm < 1.0:
        forward = np.array([1.0, 0.0])
    else:
        forward = heading / norm
    side = np.array([-forward[1], forward[0]])
    span = float(max(np.ptp(cloud[:, 0]), np.ptp(cloud[:, 1]), 1.0))
    length = float(np.clip(span * 0.018, 250.0, span * 0.06))
    width = length * 0.42
    height = length * 0.55
    base = point.copy()
    corners = []
    for dz in (0.0, height):
        for sx, fx in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            xy = base[:2] + side * (sx * width) + forward * (fx * length)
            corners.append([xy[0], xy[1], base[2] + dz])
    screen = pixel(np.array(corners))
    faces = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in faces:
        p1 = tuple(screen[a].astype(int))
        p2 = tuple(screen[b].astype(int))
        cv2.line(canvas, p1, p2, (0, 255, 204), 2, cv2.LINE_AA)


def _polyline(canvas, screen: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    if len(screen) < 2:
        return
    poly = np.round(screen).astype(np.int32)
    cv2.polylines(canvas, [poly], False, color, thickness, cv2.LINE_AA)


def _legend(canvas, names: list[str], width: int) -> None:
    x = width - 150
    y = 36
    cv2.putText(
        canvas, "track", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA
    )
    for name in names:
        y += 26
        color = _FUSED if name == "track" else _CAMERA_COLORS.get(name, (140, 140, 140))
        label = name
        cv2.circle(canvas, (x + 8, y - 5), 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas, label, (x + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA
        )
