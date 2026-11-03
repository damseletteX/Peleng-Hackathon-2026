from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from mv3d.calibration.camera_model import Camera, load_camera_track
from mv3d.detection.detector import Detection, VehicleDetector
from mv3d.detection.keypoints import reference_pixel
from mv3d.detection.quality import observation_weight
from mv3d.geometry.triangulate import intersect_ray_plane, triangulate_points
from mv3d.tracking.motion import TrajectoryFilter
from mv3d.tracking.tracks import (
    coasting_boxes,
    detection_from_supervision,
    detections_to_supervision,
    make_truck_tracker,
    track_id_at,
)

_GT_PATH = Path("configs/calibration/object_transform_SM_AA_track.csv")


@dataclass(frozen=True)
class CameraSource:
    camera_id: str
    video: Path
    calibration: Path


@dataclass
class View:
    camera_id: str
    camera: Camera
    detection: Detection
    track_id: int
    pixel: np.ndarray
    weight: float
    sharpness: float
    visibility: float


def run_tracking(
    config_path: str | Path = "configs/cameras.yml",
    frames: int = 0,
    score_threshold: float = 0.15,
    model_size: str = "medium",
    output_root: str | Path = "outputs",
) -> Path:
    """Один грузовик по всем камерам из cameras.yml, у которых есть видео."""
    plane_z, sources = _load_sources(Path(config_path))
    if not sources:
        raise SystemExit("В конфиге нет камер с существующим видео.")

    output = Path(output_root) / datetime.now().strftime("track_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    print("Камеры:", flush=True)
    for source in sources:
        print(f"  {source.camera_id}: {source.video}", flush=True)
    print(f"\nЗагрузка RF-DETR {model_size}. Первый запуск скачивает веса.", flush=True)

    detector = VehicleDetector(score_threshold=score_threshold, model_size=model_size)
    captures = [cv2.VideoCapture(str(source.video)) for source in sources]
    writers: list[cv2.VideoWriter | None] = [None] * len(sources)
    try:
        if any(not capture.isOpened() for capture in captures):
            raise RuntimeError("Не удалось открыть одно из видео.")
        fps = _fps(captures[0])
        dt = 1.0 / fps
        trackers = [make_truck_tracker(fps) for _ in sources]
        poses = [
            load_camera_track(source.calibration, source.camera_id) for source in sources
        ]
        motion = TrajectoryFilter()
        trajectory: list[tuple[int, np.ndarray, bool]] = []

        with (output / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
            table = csv.writer(handle)
            table.writerow([
                "frame", "time_s", "x_cm", "y_cm", "z_cm",
                "meas_x", "meas_y", "meas_z", "weight", "n_cameras", "camera_weights",
            ])
            frame_id = 0
            while frames == 0 or frame_id < frames:
                images = _read_frames(captures)
                if images is None:
                    break
                views, canvases = _observe(
                    images,
                    sources,
                    poses,
                    trackers,
                    detector,
                    frame_id,
                    motion,
                    dt,
                )
                measurement, weight = _fuse(views, plane_z)
                has_measurement = measurement is not None
                position = motion.step(
                    measurement if has_measurement else None,
                    weight if has_measurement else 0.0,
                    dt,
                )
                if position is not None:
                    trajectory.append((frame_id, position.copy(), measurement is not None))
                    table.writerow(
                        _trajectory_row(frame_id, fps, position, measurement, views)
                    )
                    _stamp(canvases, sources, frame_id, position, views)
                writers = _write_videos(writers, canvases, sources, output, fps)
                if position is not None and (frame_id == 0 or frame_id % 10 == 0):
                    print(
                        f"frame {frame_id}: X={position[0]:.1f} Y={position[1]:.1f} "
                        f"Z={position[2]:.1f} weight={weight:.3f} views={len(views)}",
                        flush=True,
                    )
                frame_id += 1
            if frame_id == 0:
                raise RuntimeError("Не прочитан ни один кадр.")
        _write_topdown(output / "trajectory_topdown.png", trajectory)
        _report_gt(trajectory)
    finally:
        for capture in captures:
            capture.release()
        for writer in writers:
            if writer is not None:
                writer.release()

    print(f"Сохранено: {output.resolve()}", flush=True)
    return output


def _load_sources(config_path: Path) -> tuple[float, list[CameraSource]]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    plane_z = float(data.get("object_plane_z_cm", 300.0))
    sources: list[CameraSource] = []
    for camera_id, cfg in data["cameras"].items():
        video = cfg.get("video")
        calibration = cfg.get("calibration")
        if not video or not calibration:
            continue
        video_path = Path(video)
        if video_path.is_file():
            sources.append(CameraSource(camera_id, video_path, Path(calibration)))
    return plane_z, sources


def _fps(capture: cv2.VideoCapture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        return 30.0
    return fps


def _read_frames(captures: list[cv2.VideoCapture]) -> list[np.ndarray] | None:
    frames: list[np.ndarray] = []
    for capture in captures:
        ok, frame = capture.read()
        if not ok:
            return None
        frames.append(frame)
    return frames


def _observe(images, sources, poses, trackers, detector, frame_id, motion, dt):
    views: list[View] = []
    canvases: list[np.ndarray] = []
    predicted = motion.peek(dt)
    for image, source, pose_track, tracker in zip(images, sources, poses, trackers, strict=True):
        camera = pose_track[min(frame_id, len(pose_track) - 1)]
        found = detector.detect(image)
        tracked = tracker.update(detections_to_supervision(found))
        canvas = image.copy()
        chosen = _choose(
            _candidates(tracked, image, camera, _image_motion(camera, motion)),
            camera,
            predicted,
        )
        if chosen is not None:
            views.append(chosen)
            _draw_detection(canvas, chosen)
        for track_id, box in coasting_boxes(tracker):
            _draw_box(canvas, box, f"truck ID:{track_id} hold", (180, 180, 180), held=True)
        canvases.append(canvas)
    return views, canvases


def _image_motion(camera: Camera, motion: TrajectoryFilter) -> np.ndarray | None:
    if motion.x is None:
        return None
    speed = float(np.hypot(motion.x[3], motion.x[4]))
    if speed < 30.0:
        return None
    now = motion.x[:3]
    ahead = now + motion.x[3:6]
    return camera.project(ahead) - camera.project(now)


def _candidates(tracked, image, camera, motion_uv) -> list[View]:
    candidates: list[View] = []
    count = 0 if tracked.xyxy is None else len(tracked.xyxy)
    for index in range(count):
        detection = detection_from_supervision(tracked, index)
        if detection.label != "truck":
            continue
        weight, sharpness, visibility = observation_weight(image, detection)
        candidates.append(
            View(
                camera_id=camera.camera_id,
                camera=camera,
                detection=detection,
                track_id=track_id_at(tracked, index),
                pixel=reference_pixel(detection, motion_uv),
                weight=weight,
                sharpness=sharpness,
                visibility=visibility,
            )
        )
    return candidates


def _choose(candidates: list[View], camera: Camera, predicted: np.ndarray | None) -> View | None:
    confirmed = [item for item in candidates if item.track_id >= 0]
    pool = confirmed or candidates
    if not pool:
        return None
    if predicted is None:
        return max(pool, key=lambda item: item.weight)

    projected = camera.project(predicted)

    def preference(item: View) -> float:
        distance = float(np.linalg.norm(item.pixel - projected))
        return item.weight / (1.0 + distance / 80.0)

    return max(pool, key=preference)


def _fuse(views: list[View], plane_z: float) -> tuple[np.ndarray | None, float]:
    if not views:
        return None, 0.0
    positive = [view for view in views if view.weight > 0]
    weights = [view.weight for view in positive]
    if len(positive) >= 2:
        point = triangulate_points(
            [view.camera for view in positive],
            [view.pixel for view in positive],
            weights,
        )
    elif len(positive) == 1:
        view = max(views, key=lambda item: item.weight)
        origin, direction = view.camera.ray_from_pixel(view.pixel)
        point = intersect_ray_plane(origin, direction, plane_z=plane_z)
    else:
        point = None
    return point, float(sum(weights))


def _trajectory_row(frame_id, fps, position, measurement, views) -> list[object]:
    if measurement is None:
        meas = ["", "", ""]
    else:
        meas = [round(float(v), 2) for v in measurement]
    weights = ";".join(
        f"{view.camera_id}:{view.weight:.4f}" for view in views
    )
    return [
        frame_id,
        round(frame_id / fps, 6),
        round(float(position[0]), 2),
        round(float(position[1]), 2),
        round(float(position[2]), 2),
        *meas,
        round(float(sum(view.weight for view in views)), 5),
        len(views),
        weights,
    ]


def _stamp(canvases, sources, frame_id, position, views):
    by_camera = {view.camera_id: view for view in views}
    text = f"frame {frame_id}  X={position[0]:.0f} Y={position[1]:.0f} Z={position[2]:.0f}"
    for canvas, source in zip(canvases, sources, strict=True):
        view = by_camera.get(source.camera_id)
        suffix = "" if view is None else f"  w={view.weight:.2f}"
        cv2.putText(
            canvas, text + suffix, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )


def _write_videos(writers, canvases, sources, output, fps):
    ready = list(writers)
    for index, (canvas, source) in enumerate(zip(canvases, sources, strict=True)):
        if ready[index] is None:
            height, width = canvas.shape[:2]
            path = output / f"{source.camera_id}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Не удалось создать {path}")
            ready[index] = writer
        ready[index].write(canvas)
    return ready


def _draw_detection(canvas: np.ndarray, view: View) -> None:
    label = (
        f"truck ID:{view.track_id} {view.detection.score:.2f} "
        f"w={view.weight:.2f}"
    )
    _draw_box(canvas, np.array(view.detection.bbox), label, (0, 210, 255), held=False)
    center = tuple(round(float(v)) for v in view.pixel)
    cv2.circle(canvas, center, 4, (0, 210, 255), -1)


def _draw_box(canvas, box, label, color, held: bool) -> None:
    x1, y1, x2, y2 = (round(float(v)) for v in box)
    height, width = canvas.shape[:2]
    x1 = int(np.clip(x1, 0, width - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    y2 = int(np.clip(y2, 0, height - 1))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1 if held else 2)
    cv2.putText(
        canvas, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
    )


def _write_topdown(path: Path, trajectory: list[tuple[int, np.ndarray]]) -> None:
    if len(trajectory) < 2:
        return
    points = np.array([xyz[:2] for _, xyz, _ in trajectory], dtype=np.float64)
    canvas = np.full((720, 960, 3), 255, dtype=np.uint8)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    pixels = (points - minimum) / span
    pixels[:, 0] = 40 + pixels[:, 0] * 880
    pixels[:, 1] = 680 - pixels[:, 1] * 640
    poly = np.round(pixels).astype(np.int32)
    cv2.polylines(canvas, [poly], False, (0, 120, 220), 2)
    cv2.circle(canvas, tuple(poly[0]), 5, (0, 160, 0), -1)
    cv2.circle(canvas, tuple(poly[-1]), 5, (0, 0, 200), -1)
    cv2.imwrite(str(path), canvas)


def _report_gt(trajectory: list[tuple[int, np.ndarray]]) -> None:
    if not _GT_PATH.is_file() or not trajectory:
        return
    gt: dict[int, np.ndarray] = {}
    with _GT_PATH.open(encoding="utf-8", errors="replace") as handle:
        next(handle, None)
        for line in handle:
            parts = [item.strip() for item in line.split(";")]
            if len(parts) < 4 or not parts[0]:
                continue
            try:
                gt[int(float(parts[0]))] = np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64
                )
            except ValueError:
                continue
    squared_measured: list[float] = []
    squared_all: list[float] = []
    for frame, xyz, has_measurement in trajectory:
        if frame not in gt:
            continue
        squared = float(np.sum((xyz - gt[frame]) ** 2))
        squared_all.append(squared)
        if has_measurement:
            squared_measured.append(squared)
    if squared_all:
        mse = float(np.mean(squared_all))
        print(
            f"MSE по кадрам с целым эталоном: {mse:.0f} см², "
            f"RMSE {np.sqrt(mse):.1f} см, кадров {len(squared_all)}",
            flush=True,
        )
    if squared_measured:
        mse_measured = float(np.mean(squared_measured))
        print(
            f"MSE на кадрах с детекцией: {mse_measured:.0f} см², "
            f"RMSE {np.sqrt(mse_measured):.1f} см, кадров {len(squared_measured)}",
            flush=True,
        )
