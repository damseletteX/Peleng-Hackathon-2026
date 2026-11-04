from __future__ import annotations

import csv
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from mv3d.calibration.camera_model import Camera, load_camera_track
from mv3d.detection.detector import Detection, VehicleDetector  #, select_truck_boxes
from mv3d.detection.keypoints import reference_pixel
from mv3d.detection.quality import observation_weight
from mv3d.geometry.triangulate import intersect_ray_plane, triangulate_points
from mv3d.tracking.boxes import BoxKeeper
from mv3d.tracking.motion import TrajectoryFilter, place_actor_origin
from mv3d.tracking.tracks import (
    detection_from_supervision,
    detections_to_supervision,
    make_truck_tracker,
    track_id_at,
)

_GT_PATH = Path("configs/calibration/object_transform_SM_AA_track.csv")
# Монокулярные точки одной и той же машины на плоскости должны лежать рядом.
_MONO_GATE_CM = 8000.0
# Триангуляция дальше этого от луча согласованной камеры не используется.
_TRI_AGREE_CM = 5000.0
# Прогноз должен попасть рядом с точкой на боксе, иначе это другая камера.
_REPROJ_PX = 160.0


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
    output_root: str | Path = "outputs",
    write_csv: bool = True,
    write_video: bool = True,
    camera: str | None = None,
    write_plots: bool | None = None,
    should_stop=None,
) -> Path:
    """Один грузовик по камерам из cameras.yml. camera — только этот id."""
    if write_plots is None:
        write_plots = write_csv
    if not write_csv and not write_video and not write_plots:
        raise SystemExit("Нужен выход: CSV, графики, видео или их сочетание.")
    plane_z, sources = _load_sources(Path(config_path), camera=camera)
    if not sources:
        raise SystemExit("В конфиге нет камер с существующим видео.")

    output = Path(output_root) / datetime.now().strftime("track_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    print("Камеры:", flush=True)
    for source in sources:
        print(f"  {source.camera_id}: {source.video}", flush=True)
    pieces = []
    if write_csv:
        pieces.append("CSV")
    if write_plots:
        pieces.append("графики")
    if write_video:
        pieces.append("видео")
    print(f"Выход: {', '.join(pieces)}. Модель: RF-DETR nano.", flush=True)
    print("\nЗагрузка RF-DETR nano.", flush=True)

    detector = VehicleDetector(score_threshold=score_threshold)
    captures = [cv2.VideoCapture(str(source.video)) for source in sources]
    writers: list[cv2.VideoWriter | None] = [None] * len(sources)
    try:
        if any(not capture.isOpened() for capture in captures):
            raise RuntimeError("Не удалось открыть одно из видео.")
        fps = _fps(captures[0])
        dt = 1.0 / fps
        trackers = [make_truck_tracker(fps) for _ in sources]
        keepers = [BoxKeeper() for _ in sources]
        poses = [
            load_camera_track(source.calibration, source.camera_id) for source in sources
        ]
        motion = TrajectoryFilter()
        locked_id: str | None = None
        trajectory: list[tuple[int, np.ndarray, bool]] = []

        with ExitStack() as stack:
            if write_csv:
                common, per_camera, errors = _open_track_tables(output, sources, stack)
            gt = _load_gt()
            frame_id = 0
            while frames == 0 or frame_id < frames:
                if should_stop is not None and should_stop():
                    print("Остановлено.", flush=True)
                    break
                images = _read_frames(captures)
                if images is None:
                    break
                views, canvases = _observe(
                    images,
                    sources,
                    poses,
                    trackers,
                    keepers,
                    detector,
                    frame_id,
                    motion,
                    dt,
                    draw=write_video,
                )
                velocity = None if motion.x is None else motion.x[3:6]
                measurement, weight, locked_id = _fuse(
                    views, plane_z, motion.peek(dt), locked_id, velocity
                )
                has_measurement = measurement is not None
                position = motion.step(
                    measurement if has_measurement else None,
                    weight if has_measurement else 0.0,
                    dt,
                )
                if position is not None:
                    trajectory.append((frame_id, position.copy(), measurement is not None))
                    if write_video:
                        _stamp(canvases, sources, frame_id, position, views)
                if write_csv:
                    _write_track_rows(
                        common, per_camera, sources, views, frame_id, position, plane_z
                    )
                    truth = gt.get(frame_id)
                    errors.writerow([frame_id, *_error_cells(position, truth)])
                if write_video:
                    writers = _write_videos(writers, canvases, sources, output, fps)
                if position is not None and (frame_id == 0 or frame_id % 10 == 0):
                    print(
                        f"frame {frame_id}: X={position[0]:.1f} Y={position[1]:.1f} "
                        f"Z={position[2]:.1f} weight={weight:.3f} views={len(views)}",
                        flush=True,
                    )
                frame_id += 1
            if frame_id == 0 and not (should_stop is not None and should_stop()):
                raise RuntimeError("Не прочитан ни один кадр.")
        if write_plots:
            _write_topdown(output / "trajectory_topdown.png", trajectory)
            plot = output / "coordinates_time.png"
            write_coordinate_plot(plot, [(frame, xyz) for frame, xyz, _ in trajectory], gt)
            print(f"Графики: {plot.resolve()}", flush=True)
        if write_csv:
            print(f"Ошибки по кадрам: {(output / 'errors.csv').resolve()}", flush=True)
        _report_gt(trajectory)
    finally:
        for capture in captures:
            capture.release()
        for writer in writers:
            if writer is not None:
                writer.release()

    print(f"Сохранено: {output.resolve()}", flush=True)
    return output


def _load_sources(
    config_path: Path,
    camera: str | None = None,
) -> tuple[float, list[CameraSource]]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    plane_z = float(data.get("object_plane_z_cm", 300.0))
    cameras = data["cameras"]
    if camera is not None:
        if camera not in cameras:
            known = ", ".join(cameras)
            raise SystemExit(f"Камеры {camera} нет в конфиге. Доступны: {known}")
        cameras = {camera: cameras[camera]}
    sources: list[CameraSource] = []
    missing: list[str] = []
    for camera_id, cfg in cameras.items():
        video = cfg.get("video")
        calibration = cfg.get("calibration")
        video_path = Path(video) if video else None
        if video_path is None or not video_path.is_file() or not calibration:
            missing.append(camera_id)
            continue
        sources.append(CameraSource(camera_id, video_path, Path(calibration)))
    if missing:
        names = ", ".join(missing)
        raise SystemExit(f"Нет видео или калибровки для камер: {names}")
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


def _observe(
    images, sources, poses, trackers, keepers, detector, frame_id, motion, dt, draw: bool
):
    views: list[View] = []
    canvases: list[np.ndarray] = []
    predicted = motion.peek(dt)
    found_per_camera = detector.detect_batch(images, score_threshold=0.05)
    for image, source, pose_track, tracker, keeper, found in zip(
        images, sources, poses, trackers, keepers, found_per_camera, strict=True
    ):
        camera = pose_track[min(frame_id, len(pose_track) - 1)]
        height, width = image.shape[:2]
        # if keeper.wants_faint():
        #     faint = detector.detect(image, score_threshold=0.05)
        #     found = select_truck_boxes(found + faint, float(width * height))
        stable = keeper.update(found, width, height)
        live = [] if keeper.last_live is None else [keeper.last_live]
        tracked = tracker.update(detections_to_supervision(live))
        chosen = _choose(
            _candidates(tracked, image, camera, _image_motion(camera, motion)),
            camera,
            predicted,
        )
        if chosen is not None:
            views.append(chosen)
        if not draw:
            continue
        canvas = image.copy()
        if stable is not None:
            _draw_stable(canvas, stable, keeper.last_live)
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


def _fuse(
    views: list[View],
    plane_z: float,
    predicted: np.ndarray | None,
    locked_id: str | None = None,
    velocity: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float, str | None]:
    """Сложить только камеры, в чьём боксе лежит уже найденный грузовик.

    База — луч одной такой камеры. Триангуляция заменяет его, только если
    сошлась с этим лучом: чужой вид не уводит точку на сотни метров.
    """
    chosen, locked_id = _views_for_fusion(views, plane_z, predicted, locked_id)
    if not chosen:
        return None, 0.0, locked_id
    weights = [view.weight for view in chosen]
    point = _closest_monocular(chosen, plane_z, predicted)
    if point is None:
        return None, 0.0, locked_id
    if len(chosen) >= 2:
        triangulated = triangulate_points(
            [view.camera for view in chosen],
            [view.pixel for view in chosen],
            weights,
        )
        if np.all(np.isfinite(triangulated)) and _agree(triangulated, point):
            point = triangulated
        else:
            point = place_actor_origin(point, velocity)
    else:
        point = place_actor_origin(point, velocity)
    return point, float(sum(weights)), locked_id


def _views_for_fusion(
    views: list[View],
    plane_z: float,
    predicted: np.ndarray | None,
    locked_id: str | None = None,
) -> tuple[list[View], str | None]:
    positive = [view for view in views if view.weight > 0]
    if not positive:
        return [], locked_id
    if predicted is None:
        chosen = _cluster_monocular(positive, plane_z)
        if len(chosen) == 1:
            locked_id = chosen[0].camera_id
        return chosen, locked_id
    ranked = sorted(
        ((_reprojection_px(view, predicted), view) for view in positive),
        key=lambda item: item[0],
    )
    current = [view for error, view in ranked if view.camera_id == locked_id and error <= _REPROJ_PX]
    if current:
        primary = current[0]
    elif ranked[0][0] <= 50.0:
        primary = ranked[0][1]
        locked_id = primary.camera_id
    else:
        return [], locked_id
    extras = [
        view
        for error, view in ranked
        if view.camera_id != primary.camera_id and error <= 40.0
    ]
    return [primary, *extras], locked_id


def _reprojection_px(view: View, point: np.ndarray) -> float:
    ahead = view.camera.R @ (np.asarray(point, dtype=np.float64) - view.camera.center)
    if float(ahead[2]) <= 0:
        return float("inf")
    u, v = (float(value) for value in view.camera.project(point))
    return float(np.hypot(u - view.pixel[0], v - view.pixel[1]))


def _agree(left: np.ndarray, right: np.ndarray) -> bool:
    return float(np.hypot(left[0] - right[0], left[1] - right[1])) <= _TRI_AGREE_CM


def _closest_monocular(
    views: list[View],
    plane_z: float,
    predicted: np.ndarray | None,
) -> np.ndarray | None:
    best_point = None
    best_key = None
    for view in views:
        point = _monocular_point(view, plane_z)
        if point is None:
            continue
        if predicted is None:
            key = -view.weight
        else:
            key = float(np.hypot(point[0] - predicted[0], point[1] - predicted[1]))
        if best_key is None or key < best_key:
            best_key = key
            best_point = point
    return best_point


def _cluster_monocular(views: list[View], plane_z: float) -> list[View]:
    located: list[tuple[View, np.ndarray]] = []
    for view in views:
        point = _monocular_point(view, plane_z)
        if point is not None:
            located.append((view, point))
    if not located:
        return []

    def group(index: int) -> list[tuple[View, np.ndarray]]:
        origin = located[index][1]
        return [
            item
            for item in located
            if np.hypot(item[1][0] - origin[0], item[1][1] - origin[1]) <= _MONO_GATE_CM
        ]

    best = max(range(len(located)), key=lambda i: (len(group(i)), located[i][0].weight))
    return [view for view, _ in group(best)]


def _monocular_point(view: View, plane_z: float) -> np.ndarray | None:
    origin, direction = view.camera.ray_from_pixel(view.pixel)
    return intersect_ray_plane(origin, direction, plane_z=plane_z)


def _open_track_tables(output: Path, sources: list[CameraSource], stack: ExitStack):
    """Формат как у object_transform_SM_AA_track.csv: Frame;Loc_X;Loc_Y;Loc_Z."""
    header = ["Frame", "Loc_X", "Loc_Y", "Loc_Z"]
    common_handle = stack.enter_context(
        (output / "track.csv").open("w", newline="", encoding="utf-8")
    )
    common = csv.writer(common_handle, delimiter=";")
    common.writerow(header)
    per_camera: dict[str, csv.writer] = {}
    for source in sources:
        handle = stack.enter_context(
            (output / f"{source.camera_id}.csv").open("w", newline="", encoding="utf-8")
        )
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(header)
        per_camera[source.camera_id] = writer
    error_handle = stack.enter_context(
        (output / "errors.csv").open("w", newline="", encoding="utf-8")
    )
    errors = csv.writer(error_handle, delimiter=";")
    errors.writerow(["Frame", "Err_X", "Err_Y", "Err_Z", "Dist_cm"])
    return common, per_camera, errors


def _write_track_rows(common, per_camera, sources, views, frame_id, position, plane_z) -> None:
    by_camera = {view.camera_id: view for view in views}
    for source in sources:
        view = by_camera.get(source.camera_id)
        point = None
        if view is not None:
            origin, direction = view.camera.ray_from_pixel(view.pixel)
            point = intersect_ray_plane(origin, direction, plane_z=plane_z)
        per_camera[source.camera_id].writerow([frame_id, *_loc_cells(point)])
    common.writerow([frame_id, *_loc_cells(position)])


def _error_cells(position: np.ndarray | None, truth: np.ndarray | None) -> list[str]:
    if position is None or truth is None:
        return ["", "", "", ""]
    delta = np.asarray(position, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    return [f"{float(value):.4f}" for value in delta] + [f"{distance:.4f}"]


def score_coordinates(
    track: str | Path,
    gt_path: str | Path | None = None,
    out: str | Path | None = None,
) -> Path:
    """Покадровые ошибки по уже записанному track.csv, без детектора."""
    track_path = Path(track)
    if track_path.is_dir():
        track_path = track_path / "track.csv"
    if not track_path.is_file():
        raise SystemExit(f"Нет файла координат: {track_path}")

    reference = _GT_PATH if gt_path is None else Path(gt_path)
    gt = _load_gt(reference)
    destination = track_path.with_name("errors.csv") if out is None else Path(out)
    rows = _read_track(track_path)
    squared: list[float] = []
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["Frame", "Err_X", "Err_Y", "Err_Z", "Dist_cm"])
        for frame_id, position in rows:
            truth = gt.get(frame_id)
            writer.writerow([frame_id, *_error_cells(position, truth)])
            if position is not None and truth is not None:
                squared.append(float(np.sum((position - truth) ** 2)))
    plot = destination.with_name("coordinates_time.png")
    write_coordinate_plot(plot, rows, gt)
    print(f"Графики: {plot.resolve()}", flush=True)
    if squared:
        mse = float(np.mean(squared))
        print(
            f"MSE: {mse:.0f} см², RMSE {np.sqrt(mse):.1f} см, кадров {len(squared)}",
            flush=True,
        )
    else:
        print("Нет кадров, где есть и координата, и целый эталон.", flush=True)
    print(f"Ошибки по кадрам: {destination.resolve()}", flush=True)
    return destination


def write_coordinate_plot(
    path: Path,
    rows: list[tuple[int, np.ndarray | None]],
    gt: dict[int, np.ndarray],
) -> None:
    """X, Y и Z от номера кадра. Оценка синяя, эталон зелёный."""
    usable = [(frame, point) for frame, point in rows if point is not None]
    if len(usable) < 2:
        return
    width, panel, gap = 960, 220, 16
    canvas = np.full((panel * 3 + gap * 4, width, 3), 255, dtype=np.uint8)
    frames = [frame for frame, _ in usable]
    frame_min, frame_max = min(frames), max(frames)
    for axis, name in enumerate(("X", "Y", "Z")):
        top = gap + axis * (panel + gap)
        estimate = [(frame, float(point[axis])) for frame, point in usable]
        truth = [(frame, float(gt[frame][axis])) for frame, _ in usable if frame in gt]
        _draw_series(canvas, estimate, truth, frame_min, frame_max, top, panel - 8, width)
        cv2.putText(
            canvas, name, (8, top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2
        )
    cv2.imwrite(str(path), canvas)


def _draw_series(canvas, estimate, truth, frame_min, frame_max, top, height, width) -> None:
    values = [value for _, value in estimate] + [value for _, value in truth]
    if not values:
        return
    low, high = min(values), max(values)
    if high - low < 1.0:
        high = low + 1.0
    span = max(frame_max - frame_min, 1)

    def pixel(frame: int, value: float) -> tuple[int, int]:
        x = 50 + (frame - frame_min) / span * (width - 80)
        y = top + height - 16 - (value - low) / (high - low) * (height - 32)
        return round(x), round(y)

    def line(points: list[tuple[int, float]], color: tuple[int, int, int]) -> None:
        if len(points) < 2:
            return
        poly = np.array([pixel(frame, value) for frame, value in points], dtype=np.int32)
        cv2.polylines(canvas, [poly], False, color, 2)

    line(truth, (80, 160, 80))
    line(estimate, (200, 80, 0))


def _read_track(path: Path) -> list[tuple[int, np.ndarray | None]]:
    rows: list[tuple[int, np.ndarray | None]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        next(handle, None)
        for line in handle:
            parts = [item.strip() for item in line.split(";")]
            if not parts or not parts[0]:
                continue
            frame_id = int(float(parts[0]))
            position = None
            if len(parts) >= 4 and all(parts[1:4]):
                try:
                    position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    position = None
            rows.append((frame_id, position))
    return rows


def _load_gt(path: Path | None = None) -> dict[int, np.ndarray]:
    gt: dict[int, np.ndarray] = {}
    source = _GT_PATH if path is None else path
    if not source.is_file():
        return gt
    with source.open(encoding="utf-8", errors="replace") as handle:
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
    return gt


def _loc_cells(point: np.ndarray | None) -> list[str]:
    if point is None:
        return ["", "", ""]
    return [f"{float(value):.4f}" for value in point]


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


def _draw_stable(canvas: np.ndarray, box: np.ndarray, live: Detection | None) -> None:
    if live is None:
        label = "truck"
    else:
        label = f"truck {live.score:.2f}"
    _draw_box(canvas, box, label, (0, 210, 255), held=live is None)


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
