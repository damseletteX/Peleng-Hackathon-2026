import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from mv3d.detection.detector import VehicleDetector
from mv3d.detection.keypoints import bbox_center
from mv3d.detection.quality import observation_weight
from mv3d.tracking.tracks import (
    coasting_boxes,
    detection_from_supervision,
    detections_to_supervision,
    make_truck_tracker,
    track_id_at,
)


def process_video(detector, video, folder, limit):
    cap = cv2.VideoCapture(str(video))
    writer = None

    try:
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть {video}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("Не удалось определить размер видео")

        tracker = make_truck_tracker(fps)
        writer = cv2.VideoWriter(
            str(folder / "tracked.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Не удалось создать выходное видео")

        started = time.perf_counter()
        frame_id = 0
        with (folder / "observations.csv").open("w", newline="", encoding="utf-8") as handle:
            table = csv.writer(handle)
            table.writerow([
                "frame", "time_s", "camera_id", "local_track_id", "label", "score",
                "x1", "y1", "x2", "y2", "center_u", "center_v",
                "sharpness", "visibility", "weight", "held",
            ])

            while limit == 0 or frame_id < limit:
                ok, frame = cap.read()
                if not ok:
                    break

                tracked = tracker.update(detections_to_supervision(detector.detect(frame)))
                count = 0 if tracked.xyxy is None else len(tracked.xyxy)
                for index in range(count):
                    detection = detection_from_supervision(tracked, index)
                    track_id = track_id_at(tracked, index)
                    weight, sharpness, visibility = observation_weight(frame, detection)
                    _write_row(
                        table, frame_id, fps, video.stem, track_id, detection,
                        sharpness, visibility, weight, held=0,
                    )
                    _draw(frame, detection.bbox, detection.center, (
                        f"truck ID:{track_id} {detection.score:.2f} w={weight:.2f}"
                    ), (0, 210, 255))

                for track_id, box in coasting_boxes(tracker):
                    center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
                    _write_held_row(table, frame_id, fps, video.stem, track_id, box, center)
                    _draw(frame, box, center, f"truck ID:{track_id} hold", (180, 180, 180))

                cv2.putText(
                    frame,
                    f"{video.stem} | frame {frame_id} | truck",
                    (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                writer.write(frame)
                frame_id += 1
                if frame_id == 1 or frame_id % 10 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"{video.stem}: {frame_id} кадров; "
                        f"{frame_id / max(elapsed, 0.001):.2f} кадр/с",
                        flush=True,
                    )

            if frame_id == 0:
                raise RuntimeError("Не прочитан ни один кадр")
        print(f"Сохранено: {folder.resolve()}", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def _write_row(table, frame_id, fps, camera_id, track_id, detection, sharpness, visibility, weight, held):
    u, v = bbox_center(detection)
    x1, y1, x2, y2 = detection.bbox
    table.writerow([
        frame_id, round(frame_id / fps, 6), camera_id, track_id, detection.label,
        round(detection.score, 5), round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
        round(float(u), 2), round(float(v), 2),
        round(sharpness, 4), round(visibility, 4), round(weight, 5), held,
    ])


def _write_held_row(table, frame_id, fps, camera_id, track_id, box, center):
    x1, y1, x2, y2 = (float(v) for v in box)
    table.writerow([
        frame_id, round(frame_id / fps, 6), camera_id, track_id, "truck", "",
        round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
        round(center[0], 2), round(center[1], 2), "", "", "", 1,
    ])


def _draw(frame, box, center, label, color):
    x1, y1, x2, y2 = (round(float(v)) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (int(center[0]), int(center[1])), 4, color, -1)
    cv2.putText(
        frame, label, (max(0, x1), max(22, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="configs")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--score", type=float, default=0.15)
    parser.add_argument("--model", default="medium", choices=("small", "medium", "large"))
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("--frames должен быть >= 0")

    source = Path(args.videos)
    if source.is_file():
        videos = [source]
    elif source.is_dir():
        videos = sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
        )
    else:
        raise SystemExit(f"Путь не найден: {source}")
    if not videos:
        raise SystemExit("Видеофайлы не найдены")

    output = Path("outputs") / datetime.now().strftime("rfdetr_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    print("Найдены видео:", flush=True)
    for video in videos:
        print(f"  {video}", flush=True)
    print(f"\nЗагрузка RF-DETR {args.model}. Первый запуск может скачивать веса.", flush=True)
    detector = VehicleDetector(score_threshold=args.score, model_size=args.model)

    failed = []
    for index, video in enumerate(videos, 1):
        folder = output / f"{index:02d}_{video.stem}"
        folder.mkdir()
        print(f"\n[{index}/{len(videos)}] Обработка {video}", flush=True)
        try:
            process_video(detector, video, folder, args.frames)
        except (OSError, RuntimeError, ValueError) as error:
            failed.append((video, error))
            print(f"ОШИБКА: {video}: {error}", flush=True)

    print(f"\nРезультаты: {output.resolve()}", flush=True)
    if failed:
        for video, error in failed:
            print(f"Не обработано: {video}: {error}", flush=True)
        raise SystemExit(1)
    print("Все видео обработаны.", flush=True)


if __name__ == "__main__":
    main()
