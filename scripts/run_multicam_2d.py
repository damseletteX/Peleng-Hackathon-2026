import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETRSmall
from trackers import ByteTrackTracker


VEHICLES = {"car", "truck", "bus", "motorcycle"}


def get_names(detections):
    names = detections.data.get("class_name")
    if names is not None:
        return np.asarray(names, dtype=str)

    from rfdetr.assets.coco_classes import COCO_CLASSES

    return np.asarray(
        [COCO_CLASSES[int(i)] for i in detections.class_id],
        dtype=str,
    )


def process_video(model, video, folder, limit):
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

        tracker = ByteTrackTracker(frame_rate=fps)

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

        with (folder / "observations.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            table = csv.writer(handle)
            table.writerow([
                "frame", "time_s", "camera_id", "local_track_id",
                "label", "score", "x1", "y1", "x2", "y2",
                "center_u", "center_v",
            ])

            while limit == 0 or frame_id < limit:
                ok, frame = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = model.predict(rgb, threshold=0.15)

                names = get_names(detections)
                keep = np.isin(names, list(VEHICLES))
                detections = detections[keep]
                detections.data["class_name"] = names[keep]

                tracked = tracker.update(detections)
                names = get_names(tracked)

                for i, box in enumerate(tracked.xyxy):
                    if tracked.tracker_id is None:
                        continue

                    track_id = int(tracked.tracker_id[i])
                    if track_id < 0:
                        continue

                    x1, y1, x2, y2 = map(float, box)
                    score = float(tracked.confidence[i])
                    label = str(names[i])
                    u = (x1 + x2) / 2
                    v = (y1 + y2) / 2

                    table.writerow([
                        frame_id,
                        round(frame_id / fps, 6),
                        video.stem,
                        track_id,
                        label,
                        round(score, 5),
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                        round(u, 2),
                        round(v, 2),
                    ])

                    color = (
                        (0, 210, 255)
                        if label == "truck"
                        else (70, 220, 70)
                    )

                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2,
                    )
                    cv2.circle(
                        frame, (int(u), int(v)), 4, color, -1
                    )
                    cv2.putText(
                        frame,
                        f"{label} ID:{track_id} {score:.2f}",
                        (max(0, int(x1)), max(22, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                cv2.putText(
                    frame,
                    f"{video.stem} | frame {frame_id} | 2D tracking",
                    (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                writer.write(frame)

                if frame_id in (0, 29, 150):
                    cv2.imwrite(
                        str(folder / f"frame{frame_id}.jpg"),
                        frame,
                    )

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="configs")
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    if args.frames < 0:
        parser.error("--frames должен быть >= 0")

    source = Path(args.videos)

    if source.is_file():
        videos = [source]
    elif source.is_dir():
        videos = sorted(
            path for path in source.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
        )
    else:
        raise SystemExit(f"Путь не найден: {source}")

    if not videos:
        raise SystemExit("Видеофайлы не найдены")

    output = Path("outputs") / datetime.now().strftime(
        "rfdetr_%Y%m%d_%H%M%S_%f"
    )
    output.mkdir(parents=True)

    print("Найдены видео:", flush=True)
    for video in videos:
        print(f"  {video}", flush=True)

    print(
        "\nЗагрузка RF-DETR Small. "
        "Первый запуск может скачивать веса.",
        flush=True,
    )
    model = RFDETRSmall()

    failed = []
    for index, video in enumerate(videos, 1):
        folder = output / f"{index:02d}_{video.stem}"
        folder.mkdir()

        print(
            f"\n[{index}/{len(videos)}] Обработка {video}",
            flush=True,
        )

        try:
            process_video(model, video, folder, args.frames)
        except Exception as error:
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
