from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from mv3d.calibration.camera_model import load_camera_from_csv
from mv3d.detection.detector import VehicleDetector
from mv3d.detection.localize import localize_detection
from mv3d.detection.visualize import draw_detection

CALIB_DIR = Path("configs/calibration")
CAMERA_CSV = {
    "cam1": CALIB_DIR / "object_transform_CineCameraActor1.csv",
    "cam2": CALIB_DIR / "object_transform_CineCameraActor2.csv",
    "cam3": CALIB_DIR / "object_transform_CineCameraActor3.csv",
    "cam4": CALIB_DIR / "object_transform_CineCameraActor4.csv",
    "cam5": CALIB_DIR / "object_transform_CineCameraActor5.csv",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Мультикамерный 3D-трекинг (Peleng Hackathon)")
    parser.add_argument("--version", action="version", version="peleng-hackathon-2026 0.1.0")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("check", help="Проверить окружение")

    detect = sub.add_parser("detect", help="Детекция транспорта и 3D-локализация")
    detect.add_argument("source", help="Изображение или видео")
    detect.add_argument("--camera", default="cam1", choices=sorted(CAMERA_CSV))
    detect.add_argument("--frame", type=int, default=0, help="Стартовый кадр для видео")
    detect.add_argument("--max-frames", type=int, default=1)
    detect.add_argument("--out", type=str, default=None)
    detect.add_argument("--score", type=float, default=0.35)
    detect.add_argument("--plane-z", type=float, default=300.0)
    return parser.parse_args(argv)


def _run_check() -> int:
    print("Checking project environment and dependencies...")
    try:
        import torch
        import torchvision

        print(f"torch {torch.__version__}, torchvision {torchvision.__version__}")
        print(f"opencv {cv2.__version__}")
        print("All checks passed!")
        return 0
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        return 1


def _run_detect(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"Не найден файл: {source}")

    detector = VehicleDetector(score_threshold=args.score)
    camera = load_camera_from_csv(CAMERA_CSV[args.camera], args.camera, frame=args.frame)

    image = cv2.imread(str(source))
    if image is not None:
        frames = [(args.frame, image)]
    else:
        frames = list(_iter_video_frames(source, args.frame, args.max_frames))

    if not frames:
        raise SystemExit("Не удалось прочитать кадры.")

    last_canvas = None
    for frame_index, frame in frames:
        camera = load_camera_from_csv(CAMERA_CSV[args.camera], args.camera, frame=frame_index)
        detection = detector.detect_best(frame)
        if detection is None:
            print(f"frame={frame_index}: объект не найден")
            last_canvas = frame
            continue
        xyz = localize_detection([camera], [detection], plane_z=args.plane_z)
        xyz_text = (
            f"X={xyz[0]:.1f} Y={xyz[1]:.1f} Z={xyz[2]:.1f}"
            if xyz is not None
            else "3D=n/a"
        )
        print(
            f"frame={frame_index} {detection.label} score={detection.score:.2f} "
            f"center={tuple(round(v, 1) for v in detection.center)} {xyz_text}"
        )
        last_canvas = draw_detection(frame, detection, xyz)

    if args.out and last_canvas is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), last_canvas):
            raise SystemExit(f"Не удалось сохранить {out_path}")
        print(f"Сохранено: {out_path}")
    return 0


def _iter_video_frames(path: Path, start: int, max_frames: int):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise SystemExit(f"Не удалось открыть видео: {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for offset in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            yield start + offset, frame
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "check":
        return _run_check()
    if args.command == "detect":
        return _run_detect(args)
    parse_args(["--help"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
