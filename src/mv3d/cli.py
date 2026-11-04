from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from mv3d.calibration.camera_model import load_camera_from_csv
from mv3d.detection.detector import VehicleDetector
from mv3d.detection.localize import localize_detection
from mv3d.detection.quality import observation_weight
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
    detect.add_argument("--score", type=float, default=0.15)
    detect.add_argument("--plane-z", type=float, default=300.0)

    _add_run_args(
        sub.add_parser("csv", help="Только CSV: быстрее, ролики не пишутся"),
        write_csv=True,
        write_video=False,
    )
    _add_run_args(
        sub.add_parser("video", help="Только размеченные ролики, без CSV"),
        write_csv=False,
        write_video=True,
    )
    _add_run_args(
        sub.add_parser("track", help="CSV и размеченные ролики"),
        write_csv=True,
        write_video=True,
    )

    score = sub.add_parser("score", help="Покадровые ошибки по готовому track.csv")
    score.add_argument("track", help="track.csv или папка прогона")
    score.add_argument(
        "--gt",
        default="configs/calibration/object_transform_SM_AA_track.csv",
        help="Эталон Frame;Loc_X;Loc_Y;Loc_Z",
    )
    score.add_argument("--out", default=None, help="Куда писать errors.csv")

    train = sub.add_parser(
        "train",
        help="Аугментация и дообучение RF-DETR nano, по одной камере в valid",
    )
    train.add_argument("--config", default="configs/cameras.yml")
    train.add_argument("--stride", type=int, default=12, help="Брать каждый N-й кадр")
    train.add_argument("--score", type=float, default=0.25, help="Порог псевдоразметки")
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--dataset", default="data/truck")
    train.add_argument("--output", default="outputs/rfdetr_nano")
    train.add_argument("--weights", default="weights/rfdetr-nano-truck.pth")
    train.add_argument("--seed", type=int, default=0)

    sub.add_parser("ui", help="Окно: видео, CSV, графики и ролики с боксами")
    return parser.parse_args(argv)


def _add_run_args(
    parser: argparse.ArgumentParser,
    *,
    write_csv: bool,
    write_video: bool,
) -> None:
    parser.set_defaults(write_csv=write_csv, write_video=write_video)
    parser.add_argument("--config", default="configs/cameras.yml")
    parser.add_argument(
        "--camera",
        default=None,
        help="Только эта камера из конфига, например cam1",
    )
    parser.add_argument("--frames", type=int, default=0, help="0 — все кадры")
    parser.add_argument("--score", type=float, default=0.15)
    parser.add_argument("--out", default="outputs")


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
        weight, sharpness, visibility = observation_weight(frame, detection)
        xyz_text = (
            f"X={xyz[0]:.1f} Y={xyz[1]:.1f} Z={xyz[2]:.1f}"
            if xyz is not None
            else "3D=n/a"
        )
        print(
            f"frame={frame_index} {detection.label} score={detection.score:.2f} "
            f"w={weight:.3f} sharp={sharpness:.2f} vis={visibility:.2f} "
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


def _run_score(args: argparse.Namespace) -> int:
    from mv3d.tracking.pipeline import score_coordinates

    score_coordinates(args.track, gt_path=args.gt, out=args.out)
    return 0


def _run_track(args: argparse.Namespace) -> int:
    from mv3d.tracking.pipeline import run_tracking

    if args.frames < 0:
        raise SystemExit("--frames должен быть >= 0")
    run_tracking(
        config_path=args.config,
        frames=args.frames,
        score_threshold=args.score,
        output_root=args.out,
        write_csv=args.write_csv,
        write_video=args.write_video,
        camera=args.camera,
    )
    return 0


def _run_train(args: argparse.Namespace) -> int:
    from mv3d.detection.train import train_nano

    if args.stride < 1:
        raise SystemExit("--stride должен быть >= 1")
    if args.epochs < 1:
        raise SystemExit("--epochs должен быть >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size должен быть >= 1")
    train_nano(
        config_path=args.config,
        stride=args.stride,
        score_threshold=args.score,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_dir=args.dataset,
        output_dir=args.output,
        weights_out=args.weights,
        seed=args.seed,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "check":
        return _run_check()
    if args.command == "detect":
        return _run_detect(args)
    if args.command in {"csv", "video", "track"}:
        return _run_track(args)
    if args.command == "score":
        return _run_score(args)
    if args.command == "train":
        return _run_train(args)
    if args.command == "ui":
        return _run_ui()
    parse_args(["--help"])
    return 0


def _run_ui() -> int:
    try:
        from mv3d.ui.app import main as ui_main
    except ImportError as exc:
        raise SystemExit(f"Не удалось открыть окно: {exc}") from exc
    return ui_main()


if __name__ == "__main__":
    raise SystemExit(main())
