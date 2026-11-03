"""Разметка кадров роликов и дообучение RF-DETR Nano.

Кросс-валидация по камерам: каждый фолд убирает одну камеру в valid,
остальные идут в train вместе с аугментациями.
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from mv3d.detection.augment import TRUCK_TRAIN_AUG, expand_views
from mv3d.detection.detector import FINETUNED_WEIGHTS, VehicleDetector

TRUCK_CATEGORY_ID = 1


@dataclass(frozen=True)
class LabeledFrame:
    camera_id: str
    frame_index: int
    image: np.ndarray
    boxes: list[tuple[float, float, float, float]]


@dataclass(frozen=True)
class FoldResult:
    held_out: str
    map50_95: float
    weights: Path


def train_nano(
    config_path: str | Path = "configs/cameras.yml",
    *,
    stride: int = 6,
    score_threshold: float = 0.25,
    epochs: int = 8,
    batch_size: int = 4,
    dataset_dir: str | Path = "data/truck",
    output_dir: str | Path = "outputs/rfdetr_nano",
    weights_out: str | Path = FINETUNED_WEIGHTS,
    seed: int = 0,
) -> Path:
    """Пять фолдов: по очереди одна камера только для проверки."""
    if stride < 1:
        raise ValueError("stride должен быть >= 1")

    frames = collect_labeled_frames(
        config_path,
        stride=stride,
        score_threshold=score_threshold,
    )
    folds = leave_one_camera_out(frames)
    dataset_root = Path(dataset_dir)
    run_root = Path(output_dir)
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True)

    results: list[FoldResult] = []
    for index, (held_out, train_frames, valid_frames) in enumerate(folds, start=1):
        train_cameras = ", ".join(sorted({frame.camera_id for frame in train_frames}))
        print(
            f"\nФолд {index}/{len(folds)}: valid={held_out}, train={train_cameras}.",
            flush=True,
        )
        dataset = write_augmented_dataset(
            train_frames,
            valid_frames,
            dataset_root / f"heldout_{held_out}",
            seed=seed + index,
        )
        fold_dir = run_root / f"heldout_{held_out}"
        fold_dir.mkdir(parents=True)
        _fit(dataset, fold_dir, epochs=epochs, batch_size=batch_size)
        weights = fold_dir / "checkpoint_best_total.pth"
        if not weights.is_file():
            weights = _best_checkpoint(fold_dir)
        score = _best_map(fold_dir)
        results.append(FoldResult(held_out, score, weights))
        text = "n/a" if math.isnan(score) else f"{score:.3f}"
        print(f"Фолд {held_out}: mAP50:95={text}", flush=True)

    best = _select_best(results)
    destination = Path(weights_out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best.weights, destination)
    summary = {
        "folds": [
            {
                "held_out": item.held_out,
                "map50_95": None if math.isnan(item.map50_95) else item.map50_95,
                "weights": str(item.weights),
            }
            for item in results
        ],
        "selected_held_out": best.held_out,
        "weights": str(destination),
    }
    (run_root / "cv.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Лучший фолд, камера вне обучения: {best.held_out}", flush=True)
    print(f"Веса: {destination}", flush=True)
    return destination


def collect_labeled_frames(
    config_path: str | Path,
    *,
    stride: int,
    score_threshold: float,
) -> list[LabeledFrame]:
    """Псевдоразметка предобученным nano: кадр берётся, только если есть truck."""
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    cameras = config.get("cameras") or {}
    sources = [
        (camera_id, Path(spec["video"]))
        for camera_id, spec in cameras.items()
        if spec.get("video") and Path(spec["video"]).is_file()
    ]
    if not sources:
        raise SystemExit("В конфиге нет камер с существующим видео.")

    print("Разметка предобученным RF-DETR nano.", flush=True)
    detector = VehicleDetector(score_threshold=score_threshold, pretrained=True)
    labeled: list[LabeledFrame] = []
    for camera_id, video in sources:
        found = _label_video(detector, camera_id, video, stride, score_threshold)
        labeled.extend(found)
        print(f"  {camera_id}: {len(found)} кадров с грузовиком", flush=True)
    if not labeled:
        raise SystemExit("Детектор не нашёл ни одного грузовика. Датасет пуст.")
    return labeled


def leave_one_camera_out(
    frames: list[LabeledFrame],
) -> list[tuple[str, list[LabeledFrame], list[LabeledFrame]]]:
    """Каждый фолд держит одну камеру целиком в valid."""
    cameras = sorted({frame.camera_id for frame in frames})
    if len(cameras) < 2:
        raise SystemExit("Для кросс-валидации нужны кадры хотя бы с двух камер.")
    folds: list[tuple[str, list[LabeledFrame], list[LabeledFrame]]] = []
    for held_out in cameras:
        train_frames = [frame for frame in frames if frame.camera_id != held_out]
        valid_frames = [frame for frame in frames if frame.camera_id == held_out]
        if not train_frames or not valid_frames:
            continue
        folds.append((held_out, train_frames, valid_frames))
    if len(folds) < 2:
        raise SystemExit("Не собралось двух фолдов: у части камер нет грузовика.")
    return folds


def write_augmented_dataset(
    train_frames: list[LabeledFrame],
    valid_frames: list[LabeledFrame],
    dataset_dir: str | Path,
    seed: int,
) -> Path:
    """Train получает поворот, отражение и кроп. Valid остаётся исходными кадрами."""
    root = Path(dataset_dir)
    if root.exists():
        shutil.rmtree(root)
    rng = random.Random(seed)
    train_count = _write_split(root / "train", train_frames, rng, augment=True)
    valid_count = _write_split(root / "valid", valid_frames, rng, augment=False)
    print(
        f"Датасет: train {train_count} изображений из {len(train_frames)} кадров, "
        f"valid {valid_count}.",
        flush=True,
    )
    return root


def _label_video(
    detector: VehicleDetector,
    camera_id: str,
    video: Path,
    stride: int,
    score_threshold: float,
) -> list[LabeledFrame]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video}")
    labeled: list[LabeledFrame] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                detections = detector.detect(frame, score_threshold=score_threshold)
                boxes = [item.bbox for item in detections]
                if boxes:
                    labeled.append(
                        LabeledFrame(
                            camera_id=camera_id,
                            frame_index=frame_index,
                            image=frame,
                            boxes=boxes,
                        )
                    )
            frame_index += 1
    finally:
        capture.release()
    return labeled


def _write_split(
    folder: Path,
    frames: list[LabeledFrame],
    rng: random.Random,
    *,
    augment: bool,
) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    images: list[dict] = []
    annotations: list[dict] = []
    image_id = 1
    annotation_id = 1
    for frame in frames:
        views = (
            expand_views(frame.image, frame.boxes, rng)
            if augment
            else [("orig", frame.image, frame.boxes)]
        )
        for name, image, boxes in views:
            file_name = f"{frame.camera_id}_{frame.frame_index:04d}_{name}.jpg"
            if not cv2.imwrite(str(folder / file_name), image):
                raise RuntimeError(f"Не удалось записать {file_name}")
            height, width = image.shape[:2]
            images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                }
            )
            for box in boxes:
                x1, y1, x2, y2 = box
                width_box, height_box = x2 - x1, y2 - y1
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": TRUCK_CATEGORY_ID,
                        "bbox": [x1, y1, width_box, height_box],
                        "area": width_box * height_box,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
            image_id += 1
    payload = {
        "info": {"description": "Peleng truck, RF-DETR nano"},
        "licenses": [],
        "categories": [
            {"id": TRUCK_CATEGORY_ID, "name": "truck", "supercategory": "vehicle"}
        ],
        "images": images,
        "annotations": annotations,
    }
    (folder / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")
    return len(images)


def _fit(dataset: Path, output_dir: Path, *, epochs: int, batch_size: int) -> None:
    import torch
    from rfdetr import RFDETRNano

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Дообучение RF-DETR nano на {device}, эпох: {epochs}.", flush=True)
    model = RFDETRNano()
    model.train(
        dataset_dir=str(dataset),
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=1,
        lr=1e-4,
        lr_encoder=1.5e-5,
        warmup_epochs=1.0,
        weight_decay=1e-4,
        device=device,
        num_workers=0,
        aug_config=TRUCK_TRAIN_AUG,
        augmentation_backend="cpu",
        multi_scale=False,
        expanded_scales=False,
        use_ema=True,
        tensorboard=False,
        wandb=False,
        progress_bar="tqdm",
        class_names=["truck"],
        early_stopping=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        checkpoint_interval=1,
        eval_interval=1,
        run_test=False,
    )


def _best_checkpoint(output_dir: Path) -> Path:
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "last.pth"):
        path = output_dir / name
        if path.is_file():
            return path
    found = sorted(output_dir.glob("*.pth"))
    if not found:
        raise SystemExit(f"Обучение не записало веса в {output_dir}")
    return found[-1]


def _best_map(output_dir: Path) -> float:
    path = output_dir / "metrics.csv"
    if not path.is_file():
        return float("nan")
    best = float("-inf")
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                if not key or "mAP_50_95" not in key or not value:
                    continue
                try:
                    best = max(best, float(value))
                except ValueError:
                    continue
    if best == float("-inf"):
        return float("nan")
    return best


def _select_best(results: list[FoldResult]) -> FoldResult:
    scored = [item for item in results if not math.isnan(item.map50_95)]
    if not scored:
        return results[-1]
    return max(scored, key=lambda item: item.map50_95)
