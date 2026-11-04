from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CameraEntry:
    camera_id: str
    calibration: str
    focal_mm: float
    video: str = ""


# Исходный тест хакатона. Окно стартует с него, список можно менять.
DEFAULT_CAMERAS = (
    CameraEntry(
        "cam1",
        "configs/calibration/object_transform_CineCameraActor1.csv",
        35.0,
        "configs/cam1.mp4",
    ),
    CameraEntry(
        "cam2",
        "configs/calibration/object_transform_CineCameraActor2.csv",
        60.0,
        "configs/cam2.mp4",
    ),
    CameraEntry(
        "cam3",
        "configs/calibration/object_transform_CineCameraActor3.csv",
        35.0,
        "configs/cam3.mp4",
    ),
    CameraEntry(
        "cam4",
        "configs/calibration/object_transform_CineCameraActor4.csv",
        50.0,
        "configs/cam4.mp4",
    ),
    CameraEntry(
        "cam5",
        "configs/calibration/object_transform_CineCameraActor5.csv",
        20.0,
        "configs/cam5.mp4",
    ),
)


def camera_ids(config_path: str | Path = "configs/cameras.yml") -> list[str]:
    return [entry.camera_id for entry in load_camera_entries(config_path)]


def load_camera_entries(config_path: str | Path) -> list[CameraEntry]:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    cameras = data.get("cameras") or {}
    entries: list[CameraEntry] = []
    for camera_id, cfg in cameras.items():
        cfg = cfg or {}
        entries.append(
            CameraEntry(
                str(camera_id),
                str(cfg.get("calibration") or ""),
                float(cfg.get("focal_mm", 35.0)),
                str(cfg.get("video") or ""),
            )
        )
    return entries


def startup_cameras(config_path: str | Path) -> list[CameraEntry]:
    """Пять камер исходного теста, плюс камеры, которые добавили в конфиг."""
    saved = {entry.camera_id: entry for entry in load_camera_entries(config_path)}
    ordered = [saved.get(entry.camera_id, entry) for entry in DEFAULT_CAMERAS]
    known = {entry.camera_id for entry in DEFAULT_CAMERAS}
    ordered.extend(entry for camera_id, entry in saved.items() if camera_id not in known)
    return ordered


def _as_project_path(path: str) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        relative = candidate.resolve().relative_to(Path.cwd().resolve())
    except (OSError, ValueError):
        return path
    return relative.as_posix()


def save_camera_list(config_path: str | Path, entries: list[CameraEntry]) -> None:
    """Записать список камер, не трогая ключи выше блока cameras:."""
    path = Path(config_path)
    raw = path.read_text(encoding="utf-8") if path.is_file() else ""
    data = yaml.safe_load(raw) or {}
    previous = data.get("cameras") or {}
    cameras: dict[str, dict] = {}
    seen: set[str] = set()
    for entry in entries:
        camera_id = entry.camera_id.strip()
        if not camera_id:
            raise ValueError("У камеры должно быть имя.")
        if camera_id in seen:
            raise ValueError(f"Имя {camera_id} повторяется.")
        seen.add(camera_id)
        if not entry.calibration:
            raise ValueError(f"Для {camera_id} не выбрана калибровка.")
        old = previous.get(camera_id) or {}
        cameras[camera_id] = {
            "video": _as_project_path(str(entry.video or old.get("video") or "")),
            "calibration": _as_project_path(entry.calibration),
            "focal_mm": float(entry.focal_mm),
        }
    if not cameras:
        raise ValueError("В списке нет ни одной камеры.")
    head: list[str] = []
    for line in raw.splitlines():
        if line.startswith("cameras:"):
            break
        head.append(line)
    dumped = yaml.safe_dump({"cameras": cameras}, allow_unicode=True, sort_keys=False)
    text = "\n".join(head).rstrip() + "\n" + dumped
    if not text.endswith("\n"):
        text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_run_config(
    config_path: str | Path,
    assignments: dict[str, Path],
    dest: Path,
) -> None:
    """Конфиг прогона: выбранные id и загруженные ролики, калибровка из исходного файла."""
    source = Path(config_path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    known = data["cameras"]
    chosen = {}
    for camera_id, video in assignments.items():
        if camera_id not in known:
            names = ", ".join(known)
            raise ValueError(f"Камеры {camera_id} нет в конфиге. Доступны: {names}")
        spec = dict(known[camera_id])
        spec["video"] = str(video)
        chosen[camera_id] = spec
    if not chosen:
        raise ValueError("Не выбрана ни одна камера.")
    data["cameras"] = chosen
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
