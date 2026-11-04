from pathlib import Path

import pytest
import yaml

from mv3d.tracking.pipeline import run_tracking
from mv3d.ui.session import (
    CameraEntry,
    camera_ids,
    load_camera_entries,
    save_camera_list,
    startup_cameras,
    write_run_config,
)


def test_startup_cameras_restore_original_five(tmp_path):
    config = tmp_path / "cameras.yml"
    config.write_text(
        "cameras:\n  cam1:\n    video: other.mp4\n    calibration: cal.csv\n    focal_mm: 35\n",
        encoding="utf-8",
    )
    entries = startup_cameras(config)
    assert [entry.camera_id for entry in entries] == ["cam1", "cam2", "cam3", "cam4", "cam5"]
    assert entries[0].video == "other.mp4"
    assert entries[1].video.endswith("cam2.mp4")


def test_camera_ids_follow_config():
    assert camera_ids("configs/cameras.yml") == ["cam1", "cam2", "cam3", "cam4", "cam5"]


def test_write_run_config_keeps_calibration(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    dest = tmp_path / "run.yml"
    write_run_config("configs/cameras.yml", {"cam2": video}, dest)
    data = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert list(data["cameras"]) == ["cam2"]
    assert data["cameras"]["cam2"]["video"] == str(video)
    assert data["cameras"]["cam2"]["calibration"].endswith(
        "object_transform_CineCameraActor2.csv"
    )
    assert data["object_plane_z_cm"] == 300.0


def test_save_camera_list_keeps_preamble_and_video(tmp_path):
    source = Path("configs/cameras.yml")
    config = tmp_path / "cameras.yml"
    config.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    entries = load_camera_entries(config)
    calib = entries[0].calibration
    entries.append(CameraEntry("cam6", calib, 28.0))
    save_camera_list(config, entries)

    text = config.read_text(encoding="utf-8")
    assert "object_plane_z_cm" in text
    assert text.split("cameras:", 1)[0].strip()
    saved = load_camera_entries(config)
    assert [entry.camera_id for entry in saved] == ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6"]
    assert saved[0].video.endswith("cam1.mp4")
    kept = yaml.safe_load(text)
    assert kept["cameras"]["cam3"]["frame_offset"] == 360
    assert kept["cameras"]["cam2"]["enabled"] is False
    assert saved[-1].focal_mm == 28.0
    assert saved[-1].video == ""


def test_write_run_config_rejects_unknown_camera(tmp_path):
    with pytest.raises(ValueError, match="cam9"):
        write_run_config("configs/cameras.yml", {"cam9": Path("x.mp4")}, tmp_path / "run.yml")


def test_run_tracking_requires_an_output():
    with pytest.raises(SystemExit, match="Нужен выход"):
        run_tracking(write_csv=False, write_video=False, write_plots=False)
