from pathlib import Path

import pytest
import yaml

from mv3d.tracking.pipeline import run_tracking
from mv3d.ui.session import camera_ids, write_run_config


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


def test_write_run_config_rejects_unknown_camera(tmp_path):
    with pytest.raises(ValueError, match="cam9"):
        write_run_config("configs/cameras.yml", {"cam9": Path("x.mp4")}, tmp_path / "run.yml")


def test_run_tracking_requires_an_output():
    with pytest.raises(SystemExit, match="Нужен выход"):
        run_tracking(write_csv=False, write_video=False, write_plots=False)
