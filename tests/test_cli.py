import pytest

from mv3d import __version__
from mv3d.cli import parse_args
from mv3d.tracking.pipeline import _load_sources


def test_package_import():
    assert __version__ == "0.1.0"


def test_run_commands_choose_output():
    csv_args = parse_args(["csv"])
    assert csv_args.command == "csv"
    assert csv_args.write_csv and not csv_args.write_video

    video_args = parse_args(["video"])
    assert video_args.write_video and not video_args.write_csv

    both = parse_args(["track"])
    assert both.write_csv and both.write_video

    train_args = parse_args(["train", "--epochs", "3", "--stride", "10"])
    assert train_args.command == "train"
    assert train_args.epochs == 3
    assert train_args.stride == 10

    assert parse_args(["ui"]).command == "ui"

    one = parse_args(["csv", "--camera", "cam3"])
    assert one.camera == "cam3"
    assert parse_args(["track"]).camera is None
    assert parse_args(["video", "--camera", "cam2"]).camera == "cam2"


def test_load_sources_keeps_one_camera(tmp_path):
    video = tmp_path / "cam2.mp4"
    video.write_bytes(b"x")
    calib = tmp_path / "cam2.csv"
    calib.write_text("frame\n", encoding="utf-8")
    config = tmp_path / "cameras.yml"
    config.write_text(
        "\n".join(
            [
                "object_plane_z_cm: 300.0",
                "cameras:",
                "  cam1:",
                "    video: missing.mp4",
                "    calibration: missing.csv",
                "  cam2:",
                f"    video: {video}",
                f"    calibration: {calib}",
            ]
        ),
        encoding="utf-8",
    )

    plane_z, sources = _load_sources(config, camera="cam2")
    assert plane_z == 300.0
    assert [source.camera_id for source in sources] == ["cam2"]

    with pytest.raises(SystemExit, match="cam9"):
        _load_sources(config, camera="cam9")