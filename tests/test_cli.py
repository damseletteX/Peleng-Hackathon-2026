from mv3d import __version__
from mv3d.cli import parse_args


def test_package_import():
    assert __version__ == "0.1.0"


def test_run_commands_choose_output_and_model():
    csv_args = parse_args(["csv", "--model", "small"])
    assert csv_args.command == "csv"
    assert csv_args.write_csv and not csv_args.write_video
    assert csv_args.model == "small"

    video_args = parse_args(["video", "--model", "large"])
    assert video_args.write_video and not video_args.write_csv
    assert video_args.model == "large"

    both = parse_args(["track"])
    assert both.write_csv and both.write_video
    assert both.model == "medium"

    assert parse_args(["csv", "--model", "nano"]).model == "nano"
    assert parse_args(["csv", "--model", "yolov8n"]).model == "yolov8n"
    assert parse_args(["track", "--model", "yolo26m"]).model == "yolo26m"
    assert parse_args(["video", "--model", "yolo11x"]).model == "yolo11x"