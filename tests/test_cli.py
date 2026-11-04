from mv3d import __version__
from mv3d.cli import parse_args


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