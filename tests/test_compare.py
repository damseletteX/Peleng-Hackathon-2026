import math
from pathlib import Path

import pytest

from mv3d.viz.compare import build_payload, compare_tracks, read_loc_csv, write_report


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_blank_cells_are_gaps_and_do_not_count(tmp_path: Path):
    gt = tmp_path / "gt.csv"
    pred = tmp_path / "track.csv"
    _write(gt, "Frame;Loc_X;Loc_Y;Loc_Z\n0;0;0;300\n1;10;0;300\n2;;;\n")
    _write(pred, "Frame;Loc_X;Loc_Y;Loc_Z\n0;3;4;300\n1;;;\n2;1;1;1\n")

    report = compare_tracks(read_loc_csv(pred), read_loc_csv(gt))

    assert report["frames"] == [0]
    stats = report["stats"]
    assert stats["count"] == 1
    assert stats["rmse"] == pytest.approx(5.0)
    assert stats["mae"] == pytest.approx(5.0)
    assert stats["bias_x"] == pytest.approx(3.0)
    assert stats["bias_y"] == pytest.approx(4.0)
    assert stats["bias_z"] == pytest.approx(0.0)
    assert stats["rmse_x"] == pytest.approx(3.0)


def test_two_frames_rmse(tmp_path: Path):
    gt = tmp_path / "gt.csv"
    pred = tmp_path / "track.csv"
    _write(gt, "Frame,Loc_X,Loc_Y,Loc_Z\n0,0,0,300\n1,0,0,300\n")
    _write(pred, "Frame,Loc_X,Loc_Y,Loc_Z\n0,3,4,300\n1,10,0,300\n")

    stats = compare_tracks(read_loc_csv(pred), read_loc_csv(gt))["stats"]

    assert stats["rmse"] == pytest.approx(math.sqrt(62.5))
    assert stats["mae"] == pytest.approx(7.5)
    assert stats["max"] == pytest.approx(10.0)


def test_report_embeds_run_and_cameras(tmp_path: Path):
    gt = tmp_path / "gt.csv"
    run = tmp_path / "outputs" / "track_demo"
    _write(gt, "Frame;Loc_X;Loc_Y;Loc_Z\n0;0;0;300\n1;10;0;300\n")
    _write(run / "track.csv", "Frame;Loc_X;Loc_Y;Loc_Z\n0;3;4;300\n1;10;0;300\n")
    _write(run / "cam1.csv", "Frame;Loc_X;Loc_Y;Loc_Z\n0;1;0;300\n")
    out = tmp_path / "compare.html"

    write_report(out, gt_path=gt, outputs=tmp_path / "outputs", track=run)
    text = out.read_text(encoding="utf-8")

    assert "__PAYLOAD_JSON__" not in text
    assert "track_demo" in text
    assert "cam1" in text
    assert '"gtName":"gt.csv"' in text


def test_missing_track_is_an_error(tmp_path: Path):
    gt = tmp_path / "gt.csv"
    _write(gt, "Frame;Loc_X;Loc_Y;Loc_Z\n0;0;0;300\n")
    with pytest.raises(SystemExit):
        build_payload(gt, tmp_path, tmp_path / "missing.csv")


def test_project_ground_truth_if_present():
    path = Path("configs/calibration/object_transform_SM_AA_track.csv")
    if not path.is_file():
        return
    track = read_loc_csv(path)
    assert track.frames[0] == 0
    assert track.x[0] == pytest.approx(-21500.0)
    assert track.y[0] == pytest.approx(-12800.0)
    assert track.z[0] == pytest.approx(300.0)
    assert track.frames == sorted(track.frames)
    assert len(track.frames) > 100
