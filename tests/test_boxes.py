from mv3d.detection.detector import Detection
from mv3d.tracking.boxes import BoxKeeper
from mv3d.tracking.pipeline import _error_cells


def _truck(bbox: tuple[float, float, float, float], score: float = 0.8) -> Detection:
    return Detection(bbox=bbox, score=score, label="truck", label_id=8)


def test_a_sliver_at_the_edge_does_not_shrink_the_box():
    keeper = BoxKeeper()
    full = (400.0, 400.0, 700.0, 520.0)
    keeper.update([_truck(full)], 1280, 720)
    keeper.update([_truck(full)], 1280, 720)
    kept = keeper.update([_truck((1240.0, 410.0, 1278.0, 510.0), 0.27)], 1280, 720)
    assert kept is not None
    assert kept[2] - kept[0] > 200


def test_one_missed_frame_keeps_the_box():
    keeper = BoxKeeper()
    box = (100.0, 200.0, 300.0, 400.0)
    keeper.update([_truck(box)], 1280, 720)
    keeper.update([_truck(box)], 1280, 720)
    kept = keeper.update([], 1280, 720)
    assert kept is not None


def test_a_single_false_box_does_not_stick():
    keeper = BoxKeeper()
    keeper.update([_truck((10.0, 10.0, 80.0, 80.0), 0.9)], 1280, 720)
    assert keeper.update([], 1280, 720) is None


def test_the_box_is_kept_until_it_leaves_the_frame():
    keeper = BoxKeeper()
    keeper.update([_truck((1100.0, 400.0, 1276.0, 500.0))], 1280, 720)
    keeper.update([_truck((1140.0, 400.0, 1316.0, 500.0))], 1280, 720)
    seen = 0
    for _ in range(40):
        kept = keeper.update([], 1280, 720)
        if kept is not None:
            seen += 1
    assert seen >= 3
    assert keeper.update([], 1280, 720) is None or keeper.box is None


def test_error_row_is_empty_without_a_fix():
    assert _error_cells(None, None) == ["", "", "", ""]


def test_score_reads_a_finished_track(tmp_path):
    from mv3d.tracking.pipeline import score_coordinates

    gt = tmp_path / "gt.csv"
    gt.write_text("Frame;Loc_X;Loc_Y;Loc_Z\n0;0;0;0\n1;10;0;0\n", encoding="utf-8")
    track = tmp_path / "track.csv"
    track.write_text("Frame;Loc_X;Loc_Y;Loc_Z\n0;3;4;0\n1;;;\n", encoding="utf-8")
    out = score_coordinates(track, gt_path=gt)
    text = out.read_text(encoding="utf-8").splitlines()
    assert text[0] == "Frame;Err_X;Err_Y;Err_Z;Dist_cm"
    assert text[1].startswith("0;3.0000;4.0000;0.0000;5.0000")
    assert text[2] == "1;;;;"
