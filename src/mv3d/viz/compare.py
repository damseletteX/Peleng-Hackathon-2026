from __future__ import annotations

import csv
import json
import math
import re
import webbrowser
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GT = Path("configs/calibration/object_transform_SM_AA_track.csv")
_TOKEN = "__PAYLOAD_JSON__"
_CAM_CSV = re.compile(r"cam\d+\.csv$", re.IGNORECASE)


@dataclass(frozen=True)
class Track:
    frames: list[int]
    x: list[float | None]
    y: list[float | None]
    z: list[float | None]


def read_loc_csv(path: Path) -> Track:
    """CSV ??? ? ???????: Frame;Loc_X;Loc_Y;Loc_Z. ?????? ?????? � ??????? ?????."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return Track([], [], [], [])
    delimiter = ";" if lines[0].count(";") >= lines[0].count(",") else ","
    rows = list(csv.reader(lines, delimiter=delimiter))
    header = [cell.strip().lower() for cell in rows[0]]

    def column(*names: str) -> int:
        for name in names:
            if name in header:
                return header.index(name)
        raise ValueError(names[0])

    try:
        indexes = (
            column("frame"),
            column("loc_x", "x"),
            column("loc_y", "y"),
            column("loc_z", "z"),
        )
        start = 1
    except ValueError:
        indexes = (0, 1, 2, 3)
        start = 0

    frames: list[int] = []
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    need = max(indexes)
    for row in rows[start:]:
        if len(row) <= need or not row[indexes[0]].strip():
            continue
        try:
            frame = int(float(row[indexes[0]].strip()))
        except ValueError:
            continue
        frames.append(frame)
        xs.append(_number(row[indexes[1]]))
        ys.append(_number(row[indexes[2]]))
        zs.append(_number(row[indexes[3]]))
    return Track(frames, xs, ys, zs)


def compare_tracks(pred: Track, gt: Track) -> dict:
    """?????? ?? ??????, ??? ???? ? ??????, ? ??????. ?????????? ? ??? ?? ????????, ??? CSV."""
    reference: dict[int, tuple[float, float, float]] = {}
    for frame, x, y, z in zip(gt.frames, gt.x, gt.y, gt.z, strict=True):
        if x is None or y is None or z is None:
            continue
        reference[frame] = (x, y, z)
    dx: list[float] = []
    dy: list[float] = []
    dz: list[float] = []
    frames: list[int] = []
    for frame, x, y, z in zip(pred.frames, pred.x, pred.y, pred.z, strict=True):
        if x is None or y is None or z is None or frame not in reference:
            continue
        gx, gy, gz = reference[frame]
        frames.append(frame)
        dx.append(x - gx)
        dy.append(y - gy)
        dz.append(z - gz)
    return {"frames": frames, "stats": _stats(dx, dy, dz)}


def build_payload(gt_path: Path, outputs: Path, track: Path | None) -> dict:
    if not gt_path.is_file():
        raise SystemExit(f"?????? ?? ??????: {gt_path}")
    gt = read_loc_csv(gt_path)
    runs = [_run_record(path) for path in _track_files(outputs, track)]
    return {
        "gtName": gt_path.name,
        "gtFound": True,
        "gt": _series_json(gt),
        "runs": runs,
    }


def write_report(
    output: Path,
    gt_path: Path = DEFAULT_GT,
    outputs: Path = Path("outputs"),
    track: Path | None = None,
) -> Path:
    payload = build_payload(gt_path, outputs, track)
    html = _render_html(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    gt = read_loc_csv(gt_path)
    print(f"?????? {gt_path}: {len(gt.frames)} ??????", flush=True)
    if not payload["runs"]:
        print("track.csv ?? ??????. ?? ???????? ?????? ??????, ???? ????? ??????? ???????.", flush=True)
    for run in payload["runs"]:
        report = compare_tracks(read_loc_csv(Path(run["path"])), gt)
        stats = report["stats"]
        if stats["count"]:
            print(
                f"{run['label']}: RMSE {stats['rmse']:.1f} ??, "
                f"??????? {stats['mae']:.1f} ??, ?????? {stats['count']}",
                flush=True,
            )
        else:
            print(f"{run['label']}: ??? ????? ?????? ? ????????", flush=True)
    print(f"???????: {output.resolve()}", flush=True)
    return output


def open_report(
    output: Path,
    gt_path: Path = DEFAULT_GT,
    outputs: Path = Path("outputs"),
    track: Path | None = None,
    open_browser: bool = True,
) -> Path:
    path = write_report(output, gt_path=gt_path, outputs=outputs, track=track)
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
    return path


def _number(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _stats(dx: list[float], dy: list[float], dz: list[float]) -> dict:
    count = len(dx)
    empty = {
        "count": 0,
        "rmse": None,
        "mae": None,
        "max": None,
        "rmse_x": None,
        "rmse_y": None,
        "rmse_z": None,
        "bias_x": None,
        "bias_y": None,
        "bias_z": None,
    }
    if count == 0:
        return empty
    squared = [x * x + y * y + z * z for x, y, z in zip(dx, dy, dz, strict=True)]
    errors = [math.sqrt(value) for value in squared]
    return {
        "count": count,
        "rmse": math.sqrt(sum(squared) / count),
        "mae": sum(errors) / count,
        "max": max(errors),
        "rmse_x": math.sqrt(sum(value * value for value in dx) / count),
        "rmse_y": math.sqrt(sum(value * value for value in dy) / count),
        "rmse_z": math.sqrt(sum(value * value for value in dz) / count),
        "bias_x": sum(dx) / count,
        "bias_y": sum(dy) / count,
        "bias_z": sum(dz) / count,
    }


def _series_json(track: Track) -> dict:
    def cell(value: float | None) -> float | None:
        return None if value is None else round(float(value), 4)

    return {
        "frames": track.frames,
        "x": [cell(value) for value in track.x],
        "y": [cell(value) for value in track.y],
        "z": [cell(value) for value in track.z],
    }


def _track_files(outputs: Path, track: Path | None) -> list[Path]:
    if track is None:
        if not outputs.is_dir():
            return []
        return sorted(path for path in outputs.rglob("track.csv") if path.is_file())
    if not track.exists():
        raise SystemExit(f"?? ?????? ????: {track}")
    if track.is_dir():
        csv_path = track / "track.csv"
        if not csv_path.is_file():
            raise SystemExit(f"? ????? ??? track.csv: {track}")
        return [csv_path]
    return [track]


def _run_record(track_csv: Path) -> dict:
    cameras = []
    for path in sorted(track_csv.parent.glob("*.csv")):
        if _CAM_CSV.fullmatch(path.name) is None:
            continue
        cameras.append({"id": path.stem, "series": _series_json(read_loc_csv(path))})
    label = track_csv.parent.name if track_csv.name == "track.csv" else track_csv.name
    return {
        "id": f"{track_csv.parent.name}/{track_csv.name}",
        "label": label,
        "path": str(track_csv.resolve()),
        "fused": _series_json(read_loc_csv(track_csv)),
        "cameras": cameras,
    }


def _render_html(payload: dict) -> str:
    template_path = Path(__file__).with_name("compare_page.html")
    template = template_path.read_text(encoding="utf-8")
    if _TOKEN not in template:
        raise SystemExit(f"? ??????? ??? {_TOKEN}: {template_path}")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return template.replace(_TOKEN, data.replace("<", "\\u003c"))
