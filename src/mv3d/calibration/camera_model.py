from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    focal_mm: float
    sensor_width_mm: float = 23.76
    sensor_height_mm: float = 13.365

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                "Ширина и высота изображения должны быть положительными."
            )

        values = np.array(
            [
                self.focal_mm,
                self.sensor_width_mm,
                self.sensor_height_mm,
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Фокус и размеры сенсора должны быть конечными числами."
            )

        if np.any(values <= 0):
            raise ValueError(
                "Фокус и размеры сенсора должны быть положительными."
            )

    @property
    def K(self) -> np.ndarray:
        """Внутренние параметры для исходного разрешения изображения."""
        fx = self.focal_mm * self.width / self.sensor_width_mm
        fy = self.focal_mm * self.height / self.sensor_height_mm

        cx = self.width / 2.0
        cy = self.height / 2.0

        return np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Camera:
    """Камера Unreal: внутренние параметры + поза в мировой СК (см)."""

    camera_id: str
    intrinsics: CameraIntrinsics
    center: np.ndarray
    roll_deg: float
    pitch_deg: float
    yaw_deg: float

    @property
    def K(self) -> np.ndarray:
        return self.intrinsics.K

    @property
    def R(self) -> np.ndarray:
        """Мир -> камера OpenCV (X вправо, Y вниз, Z вперёд)."""
        return unreal_rotation_to_opencv_R(self.roll_deg, self.pitch_deg, self.yaw_deg)

    @property
    def P(self) -> np.ndarray:
        """Матрица проекции 3x4: x ~ P [X Y Z 1]."""
        extra = np.hstack([self.R, -self.R @ self.center.reshape(3, 1)])
        return self.K @ extra

    def project(self, points_world: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_world, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        cam = (self.R @ (pts - self.center).T).T
        proj = (self.K @ cam.T).T
        uv = proj[:, :2] / proj[:, 2:3]
        return uv if points_world.ndim == 2 else uv[0]

    def ray_from_pixel(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Луч в мире: origin = центр камеры, direction единичный."""
        u, v = np.asarray(uv, dtype=np.float64)
        pixel = np.array([u, v, 1.0], dtype=np.float64)
        cam_dir = np.linalg.inv(self.K) @ pixel
        world_dir = self.R.T @ cam_dir
        world_dir = world_dir / np.linalg.norm(world_dir)
        return self.center.copy(), world_dir


def unreal_rotation_to_opencv_R(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """Поворот Unreal (X вперёд, Y вправо, Z вверх) -> строки осей OpenCV."""
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    # Столбцы: локальные оси камеры Unreal в мире (forward, right, up).
    r_ue = np.array(
        [
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ],
        dtype=np.float64,
    )
    return np.stack([r_ue[:, 1], -r_ue[:, 2], r_ue[:, 0]], axis=0)


def make_project_intrinsics(
    width: int = 1280,
    height: int = 720,
) -> dict[str, CameraIntrinsics]:
    """Создать внутренние параметры пяти камер по присланным данным."""
    focal_lengths_mm = {
        "cam1": 35.0,
        "cam2": 60.0,
        "cam3": 35.0,
        "cam4": 50.0,
        "cam5": 20.0,
    }

    return {
        camera_id: CameraIntrinsics(
            width=width,
            height=height,
            focal_mm=focal_mm,
        )
        for camera_id, focal_mm in focal_lengths_mm.items()
    }


def load_camera_track(
    path: str | Path,
    camera_id: str,
    width: int = 1280,
    height: int = 720,
    focal_mm: float | None = None,
) -> list[Camera]:
    """Поза камеры на каждом кадре CSV."""
    csv_path = Path(path)
    with csv_path.open(encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
        rows = [line.strip() for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"В {csv_path.name} нет кадров.")

    focal_mm = _parse_focal_mm(header, camera_id, fallback=focal_mm)
    intrinsics = CameraIntrinsics(width=width, height=height, focal_mm=focal_mm)
    cameras: list[Camera] = []
    for line in rows:
        parts = line.split(";")
        center = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
        roll, pitch, yaw = float(parts[4]), float(parts[5]), float(parts[6])
        cameras.append(
            Camera(
                camera_id=camera_id,
                intrinsics=intrinsics,
                center=center,
                roll_deg=roll,
                pitch_deg=pitch,
                yaw_deg=yaw,
            )
        )
    return cameras


def load_camera_from_csv(
    path: str | Path,
    camera_id: str,
    frame: int = 0,
    width: int = 1280,
    height: int = 720,
    focal_mm: float | None = None,
) -> Camera:
    track = load_camera_track(
        path, camera_id, width=width, height=height, focal_mm=focal_mm
    )
    if frame < 0 or frame >= len(track):
        raise ValueError(f"Кадр {frame} отсутствует в {Path(path).name}.")
    return track[frame]


def _parse_focal_mm(header: str, camera_id: str, fallback: float | None = None) -> float:
    tokens = [token.strip() for token in header.split(";")]
    for index, token in enumerate(tokens):
        if "focal" in token.lower() and index + 1 < len(tokens) and tokens[index + 1]:
            return float(tokens[index + 1].replace(",", "."))
    if fallback is not None:
        return float(fallback)
    known = make_project_intrinsics()
    if camera_id not in known:
        raise ValueError(f"Для {camera_id} не задан фокус.")
    return known[camera_id].focal_mm


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    cameras = make_project_intrinsics()

    for camera_id, intrinsics in cameras.items():
        print(
            f"\n{camera_id}: "
            f"{intrinsics.width}x{intrinsics.height}, "
            f"focal={intrinsics.focal_mm:g} мм"
        )
        print(intrinsics.K)


if __name__ == "__main__":
    main()
