from dataclasses import dataclass

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
