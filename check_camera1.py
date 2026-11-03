import numpy as np


def main() -> None:
    width = 1280
    height = 720

    focal_mm = 35.0
    sensor_width_mm = 23.76
    sensor_height_mm = 13.365

    K = np.array(
        [
            [focal_mm * width / sensor_width_mm, 0.0, width / 2.0],
            [0.0, focal_mm * height / sensor_height_mm, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    camera_center = np.array(
        [-24701.9592, 6623.5717, 1200.0],
        dtype=np.float64,
    )

    object_position = np.array(
        [-13889.5241, -7666.0552, 300.0],
        dtype=np.float64,
    )

    observed_uv = np.array([477.0, 560.0])

    # Проверяемая гипотеза: Roll=0, Pitch=0, Yaw=-70.
    yaw = np.deg2rad(-70.0)

    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    right = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    down = np.array([0.0, 0.0, -1.0])

    R = np.vstack([right, down, forward])

    point_camera = R @ (object_position - camera_center)

    print("Кадр: 150")
    print("Наблюдаемая точка:", observed_uv)
    print("Точка в системе камеры:", point_camera)

    if point_camera[2] <= 0:
        print("Гипотеза не подходит: объект оказался позади камеры.")
        return

    projected = K @ point_camera
    predicted_uv = projected[:2] / projected[2]

    error_px = np.linalg.norm(predicted_uv - observed_uv)

    print("Предсказанная точка:", np.round(predicted_uv, 2))
    print(f"Расстояние до ручной отметки: {error_px:.2f} пикс.")
    print("Это проверка гипотезы, а не подтверждённая калибровка.")


if __name__ == "__main__":
    main()
