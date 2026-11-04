from __future__ import annotations

import numpy as np

# Без измерения скорость затухает: за 5 с остаётся половина.
# Иначе точка десятки секунд летит по прямой и уходит от дороги.
_COAST_HALF_LIFE_S = 5.0


class TrajectoryFilter:
    """Фильтр постоянной скорости для одной 3D-траектории, сантиметры.

    Шум измерения растёт, когда вес кадра маленький: нечёткий или
    обрезанный вид остаётся в решении, но почти его не сдвигает.
    Кадр без измерения только предсказывает состояние.
    """

    def __init__(self, meas_std_cm: float = 80.0, accel_std_cm_s2: float = 80.0) -> None:
        self.meas_std_cm = meas_std_cm
        self.accel_std_cm_s2 = accel_std_cm_s2
        self.x: np.ndarray | None = None
        self.p: np.ndarray | None = None

    @property
    def position(self) -> np.ndarray | None:
        if self.x is None:
            return None
        return self.x[:3].copy()

    def peek(self, dt: float) -> np.ndarray | None:
        """Положение на следующем кадре без изменения состояния."""
        if self.x is None:
            return None
        return np.array(
            [
                self.x[0] + self.x[3] * dt,
                self.x[1] + self.x[4] * dt,
                self.x[2] + self.x[5] * dt,
            ],
            dtype=np.float64,
        )

    def step(
        self,
        measurement: np.ndarray | None,
        weight: float,
        dt: float,
    ) -> np.ndarray | None:
        if dt <= 0:
            raise ValueError("Шаг по времени должен быть положительным.")
        if self.x is not None:
            self._predict(dt)
        if measurement is not None and weight > 0:
            return self._update(np.asarray(measurement, dtype=np.float64), weight)
        self._fade_unseen_speed(dt)
        return self.position

    def _fade_unseen_speed(self, dt: float) -> None:
        if self.x is None:
            return
        self.x[3:6] *= 0.5 ** (dt / _COAST_HALF_LIFE_S)

    def _predict(self, dt: float) -> None:
        assert self.x is not None and self.p is not None
        transition = np.eye(6, dtype=np.float64)
        transition[0, 3] = dt
        transition[1, 4] = dt
        transition[2, 5] = dt
        self.x = transition @ self.x
        self.p = transition @ self.p @ transition.T + _process_noise(dt, self.accel_std_cm_s2)

    def _update(self, measurement: np.ndarray, weight: float) -> np.ndarray:
        if self.x is None:
            self.x = np.zeros(6, dtype=np.float64)
            self.x[:3] = measurement
            self.p = np.diag([25.0, 25.0, 25.0, 400.0, 400.0, 25.0]).astype(np.float64)
            return self.x[:3].copy()

        assert self.p is not None
        observation = np.zeros((3, 6), dtype=np.float64)
        observation[0, 0] = observation[1, 1] = observation[2, 2] = 1.0
        variance = (self.meas_std_cm ** 2) / max(weight, 1e-3)
        noise = np.eye(3, dtype=np.float64) * variance
        innovation = _along_road(self.x, measurement - observation @ self.x)
        innovation_cov = observation @ self.p @ observation.T + noise
        gain = self.p @ observation.T @ np.linalg.inv(innovation_cov)
        self.x = self.x + gain @ innovation
        self.p = (np.eye(6) - gain @ observation) @ self.p
        self._limit_speed()
        return self.x[:3].copy()

    def _limit_speed(self) -> None:
        """Грузовик не разгоняется быстрее 30 м/с и не взлетает."""
        assert self.x is not None
        horizontal = float(np.hypot(self.x[3], self.x[4]))
        if horizontal > 3000.0:
            self.x[3:5] *= 3000.0 / horizontal
        self.x[5] = float(np.clip(self.x[5], -150.0, 150.0))


def _along_road(state: np.ndarray, innovation: np.ndarray) -> np.ndarray:
    """Оставить продольный сдвиг и ослабить прыжок поперёк пройденного пути.

    Путь уже проеханных точек — это дорога, по которой едет грузовик.
    Боковой выброс детекции не должен уводить траекторию с неё.
    """
    heading = np.array([state[3], state[4], 0.0], dtype=np.float64)
    speed = float(np.hypot(heading[0], heading[1]))
    if speed < 100.0:
        return innovation
    heading /= speed
    along = float(np.dot(innovation, heading)) * heading
    cross = innovation - along
    return along + 0.25 * cross


def _process_noise(dt: float, accel_std: float) -> np.ndarray:
    variance = accel_std ** 2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    noise = np.zeros((6, 6), dtype=np.float64)
    for axis in range(3):
        noise[axis, axis] = 0.25 * dt4 * variance
        noise[axis, axis + 3] = 0.5 * dt3 * variance
        noise[axis + 3, axis] = 0.5 * dt3 * variance
        noise[axis + 3, axis + 3] = dt2 * variance
    return noise


# Луч через переднюю грань бокса на видимом участке первой камеры.
# Медиана отклонения от эталона: 1.7 м позади носа и 15.3 м вправо от хода.
_NOSE_AHEAD_CM = 167.0
_NOSE_LEFT_CM = 1535.0
_NOSE_MIN_SPEED_CM_S = 100.0


def place_actor_origin(point: np.ndarray, velocity: np.ndarray | None) -> np.ndarray:
    """Перенести точку луча к носу актора вдоль уже известного хода.

    Пока скорость мала, направление носа неизвестно и точка не сдвигается.
    """
    if velocity is None:
        return np.asarray(point, dtype=np.float64).copy()
    speed = float(np.hypot(velocity[0], velocity[1]))
    if speed < _NOSE_MIN_SPEED_CM_S:
        return np.asarray(point, dtype=np.float64).copy()
    heading = np.array([velocity[0], velocity[1]], dtype=np.float64) / speed
    left = np.array([-heading[1], heading[0]], dtype=np.float64)
    shifted = np.asarray(point, dtype=np.float64).copy()
    shifted[0] += heading[0] * _NOSE_AHEAD_CM + left[0] * _NOSE_LEFT_CM
    shifted[1] += heading[1] * _NOSE_AHEAD_CM + left[1] * _NOSE_LEFT_CM
    return shifted
