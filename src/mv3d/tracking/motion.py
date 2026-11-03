from __future__ import annotations

import numpy as np


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
        return self.position

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
        innovation = measurement - observation @ self.x
        innovation_cov = observation @ self.p @ observation.T + noise
        gain = self.p @ observation.T @ np.linalg.inv(innovation_cov)
        self.x = self.x + gain @ innovation
        self.p = (np.eye(6) - gain @ observation) @ self.p
        return self.x[:3].copy()


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
