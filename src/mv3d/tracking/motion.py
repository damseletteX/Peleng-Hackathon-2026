from __future__ import annotations

import numpy as np

from mv3d.tracking.path_prior import attach, step_path

# Поворот за кадр с измерением. Без измерения дугу не крутим.
_TURN_SMOOTH = 0.15
_MAX_YAW_RATE = 0.45
# Вне дороги без бокса точку не выдумываем. На дороге сцены идём по хвосту.


class TrajectoryFilter:
    """Фильтр постоянной скорости и поворота, сантиметры.

    Пока есть измерение, точка и курс обновляются. Без измерения
    сохраняются скорость и угловая скорость: траектория заполняется
    предсказанием по дуге.
    """

    def __init__(self, meas_std_cm: float = 80.0, accel_std_cm_s2: float = 80.0) -> None:
        self.meas_std_cm = meas_std_cm
        self.accel_std_cm_s2 = accel_std_cm_s2
        self.x: np.ndarray | None = None
        self.p: np.ndarray | None = None
        self.omega = 0.0
        self._last_heading: float | None = None
        self.unseen_s = 0.0
        self._path_t: float | None = None

    @property
    def position(self) -> np.ndarray | None:
        if self.x is None:
            return None
        return self.x[:3].copy()

    def peek(self, dt: float) -> np.ndarray | None:
        """Положение на следующем кадре без изменения состояния."""
        if self.x is None:
            return None
        if self._path_t is not None:
            point, _, _ = step_path(self._path_t, dt)
            return point
        nxt = self.x.copy()
        _advance(nxt, self.omega if self.unseen_s == 0 else 0.0, dt)
        return nxt[:3].copy()

    def step(
        self,
        measurement: np.ndarray | None,
        weight: float,
        dt: float,
    ) -> np.ndarray | None:
        if dt <= 0:
            raise ValueError("Шаг по времени должен быть положительным.")
        has_measurement = measurement is not None and weight > 0
        if self.x is not None:
            self._predict(dt, coast=not has_measurement)
        if has_measurement:
            self.unseen_s = 0.0
            self._path_t = None
            return self._update(np.asarray(measurement, dtype=np.float64), weight, dt)
        self.unseen_s += dt
        if self._path_t is None and self.unseen_s > 1.5:
            return None
        return self.position

    def _predict(self, dt: float, coast: bool = False) -> None:
        assert self.x is not None and self.p is not None
        if coast and self._follow_path(dt):
            pass
        else:
            _advance(self.x, 0.0 if coast else self.omega, dt)
        transition = np.eye(6, dtype=np.float64)
        transition[0, 3] = dt
        transition[1, 4] = dt
        transition[2, 5] = dt
        self.p = transition @ self.p @ transition.T + _process_noise(dt, self.accel_std_cm_s2)

    def _follow_path(self, dt: float) -> bool:
        assert self.x is not None
        if self._path_t is None:
            self._path_t = attach(self.x[:3])
            if self._path_t is None:
                return False
        point, velocity, self._path_t = step_path(self._path_t, dt)
        self.x[:3] = point
        self.x[3:6] = velocity
        return True

    def _update(self, measurement: np.ndarray, weight: float, dt: float) -> np.ndarray:
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
        self._remember_turn(dt)
        return self.x[:3].copy()

    def _remember_turn(self, dt: float) -> None:
        assert self.x is not None
        speed = float(np.hypot(self.x[3], self.x[4]))
        if speed < 100.0 or dt <= 0:
            return
        angle = float(np.arctan2(self.x[4], self.x[3]))
        if self._last_heading is not None:
            delta = float(np.arctan2(np.sin(angle - self._last_heading), np.cos(angle - self._last_heading)))
            sample = float(np.clip(delta / dt, -_MAX_YAW_RATE, _MAX_YAW_RATE))
            self.omega = (1.0 - _TURN_SMOOTH) * self.omega + _TURN_SMOOTH * sample
        self._last_heading = angle

    def _limit_speed(self) -> None:
        """Грузовик не разгоняется быстрее 30 м/с и не взлетает."""
        assert self.x is not None
        horizontal = float(np.hypot(self.x[3], self.x[4]))
        if horizontal > 3000.0:
            self.x[3:5] *= 3000.0 / horizontal
        self.x[5] = float(np.clip(self.x[5], -150.0, 150.0))


def _advance(state: np.ndarray, omega: float, dt: float) -> None:
    vx, vy, vz = float(state[3]), float(state[4]), float(state[5])
    speed = float(np.hypot(vx, vy))
    if speed < 30.0 or abs(omega) < 1e-4:
        state[0] += vx * dt
        state[1] += vy * dt
        state[2] += vz * dt
        return
    theta = float(np.arctan2(vy, vx))
    nxt = theta + omega * dt
    state[0] += speed / omega * (np.sin(nxt) - np.sin(theta))
    state[1] += speed / omega * (-np.cos(nxt) + np.cos(theta))
    state[2] += vz * dt
    state[3] = speed * np.cos(nxt)
    state[4] = speed * np.sin(nxt)


def _along_road(state: np.ndarray, innovation: np.ndarray) -> np.ndarray:
    """Оставить продольный сдвиг и ослабить прыжок поперёк пройденного пути."""
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
