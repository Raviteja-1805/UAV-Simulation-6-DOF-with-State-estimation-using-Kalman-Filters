import numpy as np
from dynamics import quaternion as quat

class IMU:
    def __init__(self, accel_noise_std=0.05, gyro_noise_std=0.002,
                 accel_bias_std=0.001, gyro_bias_std=0.0001, dt= 0.01, seed=None):
        self.rng = np.random.default_rng(seed)
        self.accel_noise_std = accel_noise_std  # m/s^2
        self.gyro_noise_std = gyro_noise_std  # rad/s
        self.accel_bias_walk = accel_bias_std # m/s^2 per sqrt(s) random walk
        self.gyro_bias_walk = gyro_bias_std
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.dt = dt

    def measure(self, specific_force_body, omega_body):
        self.accel_bias += self.accel_bias_walk * np.sqrt(self.dt) * self.rng.standard_normal()
        self.gyro_bias += self.gyro_bias_walk * np.sqrt(self.dt) * self.rng.standard_normal(3)
        accel_meas = (specific_force_body + self.accel_bias
                      + self.accel_noise_std * self.rng.standard_normal(3))
        gyro_meas = (omega_body + self.gyro_bias
                     + self.gyro_noise_std * self.rng.standard_normal(3))
        return accel_meas, gyro_meas

class GPS:
    def __init__(self, pos_noise_std=1.5, vel_noise_std=0.1, update_rate_hz=5.0,
                 dropout_prob=0.0, seed=None):
        self.rng = np.random.default_rng(seed)
        self.pos_noise_std = pos_noise_std  # m
        self.vel_noise_std = vel_noise_std  # m/s
        self.period = 1.0/ update_rate_hz
        self._t_last = -1e9
        self.dropout_prob = dropout_prob
        self.last_measurement = None


    def measure(self, t, pos_true, vel_true):
        if t - self._t_last < self.period:
            return None
        self._t_last = t
        if self.rng.random() < self.dropout_prob:
            return None
        pos_meas = pos_true + self.pos_noise_std * self.rng.standard_normal(3)
        vel_meas = vel_true + self.vel_noise_std * self.rng.standard_normal(3)
        self.last_measurement = (pos_meas, vel_meas)
        return pos_meas, vel_meas


class Barometer:
    def __init__(self, noise_std=0.3, bias_walk_std=0.01, update_rate_hz=20.0, seed=None):
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std
        self.bias_walk_std = bias_walk_std
        self.bias = 0.0
        self.period = 1.0 / update_rate_hz
        self._t_last = -1e9

    def measure(self, t, altitude_true, dt):
        if t - self._t_last < self.period:
            return None
        self._t_last = t
        self.bias += self.bias_walk_std * np.sqrt(dt) * self.rng.standard_normal()
        return altitude_true + self.bias + self.noise_std * self.rng.standard_normal()


class Magnetometer:
    def __init__(self, noise_std_rad=0.02, hard_iron_bias=None,
                 update_rate_hz=50.0, seed=None):
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std_rad
        self.hard_iron = hard_iron_bias if hard_iron_bias is not None else np.zeros(3)
        self.mag_inertial = np.array([1.0, 0.0, 0.0]) # simplified local nort-pointing field
        self.period = 1.0 / update_rate_hz
        self._t_last = -1e9

    def measure(self, t, q_body_to_inertial):
        if t - self._t_last < self.period:
            return None
        self._t_last = t
        R = quat.to_rotation_matrix(q_body_to_inertial)
        mag_body_true = R.T @ self.mag_inertial
        mag_body_meas = mag_body_true + self.hard_iron + self.noise_std * self.rng.standard_normal(3)
        return mag_body_meas / np.linalg.norm(mag_body_meas)