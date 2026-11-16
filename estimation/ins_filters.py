import numpy as np
from estimation.filters import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
from estimation.ins_models import(
    ins_predict,ins_jacobian, ins_linear_model, process_noise, mag_h, mag_jac,
    H_GPS, R_GPS, H_BARO, R_BARO, R_MAG, H_MAG_LIN)


class _INSFilter:
    def __init__(self, x0, P0, dt, sigma_a, sigma_g):
        self.dt = dt
        self.Q = process_noise(dt, sigma_a, sigma_g)

    @property
    def x(self):
        return self.core.x

    @property
    def P(self):
        return self.core.P

class INSKalmanFilter(_INSFilter):
    def __init__(self, x0, P0, dt, sigma_a, sigma_g):
        super().__init__(x0, P0, dt, sigma_a, sigma_g)
        self.core = KalmanFilter(x0, P0)

    def predict(self, acc, gyro):
        F, b = ins_linear_model(acc, gyro, self.dt)
        self.core.predict(F, self.Q, b)


    def update_gps(self, z):
        self.core.update(z, H_GPS, R_GPS)

    def update_baro(self, z):
        self.core.update(np.array([z]), H_BARO, R_BARO)


    def update_mag(self, z):
        self.core.update(z[1:3], H_MAG_LIN, R_MAG[1:3, 1:3])


class INSExtendedKalmanFilter(_INSFilter):
    def __init__(self, x0, P0, dt, sigma_a, sigma_g):
        super().__init__(x0, P0, dt, sigma_a, sigma_g)
        self.core = ExtendedKalmanFilter(x0, P0)

    def predict(self, acc, gyro):
        dt = self.dt
        self.core.predict_nonlinear(lambda x: ins_predict(x, acc, gyro, dt),
                                    lambda x: ins_jacobian(x, acc, gyro, dt), self.Q)


    def update_gps(self, z):
        self.core.update(z, H_GPS, R_GPS)

    def update_baro(self, z):
        self.core.update(np.array([z]), H_BARO, R_BARO)

    def update_mag(self, z):
        self.core.update_nonlinear(z, mag_h, mag_jac, R_MAG)


class INSUnscentedKalmanFilter(_INSFilter):
    def __init__(self, x0, P0, dt, sigma_a, sigma_g):
        super().__init__(x0, P0, dt, sigma_a, sigma_g)
        self.core = UnscentedKalmanFilter(x0, P0)

    def predict(self, acc, gyro):
        dt = self.dt
        self.core.predict(lambda x: ins_predict(x, acc, gyro, dt), self.Q)

    def update_gps(self, z):
        self.core.update(z, lambda x: H_GPS @ x, R_GPS)

    def update_baro(self, z):
        self.core.update(np.array([z]), lambda x: H_BARO @ x, R_BARO)

    def update_mag(self, z):
        self.core.update(z, mag_h, R_MAG)

