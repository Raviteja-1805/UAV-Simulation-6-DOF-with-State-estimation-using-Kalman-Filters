import numpy as np
from control.linearize_lqr import ( linearize_attitude_rate, linearize_position_velocity, lqr_gain)


class LQIController:
    def __init__(self, mass, J, g=9.81, max_tilt_rad=0.5, q_int=(0.05, 0.05, 0.3), int_limit=(4.0, 4.0, 15.0)):

        self.m = mass
        self.g = g
        self.max_tilt = max_tilt_rad
        self.int_limit = np.array(int_limit, dtype=float)

        Ap, Bp = linearize_position_velocity(mass, g)
        A_aug = np.zeros((9, 9))
        A_aug[0:6, 0:6] = Ap
        A_aug[6:9, 0:3] = np.eye(3)
        B_aug = np.zeros((9, 3))
        B_aug[0:6, :] = Bp
        Q_aug = np.diag([1, 1, 2, 0.5, 0.5, 1, *q_int])
        R_aug = np.diag([5.0, 5.0, 5.0])
        self.K, _ = lqr_gain(A_aug, B_aug, Q_aug, R_aug)
        self.A_aug, self.B_aug = A_aug, B_aug

        Aa, Ba = linearize_attitude_rate(J)
        Qa = np.diag([50, 50, 20, 5, 5, 2])
        Ra = np.diag([1.0, 1.0, 1.0])
        self.Ka, _ = lqr_gain(Aa, Ba, Qa, Ra)

        self.z = np.zeros(3)

    def reset(self):
        self.z = np.zeros(3)

    def update(self, pos, vel, euler, omega, pos_setpoint, yaw_setpoint, dt):
        pos_err = pos - pos_setpoint
        x_aug = np.concatenate([pos_err, vel, self.z])
        u_outer = -self.K @ x_aug

        aN_des = -self.g * u_outer[0]
        aE_des = self.g * u_outer[1]
        c, s = np.cos(euler[2]), np.sin(euler[2])
        ax_body = c*aN_des + s*aE_des
        ay_body = c*aE_des - s*aN_des
        theta_cmd = -ax_body / self.g
        phi_cmd = ay_body / self.g

        tilt_mag = np.hypot(theta_cmd, phi_cmd)
        tilt_saturated = tilt_mag > self.max_tilt
        if tilt_saturated:
            scale = self.max_tilt / tilt_mag
            theta_cmd *= scale
            phi_cmd *= scale

        thrust_vertical_des = self.m * self.g + u_outer[2]
        cos_tilt = max(np.cos(theta_cmd) * np.cos(phi_cmd), 0.3)
        thrust_cmd = max(thrust_vertical_des / cos_tilt, 0.0)

        dz = pos_err.copy()
        if tilt_saturated:
            dz[0:2] = 0.0
        self.z = np.clip(self.z + dz*dt, -self.int_limit, self.int_limit)

        att_setpoint = np.array([phi_cmd, theta_cmd, yaw_setpoint])
        att_error = euler - att_setpoint
        att_error[2] = np.arctan2(np.sin(att_error[2]), np.cos(att_error[2]))
        moments_cmd = -self.Ka @ np.concatenate([att_error, omega])
        return thrust_cmd, moments_cmd
