import numpy as np
from control.linearize_lqr import(linearize_attitude_rate, linearize_position_velocity, lqr_gain)

class LQRCascadedController:
    def __init__(self, mass, J, g=9.81, max_tilt_rad=0.5):
        self.m = mass
        self.g = g
        self.max_tilt = max_tilt_rad

        Ap, Bp = linearize_position_velocity(mass, g)
        Qp = np.diag([1, 1, 2, 0.5, 0.5, 1])
        Rp = np.diag([5.0, 5.0, 5.0])
        self.Kp, _ = lqr_gain(Ap, Bp, Qp, Rp)


        Aa, Ba = linearize_attitude_rate(J)
        Qa = np.diag([50, 50, 20, 5, 5, 2])
        Ra = np.diag([1.0, 1.0, 1.0])
        self.Ka, _ = lqr_gain(Aa, Ba, Qa, Ra)





    def reset(self):
        pass  # LQR here has no integrator state

    def update(self, pos, vel, euler, omega, pos_setpoint, yaw_setpoint, dt):
        x_outer = np.concatenate([pos - pos_setpoint, vel])  # error state
        u_outer = -self.Kp @ x_outer  # [theta_cmd, phi_cmd, T_offset]
        aN_des = -self.g * u_outer[0]
        aE_des = self.g * u_outer[1]
        c, s = np.cos(euler[2]), np.sin(euler[2])
        ax_body = c*aN_des + s*aE_des
        ay_body = -s*aN_des + c*aE_des
        theta_cmd = -ax_body / self.g
        phi_cmd = ay_body / self.g

        tilt_mag = np.hypot(theta_cmd, phi_cmd)
        if tilt_mag > self.max_tilt:
            scale = self.max_tilt / tilt_mag
            theta_cmd *= scale
            phi_cmd *= scale
        psi_cmd = yaw_setpoint

        # thrust = hover thrust + LQR correction, divided by cos(tilt)
        thrust_vertical_des = self.m * self.g + u_outer[2]
        cos_tilt = max(np.cos(theta_cmd) * np.cos(phi_cmd), 0.3)
        thrust_cmd = max(thrust_vertical_des / cos_tilt, 0.0)

        att_setpoint = np.array([phi_cmd, theta_cmd, psi_cmd])
        att_error = euler - att_setpoint
        att_error[2] = np.arctan2(np.sin(att_error[2]), np.cos(att_error[2]))
        x_inner = np.concatenate([att_error, omega])
        moments_cmd = -self.Ka @ x_inner

        return thrust_cmd, moments_cmd