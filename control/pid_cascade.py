import numpy as np

class PID:
    def __init__(self, kp, ki, kd, out_min=-np.inf, out_max=np.inf, integral_limit=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, error, dt):
        deriv = 0.0 if self.prev_error is None else (error - self.prev_error) /dt
        self.prev_error = error

        u_unsat = self.kp*error + self.ki*self.integral + self.kd*deriv
        u = np.clip(u_unsat, self.out_min, self.out_max)

        saturated_high = u_unsat > self.out_max
        saturated_low = u_unsat < self.out_min
        if not ((saturated_high and error > 0) or (saturated_low and error < 0)):
            self.integral += error * dt
            if self.integral_limit is not None:
                self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        return u


class CascadedPIDController:
        def __init__(self, mass, g=9.81, max_tilt_rad=0.5):
                    self.m = mass
                    self.g = g
                    self.max_tilt = max_tilt_rad
        
                    self.pos_pid = [PID(kp=1.2, ki=0.0, kd=0.3, out_min=-5, out_max=5) for _ in range(3)]
        
                    self.vel_pid = [PID(kp=2.0, ki=0.5, kd=0.05, out_min=-6, out_max=6, integral_limit=3 ) for _ in range(3)]
        
                    self.att_pid = [PID(kp=6.0, ki= 0.0, kd=0.3, out_min=-6, out_max=6) for _ in range(3)]
        
                    self.rate_pid = [PID(kp=0.15, ki=0.05, kd=0.003, out_min=-4, out_max=4, integral_limit=1) for _ in range(3)]
        
        
        def reset(self):
            for group in (self.pos_pid, self.vel_pid, self.att_pid, self.rate_pid):
                for pid in group:
                    pid.reset()

        def update(self, pos, vel, euler, omega, pos_setpoint, yaw_setpoint, dt):
            vel_sp = np.array([self.pos_pid[i].update(pos_setpoint[i] - pos[i], dt) for i in range(3)])
            acc_des = np.array([self.vel_pid[i].update(vel_sp[i] - vel[i], dt) for i in range(3)])

            c, s = np.cos(euler[2]), np.sin(euler[2])
            ax_body = c*acc_des[0] + s*acc_des[1]
            ay_body = -s*acc_des[0] + c*acc_des[1]
            theta_cmd = -ax_body / self.g
            phi_cmd = ay_body / self.g

            tilt_mag = np.hypot(theta_cmd, phi_cmd)
            if tilt_mag > self.max_tilt:
                scale = self.max_tilt / tilt_mag
                theta_cmd *= scale
                phi_cmd *= scale
            psi_cmd = yaw_setpoint

            thrust_vertical_des = self.m * (self.g - acc_des[2])
            cos_tilt = max(np.cos(theta_cmd) * np.cos(phi_cmd), 0.3)
            thrust_cmd = max(thrust_vertical_des / cos_tilt, 0.0)
        
            att_setpoint = np.array([phi_cmd, theta_cmd, psi_cmd])
      
            att_error = att_setpoint - euler
            att_error[2] = np.arctan2(np.sin(att_error[2]), np.cos(att_error[2]))
            rate_sp = np.array([self.att_pid[i].update(att_error[i], dt) for i in range(3)])


            moments_cmd = np.array([self.rate_pid[i].update(rate_sp[i] - omega[i], dt)for i in range(3)])

            return thrust_cmd, moments_cmd
