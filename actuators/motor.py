import numpy as np


class MotorParams:
    def __init__(self, k_thrust=1.5e-5, k_torque=2.4e-7, tau=0.02, 
                 omega_min=0.0, omega_max=900.0, max_rate=4000.0):
        self.k_thrust = k_thrust  # N / (rad/s)^2
        self.k_torque = k_torque  # N*m / (rad/s)^2
        self.tau = tau            # motor time constant (s)
        self.omega_min = omega_min # rad/s
        self.omega_max = omega_max # rad/s (saturation limit)
        self.max_rate = max_rate   # rad/s^2 (max allowed d(omega_cmd) / dt)


class Motor:
    def __init__(self, params:MotorParams, omega0=0.0):
        self.p = params
        self.omega = omega0  # Actual Control Speed
        self._omega_cmd_prev = omega0 # for rate limitting

    def step(self, omega_cmd, dt):
        d_cmd = omega_cmd - self._omega_cmd_prev
        max_d = self.p.max_rate * dt
        d_cmd = np.clip(d_cmd, -max_d, max_d)
        omega_cmd_limited = self._omega_cmd_prev + d_cmd
        self._omega_cmd_prev = omega_cmd_limited

        omega_cmd_sat = np.clip(omega_cmd_limited, self.p.omega_min, self.p.omega_max)

        self.omega += dt * (omega_cmd_sat - self.omega) / self.p.tau
        self.omega = np.clip(self.omega, self.p.omega_min, self.p.omega_max)
        return self.omega

    def thrust(self):
        return self.p.k_thrust * self.omega**2

    def reaction_torque(self):
        return self.p.k_torque * self.omega**2

class QuadMixer:
    def __init__(self, motor_params: MotorParams, arm_length=0.25):
        self.p = motor_params
        self.L = arm_length
        k_t, k_q = motor_params.k_thrust, motor_params.k_torque
        L = arm_length
        self.mix_matrix = np.array([
            [1.0,  1.0,  1.0,  1.0],      # total thrust
            [0.0,  L,   0.0,   -L],       # roll moment (right - left)
            [L,  0.0,  -L,  0.0],         # pitch moment (front - back)
            [-k_q/k_t, k_q/k_t, -k_q/k_t, k_q/k_t], # yaw moment from tc
        ])
        self.mix_matrix_inv = np.linalg.inv(self.mix_matrix)
        self.motors = [Motor(motor_params) for _ in range(4)]

    def command(self, thrust_des, moments_des, dt):
        tm_vec = np.concatenate([[thrust_des], moments_des])
        f_desired = self.mix_matrix_inv @ tm_vec
        f_desired = np.clip (f_desired, 0.0, None)
        omega_cmd = np.sqrt(f_desired / self.p.k_thrust)

        for m, w_cmd in zip(self.motors, omega_cmd):
            m.step(w_cmd, dt)

        f_actual = np.array([m.thrust() for m in self.motors])
        tm_actual = self.mix_matrix @ f_actual
        thrust_actual = tm_actual[0]
        moments_actual = tm_actual[1:4]
        return thrust_actual, moments_actual