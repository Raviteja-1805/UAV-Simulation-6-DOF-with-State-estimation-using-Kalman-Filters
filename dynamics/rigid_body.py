import numpy as np
from . import quaternion as quat

class RigidBodyParams:
    def __init__(self, mass, inertia_matrix):
        self.m = mass
        self.J = np.asarray(inertia_matrix, dtype=float)
        self.J_inv = np.linalg.inv(self.J)

def state_derivative(state, forces_body, moments_body, params: RigidBodyParams):
    pos = state[0:3]
    vel_b = state[3:6]
    q = state[6:10]
    omega = state[10:13]

    q = quat.normalize(q)

    #Translational dynamics (body frame velocity)
    vel_dot = forces_body / params.m - np.cross(omega, vel_b)

    #Rotational dynamics (Euler's equation)
    omega_dot = params.J_inv @ (moments_body - np.cross(omega, params.J @ omega))

    #Kinematics
    pos_dot = quat.to_rotation_matrix(q) @ vel_b
    q_dot = quat.derivative(q, omega)

    return np.concatenate([pos_dot, vel_dot, q_dot, omega_dot])


def rk4_step(state, dt, forces_body, moments_body, params: RigidBodyParams):


    def f(s):
        return state_derivative(s, forces_body, moments_body, params)

    k1 = f(state)
    k2 = f(state + 0.5*dt*k1)
    k3 = f(state + 0.5*dt*k2)
    k4 = f(state + dt*k3)

    new_state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    new_state[6:10] = quat.normalize(new_state[6:10])
    return new_state

def gravity_force_body(mass, q, g=9.81):
    g_inertial = np.array([0.0, 0.0, mass*g])
    R = quat.to_rotation_matrix(q)
    return R.T @ g_inertial
