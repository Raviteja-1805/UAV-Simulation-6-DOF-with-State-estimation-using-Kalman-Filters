import numpy as np 
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamics import quaternion as quat
from dynamics.rigid_body import (
    RigidBodyParams, state_derivative, rk4_step, gravity_force_body
)

TOL = 1e-8

def test_quaternion_identity():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    v = np.array([1.0, 2.0, 3.0])
    v_rot = quat.rotate_vector(q, v)
    assert np.allclose(v_rot, v, atol=TOL)

def test_quaternion_normalize():
    q = np.array([2.0, 0.0, 0.0, 0.0])
    qn = quat.normalize(q)
    assert abs(np.linalg.norm(qn) - 1.0) < TOL


def test_quaternion_euler_roundtrip():
    for roll, pitch, yaw in [(0.1, 0.2, 0.3), (-0.5, 0.4, 1.0), (0, 0, 0)]:
        q = quat.from_euler(roll, pitch, yaw)
        rpy = quat.to_euler(q)
        assert np.allclose(rpy, [roll, pitch, yaw], atol=1e-6)

def test_quaternion_90deg_yaw_rotates_x_to_y():
    q = quat.from_euler(0, 0, np.pi/2)
    v = np.array([1.0, 0.0, 0.0])
    v_rot = quat.rotate_vector(q, v)
    assert np.allclose(v_rot, [0.0, 1.0, 0.0], atol=1e-6)

def test_rotation_matrix_is_orthonormal():
    q = quat.from_euler(0.3, -0.6, 1.2)
    R = quat.to_rotation_matrix(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
    assert abs(np.linalg.det(R) - 1.0) < 1e-8

def test_quaternion_multiply_matches_matrix_composition():
    q1 = quat.from_euler(0.2, 0.0, 0.0)
    q2 = quat.from_euler(0.0, 0.3, 0.0)
    q_combined = quat.multiply(q2, q1)
    R1 = quat.to_rotation_matrix(q1)
    R2 = quat.to_rotation_matrix(q2)
    R_combined_expected = R2 @ R1
    R_combined_actual = quat.to_rotation_matrix(q_combined)
    assert np.allclose(R_combined_actual, R_combined_expected, atol=1e-6)

def test_torque_free_symmetric_body_conserves_angular_momentum_magnitude():
    J = np.diag([0.02, 0.03, 0.04])
    params = RigidBodyParams(mass=1.0, inertia_matrix=J)

    state = np.zeros(13)
    state[6] = 1.0
    state[10:13] = [0.5, 1.0, -0.3]

    dt = 0.001
    H_mags = []
    for _ in range(2000):
        forces = np.zeros(3)
        moments = np.zeros(3)
        q = state[6:10]
        omega = state[10:13]
        R = quat.to_rotation_matrix(q)
        H_inertial = R @ (J @ omega)
        H_mags.append(np.linalg.norm(H_inertial))
        state = rk4_step(state, dt, forces, moments, params)

    H_mags = np.array(H_mags)
    drift = (H_mags.max() - H_mags.min()) / H_mags[0]
    assert drift < 1e-4, f"Angular Momentum magnitude drifted by {drift:.2e}"


def test_quaternion_stays_normalized_after_many_integration_steps():
    J = np.diag([0.02, 0.03, 0.04])
    params = RigidBodyParams(mass=1.0, inertia_matrix=J)
    state = np.zeros(13)
    state[6]= 1.0
    state[10:13] = [2.0, -1.5, 0.7]

    dt = 0.002
    for _ in range(5000):
        state = rk4_step(state, dt, np.zeros(3), np.zeros(3), params)
        norm = np.linalg.norm(state[6:10])
        assert abs(norm - 1.0) < 1e-6

def test_free_fall_under_gravity():
    J = np.diag([0.02, 0.03, 0.04])
    params = RigidBodyParams(mass=1.0, inertia_matrix=J)
    state = np.zeros(13)
    state[6] = 1.0

    dt = 0.001
    g = 9.81
    t = 0.0
    for _ in range(1000):
        q = state[6:10]
        gravity = gravity_force_body(params.m, q, g=g)
        state = rk4_step(state, dt, gravity, np.zeros(3), params)
        t += dt

    vel_body = state[3:6]
    pos = state[0:3]
    assert abs(vel_body[2] - g*t) < 1e-2
    assert abs(pos[2] - 0.5*g*t**2) < 1e-2

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))


