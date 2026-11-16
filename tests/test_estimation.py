import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dynamics import quaternion as quat
from estimation.filters import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
from estimation.ins_models import (rot_zyx, ins_predict, ins_jacobian, ins_linear_model,
                                   mag_h, mag_jac, process_noise, H_GPS, R_GPS)


rng = np.random.default_rng(1)

def numerical_jacobian(f, x, eps=1e-6):
    J = np.zeros((len(f(x)), len(x)))
    for i in range(len(x)):
        dx = np.zeros(len(x));
        dx[i] = eps
        J[:, i] = (f(x + dx) - f(x - dx)) / (2*eps)
    return J

def random_pd(n):
    A = rng.standard_normal((n, n))
    return A @ A.T + 0.5*np.eye(n)

def test_rot_zyx_matches_quaternion_rotation():
    for _ in range(5):
        att = rng.uniform(-0.8, 0.8, 3)
        R_q = quat.to_rotation_matrix(quat.from_euler(*att))
        assert np.allclose(rot_zyx(*att), R_q, atol=1e-12)


def test_process_jacobian_matches_numerical():
    for _ in range(5):
        x = np.concatenate([rng.standard_normal(6), rng.uniform(-0.7, 0.7, 3)])
        acc, gyro = rng.standard_normal(3)*5, rng.standard_normal(3)
        F = ins_jacobian(x, acc, gyro, 0.01)
        Fn = numerical_jacobian(lambda s: ins_predict(s, acc, gyro, 0.01), x)
        assert np.allclose(F, Fn, atol=1e-7)

def test_mag_jacobian_matches_numerical():
    for _ in range(5):
        x = np.concatenate([np.zeros(6), rng.uniform(-0.7, 0.7, 3)])
        assert np.allclose(mag_jac(x), numerical_jacobian(mag_h, x), atol=1e-7)


def test_linear_model_matches_nonlinear_at_small_angles():
    x = np.concatenate([rng.standard_normal(6), 1e-3*rng.standard_normal(3)])
    acc, gyro = np.array([0.3, -0.2, -9.81]), 0.1*rng.standard_normal(3)
    F, b = ins_linear_model(acc, gyro, 0.01)
    assert np.allclose(F @ x + b, ins_predict(x, acc, gyro, 0.01), atol=1e-6)


def test_gps_update_identical_for_kf_ekf_ukf():
    x0, P0 = rng.standard_normal(9), random_pd(9)
    z = rng.standard_normal(6)
    kf, ekf, ukf = KalmanFilter (x0, P0), ExtendedKalmanFilter (x0, P0), UnscentedKalmanFilter (x0, P0)
    kf.update(z, H_GPS, R_GPS)
    ekf.update_nonlinear(z, lambda x: H_GPS @ x, lambda x: H_GPS, R_GPS)
    ukf.update(z, lambda x: H_GPS @ x, R_GPS)
    assert np.allclose(kf.x, ekf.x) and np.allclose(kf.x, ukf.x, atol=1e-9)
    assert np.allclose(kf.P, ekf.P) and np.allclose(kf.P, ukf.P, atol=1e-9)


def test_ukf_predict_is_exact_for_linear_model():
    x0, P0 = rng.standard_normal(9), random_pd(9)
    F, b, Q = np.eye(9) + 0.1*rng.standard_normal((9, 9)), rng.standard_normal(9), 0.001*np.eye(9)
    kf, ukf = KalmanFilter(x0, P0), UnscentedKalmanFilter(x0, P0)
    kf.predict(F, Q, b)
    ukf.predict(lambda x: F @ x + b, Q)
    assert np.allclose(kf.x, ukf.x, atol=1e-9) and np.allclose(kf.P, ukf.P, atol=1e-9)


def test_update_reduces_covariance_and_keeps_it_symmetric():
    x0, P0 = np.zeros(9), random_pd(9)
    kf = KalmanFilter (x0, P0)
    kf.update(np.ones(6), H_GPS, R_GPS)
    assert np.trace(kf.P) < np.trace(P0)
    assert np.allclose(kf.P, kf.P.T)
    assert np.all(np.linalg.eigvalsh(kf.P) > 0)