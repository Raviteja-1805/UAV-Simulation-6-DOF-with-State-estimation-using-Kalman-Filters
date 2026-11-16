import numpy as np

def normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Cannot normalize a near-zero quaternion")
    return q / n

def conjugate(q):
    q0, q1, q2, q3 = q
    return np.array([q0, -q1, -q2, -q3])

def multiply(q, r):
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 + x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ])

def to_rotation_matrix(q):
    q = normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z),  1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),  2*(y*z + w*x), 1 - 2*(x**2 + y**2)],
    ])

def from_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])

def to_euler(q):
    w, x, y, z = normalize(q)
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    sinp = 2*(w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return np.array([roll, pitch, yaw])

def derivative(q, omega_body):
    wx, wy, wz = omega_body
    omega_quat = np.array([0.0, wx, wy, wz])
    return 0.5 * multiply(q, omega_quat)

def rotate_vector(q, v):
    return to_rotation_matrix(q) @ np.asarray(v, dtype=float)