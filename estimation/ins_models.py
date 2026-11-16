import numpy as np

G_VEC = np.array([0.0, 0.0, 9.81])

SIGMA_GPS_POS, SIGMA_GPS_VEL = 1.5, 0.1
SIGMA_BARO, SIGMA_MAG = 0.3, 0.03

H_GPS = np.hstack([np.eye(6), np.zeros((6, 3))])
R_GPS = np.diag([SIGMA_GPS_POS**2]*3 + [SIGMA_GPS_VEL**2]*3)
H_BARO = np.zeros((1, 9)); H_BARO[0, 2] = -1.0
R_BARO = np.array([[SIGMA_BARO**2]])
R_MAG = SIGMA_MAG**2 * np.eye(3)
H_MAG_LIN = np.zeros((2, 9))
H_MAG_LIN[0, 8] = -1.0
H_MAG_LIN[1, 7] = 1.0

def _axis_rotations(phi, theta, psi):
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    dRx = np.array([[0, 0, 0], [0, -sp, -cp], [0, cp, -sp]])
    dRy = np.array([[-st, 0, ct], [0, 0, 0], [-ct, 0, -st]])
    dRz = np.array([[-sy, -cy, 0], [cy, -sy, 0], [0, 0, 0]])
    return Rx, Ry, Rz, dRx, dRy, dRz


def rot_zyx(phi, theta, psi):
    Rx, Ry, Rz, _, _, _, = _axis_rotations(phi, theta, psi)
    return Rz @ Ry @ Rx

def d_rot_zyx(att):
    Rx, Ry, Rz, dRx, dRy, dRz = _axis_rotations(*att)
    return [Rz @ Ry @ dRx,  Rz @ dRy @ Rx, dRz @ Ry @ Rx ]

def euler_rate(att, w):
    phi, theta, _ = att
    p, q, r = w
    sp, cp, ct, tt = np.sin(phi), np.cos(phi), np.cos(theta), np.tan(theta)
    return np.array([p + (q*sp + r*cp)*tt, q*cp - r*sp, (q*sp + r*cp)/ct])

def ins_predict(x, acc, gyro, dt):
    p, v, att = x[0:3], x[3:6], x[6:9]
    a = rot_zyx(*att) @ acc + G_VEC
    return np.concatenate([p + v*dt + 0.5*a*dt**2, v + a*dt, att + euler_rate(att, gyro)*dt])

def ins_jacobian(x, acc, gyro, dt):
    att = x[6:9]
    da = np.column_stack([dR @ acc for dR in d_rot_zyx(att)])
    phi, theta = att[0], att[1]
    p, q, r = gyro
    sp, cp, st, ct = np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta)
    tt = st / ct
    dE = np.array([
        [(q*cp - r*sp)*tt, (q*sp + r*cp)/ct**2,  0.0],
        [-q*sp - r*cp,  0.0,  0.0],
        [(q*cp - r*sp)/ct, (q*sp + r*cp)*st/ct**2, 0.0]])
    F = np.eye(9)
    F[0:3, 3:6] = dt*np.eye(3)
    F[0:3, 6:9] = 0.5*dt**2*da
    F[3:6, 6:9] = dt*da
    F[6:9, 6:9] += dt*dE
    return F

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def ins_linear_model(acc, gyro, dt):
    S = skew(acc)
    F = np.eye(9)
    F[0:3, 3:6] = dt*np.eye(3)
    F[0:3, 6:9] = -0.5*dt**2*S
    F[3:6, 6:9] = -dt*S
    a0 = acc + G_VEC
    b = np.concatenate([0.5*dt**2*a0, dt*a0, dt*gyro])
    return F, b

def process_noise(dt, sigma_a, sigma_g):
    Q = np.zeros((9, 9))
    I = np.eye(3)
    Q[0:3, 0:3] = 0.25*dt**4*sigma_a**2*I
    Q[0:3, 3:6] = Q[3:6, 0:3] = 0.5*dt**3*sigma_a**2*I
    Q[3:6, 3:6] = dt**2*sigma_a**2*I
    Q[6:9, 6:9] = (sigma_g*dt)**2*I
    return Q

def mag_h(x):
    return rot_zyx(*x[6:9])[0, :]

def mag_jac(x):
    H = np.zeros((3, 9))
    for j, dR in enumerate(d_rot_zyx(x[6:9])):
        H[:, 6 + j] = dR[0, :]
    return H