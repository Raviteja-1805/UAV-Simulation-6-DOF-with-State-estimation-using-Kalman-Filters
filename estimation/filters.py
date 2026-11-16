import numpy as np

class KalmanFilter:
    def __init__(self, x0, P0):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)

    def predict(self, F, Q, b=None):
        self.x = F @ self.x + (0.0 if b is None else b)
        self.P = F @ self.P @ F.T + Q

    def _correct(self, y, H, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ H
        self.P= I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def update(self, z, H, R):
        self._correct(z - H @ self.x, H, R)


class ExtendedKalmanFilter(KalmanFilter):
    def predict_nonlinear(self, f, F_jac, Q):
        F = F_jac(self.x)
        self.x = f(self.x)
        self.P = F @ self.P @ F.T + Q

    def update_nonlinear(self, z, h, H_jac, R):
        H = H_jac(self.x)
        self._correct(z - h(self.x), H, R)


class UnscentedKalmanFilter:
    def __init__(self, x0, P0, alpha=1.0, beta=2.0, kappa=0.0):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        n = self.n = len(self.x)
        self.lam = alpha**2 * (n + kappa) - n
        self.Wm = np.full(2*n + 1, 1.0 / (2*(n + self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (n + self.lam)
        self.Wc[0] = self.lam / (n + self.lam) + (1 - alpha**2 + beta)


    def _sigma_points(self):
        n = self.n
        P = 0.5 * (self.P + self.P.T)
        try:
            S = np.linalg.cholesky((n + self.lam) * P)
        except np.linalg.LinAlgError:
            S = np.linalg.cholesky((n + self.lam) * (P + 1e-9*np.eye(n)))
        pts = np.empty((2*n + 1, n))
        pts[0] = self.x
        for i in range(n):
            pts[1 + i] = self.x + S[:, i]
            pts[1 + n + i] = self.x - S[:, i]
        return pts

    def predict(self, f, Q):
        pts = np.array([f(p) for p in self._sigma_points()])
        self.x = self.Wm @ pts
        d = pts - self.x
        self.P = (self.Wc[:, None] * d).T @ d + Q


    def update(self, z, h, R):
        pts = self._sigma_points()
        zs = np.array([h(p) for p in pts])
        z_mean = self.Wm @ zs
        dz = zs - z_mean
        dx = pts - self.x
        S = (self.Wc[:, None] * dz).T @ dz + R
        Pxz = (self.Wc[:, None] * dx).T @ dz
        K = Pxz @ np.linalg.inv(S)
        self.x = self.x + K @ (z - z_mean)
        P = self.P - K @ S @ K.T
        self.P = 0.5 * (P + P.T)