import numpy as np
from scipy.linalg import solve_continuous_are, eigvals

def linearize_attitude_rate(J):
    J_inv = np.linalg.inv(J)
    A = np.zeros((6, 6))
    A[0, 3] = 1.0 # phi_dot = p
    A[1, 4] = 1.0 # theta_dot =q
    A[2, 5] = 1.0 # psi_dot = r
    B = np.zeros((6, 3))
    B[3:6, 0:3] = J_inv  # omega_dot = J_inv @ M
    return A, B

def lqr_gain(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K, P


def stability_report(A, B=None, K=None, label=""):
    if K is not None:
        A_cl = A - B @ K
        eigs = eigvals(A_cl)
        tag = f"CLOSED LOOP [{label}]"
    else:
        eigs = eigvals(A)
        tag = f"OPEN LOOP [{label}]"

    stable = np.all(eigs.real < -1e-9)
    marginal = np.any(np.abs(eigs.real) < 1e-9) and not stable
    print(f"\n--- {tag} ---")
    for e in eigs:
        print(f" eigenvalue: {e.real:+.4f} {'+' if e.imag >= 0 else '-'} {abs(e.imag):.4f}j")
    if stable:
        print(" STABLE (all Re(eig) < 0)")
    elif marginal :
        print(" MARGINALLY STABLE / UNCONTROLLED MODE (Re(eig) == 0)")
    else :
        print(" UNSTABLE (at least one Re(eig) > 0)")
    return eigs, stable

def linearize_position_velocity(mass, g=9.81):
    A = np.zeros((6, 6))
    A[0, 3] = 1.0
    A[1, 4] = 1.0
    A[2, 5] = 1.0
    B = np.zeros((6, 3))
    B[3, 0] = -g
    B[4, 1] = g
    B[5, 2] = -1.0/mass
    return A,B

if __name__ == "__main__":
    J = np.diag([0.0347, 0.0459, 0.0977])
    A, B = linearize_attitude_rate(J)

    print("### Open-loop analysis: attitude+rate plant ###")
    stability_report(A, label="attitude-rate")

    Q = np.diag([50, 50, 20, 5, 5, 2]) # penalize angle error more than rate
    R = np.diag([1.0, 1.0, 1.0])
    K, P = lqr_gain(A, B, Q, R)
    print("\nLQR gain K (attitude-rate):\n", np.round(K, 3))
    stability_report(A, B, K, label="Attitude-rate")

    print("\n### Position/Velocity outer loop ###")
    Ap, Bp = linearize_position_velocity(mass=1.2)
    stability_report(Ap, label="position-velocity")
    Qp = np.diag([10, 10, 20, 1, 1, 2])
    Rp = np.diag([1.0, 1.0, 1.0])
    Kp, Pp = lqr_gain(Ap, Bp, Qp, Rp)
    print("\nLQR gain K (position-velocity):\n", np.round(Kp, 3))
    stability_report(Ap, Bp, Kp, label="position-velocity")