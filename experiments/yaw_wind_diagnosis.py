import numpy as np
from sim.simulate import run_sim, make_vehicle
from sim.analysis import make_wind
from control.lqr_controller import LQRCascadedController
from control.pid_cascade import CascadedPIDController

mass = 1.2
J = np.diag([0.0347, 0.0459, 0.0977])
target = np.array([5.0, 3.0, -10.0])


def err_at(log, times, dt=0.002):
    out = []
    for t in times:
        i = min(int(t / dt), len(log['t']) - 1)
        out.append(np.linalg.norm(log['pos'][i] - target))
    return out


cases = [
    ("A: no wind, yaw 0.0", 0.0, False),
    ("B: no wind, yaw 0.5", 0.5, False),
    ("C: wind,    yaw 0.0", 0.0, True),
    ("D: wind,    yaw 0.5", 0.5, True),
]
controllers = [
    ("PID", lambda: CascadedPIDController(mass=mass)),
    ("LQR", lambda: LQRCascadedController(mass=mass, J=J)),
]

for name, yaw, windy in cases:
    for ctrl_name, make_ctrl in controllers:
        wind = make_wind(seed=0) if windy else None
        log = run_sim(target, yaw_setpoint=yaw, sim_time=15.0,
                      controller=make_ctrl(), params=make_vehicle(mass, J),
                      wind_body=wind)
        e = err_at(log, [5, 10, 14.9])
        print(f"{name} | {ctrl_name}: error at t=5,10,15 s -> "
              f"{e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f} m")
