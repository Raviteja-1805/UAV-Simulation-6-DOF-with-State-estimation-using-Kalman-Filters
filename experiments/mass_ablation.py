import numpy as np
from sim.simulate import run_sim, make_vehicle
from control.pid_cascade import CascadedPIDController
from control.lqr_controller import LQRCascadedController

J = np.diag([0.0347, 0.0459, 0.0977])
target = np.array([5.0, 3.0, -10.0])

for mass_true in [1.2, 1.38, 1.02]:
    for name, ctrl in [("PID", CascadedPIDController(mass=1.2)),
                       ("LQR", LQRCascadedController(mass=1.2, J=J))]:
        log = run_sim(target, yaw_setpoint=0.5, sim_time=15.0, controller=ctrl,
                      params=make_vehicle(mass_true, J))
        e = log['pos'][-1] - target
        print(f"true mass {mass_true}: {name} final error (N,E,D) = {np.round(e, 2)}")

