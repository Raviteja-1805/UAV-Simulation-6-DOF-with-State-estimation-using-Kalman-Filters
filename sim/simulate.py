import numpy as np
from dynamics import quaternion as quat
from dynamics.rigid_body import RigidBodyParams, rk4_step, gravity_force_body
from actuators.motor import MotorParams, QuadMixer
from control.pid_cascade import CascadedPIDController

def make_vehicle(mass=1.2, J=None):
    if J is None:
        J = np.diag([0.0347, 0.0459, 0.0977])
    return RigidBodyParams(mass=mass, inertia_matrix=J)


def run_sim(pos_setpoint, yaw_setpoint=0.0, sim_time=8.0, dt=0.002,
            mass=1.2, J=None, wind_body=None, controller=None, params=None):
    params = params or make_vehicle(mass, J)
    controller = controller or CascadedPIDController(mass=params.m)
    mixer = QuadMixer(MotorParams())

    state = np.zeros(13)
    state[6] = 1.0

    n_steps = int(sim_time / dt)
    log = {k : np.zeros((n_steps, 3)) for k in ['pos', 'vel', 'euler', 'omega', 'specific_force']}
    log['t'] = np.zeros(n_steps)
    log['thrust'] = np.zeros(n_steps)

    for i in range(n_steps):
        t = i * dt
        pos = state[0:3]
        vel_b = state[3:6]
        q = state[6:10]
        omega = state[10:13]
        euler = quat.to_euler(q)
        R = quat.to_rotation_matrix(q)
        vel_i = R @ vel_b

        thrust_cmd, moments_cmd = controller.update(
            pos, vel_i, euler, omega, pos_setpoint, yaw_setpoint,dt)

        
        thrust_actual, moments_actual = mixer.command(thrust_cmd, moments_cmd, dt)

        forces_nongrav = np.array([0.0, 0.0, -thrust_actual]) 
        if wind_body is not None:
            forces_nongrav = forces_nongrav + wind_body(t)
        forces_body = forces_nongrav + gravity_force_body(params.m, q)
        log['specific_force'][i] = forces_nongrav / params.m

        state = rk4_step(state, dt, forces_body, moments_actual, params)

        log['t'][i] = t
        log['pos'][i]=pos
        log['vel'][i]=vel_i
        log['euler'][i]=euler
        log['omega'][i]=omega
        log['thrust'][i]=thrust_actual

    return log


if __name__ == "__main__":
    target = np.array([5.0, 3.0, -10.0])
    log = run_sim(pos_setpoint=target, yaw_setpoint=0.5, sim_time=8.0)
    print("Target:", target, "yaw=0.5")
    print("Final position:", np.round(log['euler'][-1, 2], 3))
    print(f"Final position error: {np.linalg.norm(log['pos'][-1] - target):.4f} m")

    