import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamics import quaternion as quat
from sim.simulate import run_sim, make_vehicle
from sim.analysis import make_wind, plot_ecdf
from control.pid_cascade import CascadedPIDController
from sensors.sensors import IMU, GPS, Barometer, Magnetometer
from estimation.ins_filters import(
    INSKalmanFilter, INSExtendedKalmanFilter, INSUnscentedKalmanFilter)

SIM_DT, EST_DT = 0.002, 0.01
TARGET = np.array([5.0, 3.0, -10.0])
FILTERS = {'KF' : INSKalmanFilter, 'EKF': INSExtendedKalmanFilter, 'UKF' : INSUnscentedKalmanFilter}
COLORS = {'KF' : 'tab:red', 'EKF' : 'tab:blue', 'UKF' : 'tab:green'}
LSTYLE = {'KF' : '-', 'EKF' : '-', 'UKF' : '--'}


SIGMA_A, SIGMA_G = 0.1, 0.005
INIT_ERR = np.array([0.5, -0.4, 0.3, 0.1, -0.1, 0.05, 0.03, -0.02, 0.05])
P0 = np.diag([1.0]*3 + [0.3**2]*3 + [0.1**2]*3)
T_SETTLE = 2.0

def fly():
    return run_sim(TARGET, yaw_setpoint=0.5, sim_time=15.0, dt=SIM_DT,
                   controller=CascadedPIDController(mass=1.2),
                   params=make_vehicle(1.2), wind_body=make_wind(seed=0))

def generate_sensor_data(log, seed):
    step = int(round(EST_DT / SIM_DT))
    idx = np.arange(0, len(log['t']), step)
    imu = IMU(dt=EST_DT, seed=seed)
    gps, baro, mag = GPS(seed=seed + 1), Barometer(seed=seed + 2), Magnetometer(seed=seed +3)
    data = {k: [] for k in ['acc', 'gyro', 'gps', 'baro', 'mag']}
    for i in idx:
        t = log['t'][i]
        a, g = imu.measure(log['specific_force'][i], log['omega'][i])
        data['acc'].append(a)
        data['gyro'].append(g)
        fix = gps.measure(t, log['pos'][i], log['vel'][i])
        data['gps'].append(None if fix is None else np.concatenate(fix))
        data['baro'].append(baro.measure(t, -log['pos'][i][2], EST_DT))
        data['mag'].append(mag.measure(t, quat.from_euler(*log['euler'][i])))
    data['t'] = log['t'][idx]
    data['truth']=np.hstack([log['pos'][idx], log['vel'][idx], log['euler'][idx]])
    return data


def run_filter(cls, data):
    x0 = data['truth'][0] + INIT_ERR
    filt = cls(x0, P0, EST_DT, SIGMA_A, SIGMA_G)
    n = len(data['t'])
    xs, Ps = np.zeros((n, 9)), np.zeros((n, 9, 9))
    t0 = time.perf_counter()
    for k in range(n):
        if k > 0:
            filt.predict(data['acc'][k - 1], data['gyro'][k - 1])
        if data['gps'][k]is not None:
            filt.update_gps(data['gps'][k])
        if data['baro'][k] is not None:
            filt.update_baro(data['baro'][k])
        if data['mag'][k]is not None:
            filt.update_mag(data['mag'][k])
        xs[k], Ps[k] = filt.x, filt.P
    return xs, Ps, (time.perf_counter() - t0) / n


def score(xs, Ps, data):
    e = xs - data['truth']
    m = data['t'] >= T_SETTLE
    nees = [e[k] @ np.linalg.solve(Ps[k], e[k]) for k in np.where(m)[0]]
    return {'pos_rmse' : np.sqrt(np.mean(np.sum(e[m, 0:3]**2, axis=1))),
            'vel_rmse' : np.sqrt(np.mean(np.sum(e[m, 3:6]**2, axis=1))),
            'att_rmse_deg' : np.degrees(np.sqrt(np.mean(np.sum(e[m, 6:9]**2, axis=1)))),
            'nees': np.mean(nees)}

if __name__ == "__main__":
    n_mc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(plots_dir, exist_ok=True)

    print("flying")
    log = fly()
    data = generate_sensor_data(log, seed=0)

    fixes = [(k, z) for k, z in enumerate(data['gps']) if z is not None]
    gps_err = [np.linalg.norm(z[0:3] - data['truth'][k, 0:3]) for k, z in fixes if data['t'][k] >= T_SETTLE]
    print(f"raw GPS position RMSE (reference): {np.sqrt(np.mean(np.square(gps_err))):.3f} m")

    print(f"\n=== single flight, t >= {T_SETTLE} s ===")
    print(f"{'filter' :6s} {'pos RMSE' :.13s} {'vel RMSE (m/s)' :>15s} {'att RMSE (deg)' :>15s} {'avg NEES' :9s} {'ms/step':>8s}")
    runs = {}
    for name, cls in FILTERS.items():
        xs, Ps, t_step = run_filter(cls, data)
        s = score(xs, Ps, data)
        runs[name] = xs
        print(f"{name:6s} {s['pos_rmse']:13.3f} {s['vel_rmse']:15.3f} {s['att_rmse_deg']:15.3f}"
              f" {s['nees']:9.1f} {t_step*1000:8.2f}")
        print("(ideal averages NEES for a 9-state filter is about 9)")

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        t = data['t']
        for name, xs in runs.items():
            e = xs - data['truth']
            axes[0].plot(t, np.linalg.norm(e[:, 0:3], axis=1), color=COLORS[name], ls=LSTYLE[name], label=name)
            axes[1].plot(t, np.linalg.norm(e[:, 3:6], axis=1), color=COLORS[name], ls=LSTYLE[name], label=name)
            axes[2].plot(t, np.degrees(np.linalg.norm(e[:, 6:9], axis=1)), color=COLORS[name], ls=LSTYLE[name], label=name)
        for ax, lab in zip(axes, ['position error (m)', 'velocity error (m/s)', 'attitude error (deg)']):
            ax.set_ylabel(lab); ax.grid(alpha=0.3); ax.legend()

        axes[2].set_xlabel('t (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "estimation_errors.png"), dpi=120)
        plt.close(fig)
        print("Saved plots/estimation_errors.png")

        print(f"\nMonte Carlo: {n_mc} sensor-noise seeds on the same flight")
        res = {m: {n: [] for n in FILTERS} for m in ['pos_rmse', 'att_rmse_deg']}
        for seed in range(1, n_mc + 1):
            d = generate_sensor_data(log, seed=100 * seed)
            for name, cls in FILTERS.items():
                xs, Ps, _ = run_filter(cls, d)
                s = score(xs, Ps, d)
                res['pos_rmse'][name].append(s['pos_rmse'])
                res['att_rmse_deg'][name].append(s['att_rmse_deg'])
            print(f" seed {seed}/{n_mc} done")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, key, xl in zip(axes, ['pos_rmse', 'att_rmse_deg'],
                               ['position RMSE (m)', 'attitude RMSE (deg)']):
            plot_ecdf(ax, res[key], COLORS)
            ax.set_xlabel(xl)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "estimation_ecdf.png"), dpi=120)
        plt.close(fig)
        print("saved plots / estimation_ecdf.png")
        for key in res:
            for name in FILTERS:
                v = np.array(res[key][name])
                print(f" {key:13s} {name}: median {np.median(v): .3f} [{np.percentile(v,25):.3f}, {np.percentile(v,75):.3f}]")






