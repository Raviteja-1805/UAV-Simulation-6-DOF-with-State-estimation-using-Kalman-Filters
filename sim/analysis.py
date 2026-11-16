import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.simulate import run_sim, make_vehicle
from control.pid_cascade import CascadedPIDController
from control.lqr_controller import LQRCascadedController
from control.lqi_controller import LQIController

def make_wind(mean_body= (1.5, 1.0, 0.0), gust_std=1.0, gust_tau=1.5, seed=None):
    rng = np.random.default_rng(seed)
    state = {'gust': np.zeros(3), 't_last':0.0}

    def wind_body(t):
        dt = max(t - state['t_last'], 1e-4)
        state['t_last']= t
        alpha = dt / gust_tau
        state ['gust'] += alpha * (-state['gust'] + gust_std * rng.standard_normal(3))
        return np.array(mean_body) + state['gust']

    return wind_body

def compute_metrics(log, target, step_size):
    pos = log['pos']; t = log['t']
    err = np.linalg.norm(pos - target, axis=1)

    tol = 0.05 * step_size
    settled_idx = len(t) - 1
    for i in range(len(t)):
        if np.all(err[i:]<tol):
            settled_idx = i
            break
    settling_time = t[settled_idx]

    overshoot_vec = pos -target
    overshoot = 0.0
    for axis in range(3):
        proj = pos[:, axis] - pos[0, axis]
        final = target[axis] - pos[0, axis]
        if abs(final) > 1e-6:
            peak = proj[np.argmax(np.abs(proj))] if abs(final) > 0 else 0
            os_pct = max(0.0, (np.sign(final)* peak - abs(final)) / abs(final) * 100)
            overshoot = max(overshoot, os_pct)

    steady_region = err[int(0.7*len(err)):]
    rmse_ss = np.sqrt(np.mean(steady_region**2))
    return {
        'settling_time': settling_time,
        'overshoot_pct': overshoot,
        'rmse_steady_state': rmse_ss,
        'final_error': err[-1],
    }

def compare_pid_vs_lqr(mass=1.2, J=None, target=np.array([5.0, 3.0, -10.0]),
                       yaw_setpoint=0.5, sim_time= 15.0, wind_factory=None, mass_ctrl=1.2, J_ctrl=None):
    if J is None:
        J = np.diag([0.0347, 0.0459, 0.0977])
    if J_ctrl is None:
        J_ctrl = np.diag([0.0347, 0.0459, 0.0977])
    params = make_vehicle(mass, J)
    results = {}
    logs = {}

    for name, ctrl in [('PID', CascadedPIDController(mass=mass_ctrl)),
                       ('LQR', LQRCascadedController(mass=mass_ctrl, J=J_ctrl)),
                       ('LQI', LQIController(mass_ctrl, J_ctrl))]:
        wind = wind_factory() if wind_factory is not None else None
        log = run_sim(pos_setpoint=target, yaw_setpoint=yaw_setpoint,
                          sim_time=sim_time, wind_body= wind, controller=ctrl, params=params)

        step_size = np.linalg.norm(target - np.zeros(3))
        results[name] = compute_metrics(log, target, step_size)
        logs[name] = log
    return results, logs

def plot_comparison(logs, target, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {'PID': 'tab:blue', 'LQR':'tab:orange', 'LQI': 'tab:green'}

    ax = axes[0, 0]
    for name, log in logs.items():
        err = np.linalg.norm(log['pos'] - target, axis=1)
        ax.plot(log['t'], err, label=name, color=colors[name])
        ax.set_title('position error norm vs time')
        ax.set_xlabel('t (s)'); ax.set_ylabel('||pos - target|| (m)')
        ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for name, log in logs.items():
        ax.plot(log['t'], log['pos'] [:, 2], label=f'{name}D', color=colors[name])
        ax.axhline(target[2], color='k', linestyle='--', linewidth=1, label='target D')
        ax.set_title('Altitude (D, NED) response')
        ax.set_xlabel('t(s)'); ax.set_ylabel('D position (m)')
        ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for name, log in logs.items():
        ax.plot(log['pos'][:, 0], log['pos'] [:, 1], label=name , color=colors[name])
    ax.plot(target[0], target[1], 'k*', markersize=12, label='target')
    ax.plot(0, 0, 'ko', markersize=6, label='start')
    ax.set_title('Horizontal (N-E) trajectory')
    ax.set_xlabel('North(m)'); ax.set_ylabel('East (m)')
    ax.legend(); ax.grid(alpha=0.3); ax.axis('equal')

    ax = axes[1, 1]
    for name, log in logs.items():
        ax.plot(log['t'], log['thrust'], label=name, color=colors[name])
    ax.set_title('Thrust command (control effort)')
    ax.set_xlabel('t (s)'); ax.set_ylabel('Thrust (N)')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)

def monte_carlo(n_trials=100, mass_nom=1.2, J_nom=None, target=np.array([5.0, 3.0, -10.0]),
                sim_time=15.0, seed=42):
    if J_nom is None:
        J_nom = np.diag([0.0347, 0.0459, 0.0977])
    rng = np.random.default_rng(seed)
    records = {n : [] for n in NAMES}

    for trial in range(n_trials):
        mass = mass_nom * (1 + rng.uniform(-0.15, 0.15))
        J = np.diag(J_nom) * (1 + rng.uniform(-0.15, 0.15, size=3))
        J = np.diag(np.diag(J_nom) * (1 + rng.uniform(-0.15, 0.15, size=3)))
        mean_wind = tuple(rng.uniform(-2, 2, size=3))
        gust = rng.uniform(0.3, 1.5)

        def wind_factory(mean_wind=mean_wind, gust=gust, trial=trial):
            return make_wind(mean_body=mean_wind, gust_std=gust, seed=trial)
        
        results, _ = compare_pid_vs_lqr(mass=mass, J= J, target=target,
                                         sim_time=sim_time, wind_factory=wind_factory)

        for name in NAMES:
            records[name].append(results[name])
        print(f" trial {trial + 1}/{n_trials} done")
    return records

NAMES = ('PID', 'LQR', 'LQI')
COLORS = {'PID': 'tab:blue', 'LQR': 'tab:orange', 'LQI': 'tab:green'}

def ecdf(values):
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def plot_ecdf(ax, samples_by_name, colors, log_x=False):
    for name, values in samples_by_name.items():
        if len(values) == 0:
            raise ValueError(f"no samples recorded for {name}")
        
    curves = {name: ecdf(v) for name, v in samples_by_name.items()}
    x_max = max(x[-1] for x, _ in curves.values())
    for name, (x, y) in curves.items():
        xs = np.concatenate([[x[0]], x, [x_max]])
        ys = np.concatenate([[0.0], y, [1.0]])
        ax.step(xs, ys, where='post', label=name, color=colors[name], lw=2)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel('fraction of runs <= x')
    if log_x:
        ax.set_xscale('log')
    ax.grid(alpha=0.3)
    ax.legend(loc='best')

def plot_monte_carlo(records, out_path):
    metrics = ['settling_time', 'overshoot_pct', 'rmse_steady_state', 'final_error']
    titles = ['Settling_time (s)', 'Overshoot (%)', 'Steady-state RMSE (m)', 'Final_error']
    log_x = [False, False, True, True]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric, title, lx in zip(axes.flat, metrics, titles, log_x):
        samples = {n: [r[metric] for r in records[n]] for n in NAMES}
        plot_ecdf(ax, samples, COLORS, log_x=lx)
        ax.set_title(title)
        ax.set_xlabel(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)

    print("\n== Monte Carlo summary: median [25th, 75th percentile] ===")
    for name in NAMES:

        print(f"\nController: {name}")

        for metric, title in zip(metrics, titles):
            
            


            vals = np.array([
                r[metric] for r in records[name]
            ])

            q1, med, q3 = np.percentile(
                vals, [25, 50, 75]
            )

            print(
                f" {title:24s}: "
                f"{med:9.3f} "
                f"[{q1:.3f}, {q3:.3f}]"
            )

    # Divergence statistics
    print("\n== Divergence summary ==")

    for name in NAMES:

        n_bad = sum(
            r['final_error'] > 50
            for r in records[name]
        )

        print(
            f"{name}: {n_bad} / {len(records[name])} "
            f"runs diverged (final error > 50 m)"
        )

if __name__ == "__main__":
    
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    target = np.array([5.0, 3.0, -10.0])

    print("### Single scenario: step response with wind  ###")
    results, logs = compare_pid_vs_lqr(
        target=target, sim_time=15.0,
        wind_factory=lambda: make_wind(mean_body=(1.5, 1.0, 0.0), gust_std=1.0, seed=0))
    for name, m in results.items():
        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
    plot_comparison(logs, target, os.path.join(plots_dir, "pid_vs_lqr_step_wind.png"))
    print("Saved plots/pid_vs_lqr_step_wind.png")

    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"\n### Monte Carlo ({n_trials} trials, +/-15% mass/inertia, random wind) ###")
    records = monte_carlo(n_trials=n_trials, target=target, sim_time=15.0)
    plot_monte_carlo(records, os.path.join(plots_dir, "monte_carlo_comparison.png"))
    print("Saved plots/monte_carlo_comparison.png")