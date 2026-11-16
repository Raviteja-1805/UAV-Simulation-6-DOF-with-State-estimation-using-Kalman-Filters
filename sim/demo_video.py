import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from dynamics import quaternion as quat
from dynamics.rigid_body import RigidBodyParams, rk4_step, gravity_force_body
from sim.simulate import run_sim, make_vehicle
from sim.analysis import make_wind
from control.pid_cascade import CascadedPIDController
from control.lqr_controller import LQRCascadedController
from control.lqi_controller import LQIController

SIM_TIME = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
DT = 0.002
FPS = 20
TARGET = np.array([5.0, 3.0, -10.0])        # NED: D is negative above the ground
YAW = 0.5
MASS_NOM, MASS_TRUE = 1.2, 1.38             # controllers assume 1.2, vehicle is 1.38
J = np.diag([0.0347, 0.0459, 0.0977])
COLORS = {'PID': 'tab:blue', 'LQR': 'tab:orange', 'LQI': 'tab:green'}


def simulate_all():
    controllers = {
        'PID': CascadedPIDController(mass=MASS_NOM),
        'LQR': LQRCascadedController(mass=MASS_NOM, J=J),
        'LQI': LQIController(MASS_NOM, J),
    }
    logs = {}
    for name, ctrl in controllers.items():
        print(f"simulating {name} ...")
        logs[name] = run_sim(TARGET, YAW, SIM_TIME, DT, controller=ctrl,
                             params=make_vehicle(MASS_TRUE, J),
                             wind_body=make_wind(seed=0))   # same wind for all three
    return logs


def ned_to_plot(p):
    """NED (North, East, Down) -> plot axes (North, East, Up)."""
    p = np.atleast_2d(p)
    return p[:, 0], p[:, 1], -p[:, 2]


def build_animation(logs):
    step = int(round(1.0 / (FPS * DT)))          # physics steps per video frame
    t = logs['PID']['t']
    n_frames = len(t) // step

    fig = plt.figure(figsize=(12, 5.5))
    fig.subplots_adjust(left=0.03, right=0.97, wspace=0.18)
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    ax3.set_xlim(-1, 7); ax3.set_ylim(-1, 5); ax3.set_zlim(0, 12)
    ax3.set_xlabel('North (m)'); ax3.set_ylabel('East (m)'); ax3.set_zlabel('Up (m)')
    ax3.view_init(elev=22, azim=-55)
    tx, ty, tz = ned_to_plot(TARGET)
    ax3.scatter(tx, ty, tz, marker='*', s=220, c='k', label='target')
    time_text = ax3.text2D(0.02, 0.95, "", transform=ax3.transAxes)

    trails, arms = {}, {}
    for name in logs:
        trails[name], = ax3.plot([], [], [], color=COLORS[name], lw=1, alpha=0.6, label=name)
        arm_a, = ax3.plot([], [], [], color=COLORS[name], lw=4)
        arm_b, = ax3.plot([], [], [], color=COLORS[name], lw=4)
        arms[name] = (arm_a, arm_b)
    ax3.legend(loc='lower left')

    # right panel: position error vs time, with a moving time marker
    for name, log in logs.items():
        err = np.linalg.norm(log['pos'] - TARGET, axis=1)
        ax2.plot(log['t'], err, color=COLORS[name], label=name)
    ax2.set_title('Position error  ||p - target||  (m)')
    ax2.set_xlabel('t (s)')
    ax2.legend(); ax2.grid(alpha=0.3)
    marker = ax2.axvline(0.0, color='k', lw=1)

    arm_len = 0.7                                  # drawn larger than the real 0.25 m
    front_back = np.array([[arm_len, 0, 0], [-arm_len, 0, 0]])
    left_right = np.array([[0, arm_len, 0], [0, -arm_len, 0]])

    def draw_arm(line, pts_body, R, pos):
        ned = (R @ pts_body.T).T + pos
        x, y, z = ned_to_plot(ned)
        line.set_data(x, y)
        line.set_3d_properties(z)

    def update(k):
        i = k * step
        for name, log in logs.items():
            R = quat.to_rotation_matrix(quat.from_euler(*log['euler'][i]))
            draw_arm(arms[name][0], front_back, R, log['pos'][i])
            draw_arm(arms[name][1], left_right, R, log['pos'][i])
            x, y, z = ned_to_plot(log['pos'][:i + 1:step])
            trails[name].set_data(x, y)
            trails[name].set_3d_properties(z)
        time_text.set_text(f"t = {t[i]:.1f} s   (wind + 15% mass error)")
        marker.set_xdata([t[i], t[i]])

    return fig, update, n_frames

def save_animation(fig, update, n_frames, out_base):
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    if animation.writers.is_available('ffmpeg'):
        path = out_base + '.mp4'
        anim.save(path, writer=animation.FFMpegWriter(fps=FPS, bitrate=2000), dpi=90)
    else:
        path = out_base + '.gif'
        print("ffmpeg not found: writing a GIF instead (sudo apt install ffmpeg for mp4)")
        anim.save(path, writer=animation.PillowWriter(fps=FPS), dpi=70)
    return path


if __name__ == "__main__":
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logs = simulate_all()
    fig, update, n_frames = build_animation(logs)
    print(f"rendering {n_frames} frames ...")
    path = save_animation(fig, update, n_frames, os.path.join(plots_dir, "demo"))
    print("saved", path)