import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from core.viz_copter import HexacopterSprite


def wrap_angle(a) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi


def YPR_to_R(yaw, pitch, roll) -> NDArray[np.float64]:
    """
    Rotates vectors from body frame to world frame given yaw, pitch, and roll.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R_BE = np.array([[cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
                     [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
                    [-sp, sr*cp, cr*cp]])

    return R_BE


def euler_rates_matrix(roll, pitch) -> NDArray[np.float64]:
    """
    Time derivative of euler angles.
    """
    sr, cr = np.sin(roll), np.cos(roll)
    tp, cp = np.tan(pitch), np.cos(pitch)

    return np.array([[1, sr*tp, cr*tp],
                     [0, cr, -sr],
                     [0, sr/cp, cr/cp]])


def euler_from_R(R_BE) -> tuple[float, float, float]:
    """
    Extracts Euler angles from rotation matrix.
    """
    theta = -np.arcsin(R_BE[2, 0])
    phi = np.arctan2(R_BE[2, 1], R_BE[2, 2])
    psi = np.arctan2(R_BE[1, 0], R_BE[0, 0])

    return phi, theta, psi


def lagrange_multiplier(p1, p1dot, p2, p2dot, u, L, md, mp) -> float:
    """
    Computes the Lagrange multiplier of the taught two-point-mass slung payload problem.
    """
    r = p2 - p1
    rd = p2dot - p1dot

    num = (1/md)*(r.T@u) - (rd@rd)
    den = 2*(L*L)*(1/mp + 1/md)

    return num/den


def pd_hover_controller(p1, p1dot, euler, omega, m, g,
                        p_ref=np.array([0, 0, 2]),
                        yaw_ref=0,
                        Kp_pos=np.diag([2, 2, 6]),
                        Kd_pos=np.diag([2.5, 2.5, 4]),
                        Kp_att=np.diag([8, 8, 4]),
                        Kd_att=np.diag([2.5, 2.5, 1.5])) -> tuple[float, float, float, float]:
    """
    Cascaded PD controller
    """
    phi, theta, psi = euler
    max_tilt = np.deg2rad(35)

    # outer loop PD (lab frame)
    a_cmd = Kp_pos@(p_ref - p1) + Kd_pos@(0 - p1dot)

    # vertial thrust
    Z = m*(g + a_cmd[2])

    # compute approximate desired roll and pitch
    ax = a_cmd[0]
    ay = a_cmd[1]
    c_psi = np.cos(yaw_ref)
    s_psi = np.sin(yaw_ref)

    theta_cmd = (ax*c_psi + ay*s_psi)/g
    phi_cmd = (ax*s_psi - ay*c_psi)/g

    # tilt limits
    theta_cmd = np.clip(theta_cmd, -max_tilt, max_tilt)
    phi_cmd = np.clip(phi_cmd,   -max_tilt, max_tilt)

    # inner loop attitude PD
    e_eta = np.array([phi_cmd - phi,
                      theta_cmd - theta,
                      wrap_angle(yaw_ref - psi)])

    tau = Kp_att@e_eta + Kd_att@(0 - omega)
    L, M, N = tau

    return Z, L, M, N


def uav_payload_system(t, x, params) -> NDArray[np.float64]:
    """
    The 12-state aircraft dynamics + 6-state point mass payload model.
    """
    md = params["md"]
    mp = params["mp"]
    Lc = params["Lc"]
    g = params.get("g", 9.81)

    p1 = x[0:3]
    v_b = x[3:6]
    rpy = x[6:9]
    omega = x[9:12]
    p2 = x[12:15]
    p2dot = x[15:18]

    phi, theta, psi = rpy
    R_BE = YPR_to_R(psi, theta, phi)
    e_z_lab = np.array([0, 0, 1])  # lab frame

    # kinematics
    p1dot = R_BE@v_b
    E = euler_rates_matrix(phi, theta)
    rpydot = E@omega

    # control
    Z, L, M, N = pd_hover_controller(p1, p1dot, rpy, omega, md + mp, g,
                                     p_ref=params["p_ref"],
                                     yaw_ref=params["yaw_ref"],
                                     Kp_pos=params["Kp_pos"],
                                     Kd_pos=params["Kd_pos"],
                                     Kp_att=params["Kp_att"],
                                     Kd_att=params["Kd_att"])

    u_lab = R_BE@np.array([0, 0, Z])
    lam = lagrange_multiplier(p1, p1dot, p2, p2dot, u_lab, Lc, md, mp)
    r = p2 - p1

    F_cable_lab = -2*lam*r
    F_b = np.array([0, 0, Z]) + R_BE.T@F_cable_lab

    # translational dynamics (body frame)
    vdot_b = -np.cross(omega, v_b) + (1/md)*F_b - g*(R_BE.T@e_z_lab)

    # rotational dynamics (body frame)
    J = params["J"]
    tau = np.array([L, M, N])
    omega_dot = np.linalg.inv(J)@(tau - np.cross(omega, J@omega))

    # payload dynamics (lab frame)
    p2ddot = -g*e_z_lab - (1/mp)*F_cable_lab

    xdot = np.zeros_like(x)
    xdot[0:3] = p1dot
    xdot[3:6] = vdot_b
    xdot[6:9] = rpydot
    xdot[9:12] = omega_dot
    xdot[12:15] = p2dot
    xdot[15:18] = p2ddot

    return xdot


if __name__ == '__main__':
    p10 = np.array([0, 0, 0])
    v0b = np.zeros(3)
    euler0 = np.zeros(3)
    omega0 = np.zeros(3)

    Lc = 3

    p20 = p10 + np.array([0, 0, -Lc])
    p2dot0 = v0b + np.array([0, 0, 0])

    x0 = np.hstack([p10, v0b, euler0, omega0, p20, p2dot0])

    params = {"md": 2,
              "mp": 1,
              "Lc": Lc,
              "g": 9.81,
              "J": np.diag([0.03, 0.03, 0.05]),
              "p_ref": np.array([3, 2, 2]),
              "yaw_ref": 1,
              "Kp_pos": np.diag([2, 2, 6]),
              "Kd_pos": np.diag([2.5, 2.5, 4]),
              "Kp_att": np.diag([8, 8, 4]),
              "Kd_att": np.diag([2.5, 2.5, 1.5])}

    # make proper frame rate for animation
    t0, tf = 0, 15
    fps = 30
    total_frames = fps*(tf - t0)
    t_eval = np.linspace(t0, tf, total_frames)

    sol = solve_ivp(uav_payload_system, (t0, tf),
                    x0, args=[params], t_eval=t_eval, method='RK45')

    X = sol.y.T

    p1 = X[:, 0:3]
    p2 = X[:, 12:15]
    rpy = X[:, 6:9]

    parchment = "#f4f1ea"

    fig = plt.figure(facecolor=parchment)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(parchment)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#F8DE7E")
        axis.pane.set_edgecolor("#d8d2c5")
        axis.pane.set_alpha(1)

    ax.grid(color="#d0c8b8", linestyle="--", linewidth=0.5)
    ax.tick_params(colors="#3a3a3a")

    # additional cable, payload, and attachment artists
    cable_line = ax.plot([p1[0, 0], p2[0, 0]], [p1[0, 1], p2[0, 1]], [
        p1[0, 2], p2[0, 2]], color="gray", lw=2)[0]
    payload_point = ax.plot([p2[0, 0]], [p2[0, 1]], [p2[0, 2]],
                            color='c', marker='o', markersize=7)[0]
    attach_point = ax.plot([p1[0, 0]], [p1[0, 1]], [p1[0, 2]],
                           color='m', marker='o', markersize=7)[0]
    fig_title = ax.set_title("Cascaded PD Controller\n"
                             f"$p = ({p1[0, 0]:.2f}, {
                                 p1[0, 1]:.2f}, {p1[0, 2]:.2f})$ "
                             f"$p_{{ref}} = ({params['p_ref'][0]:.2f}, {params['p_ref'][1]:.2f}, {
                                 params['p_ref'][2]:.2f})$\n"
                             f"$p = ({p2[0, 0]:.2f}, {
                                 p2[0, 1]:.2f}, {p2[0, 2]:.2f})$ "
                             f"$p_{{ref}} = (\\cdot)$")

    def plot_3d_plus(ax, x, y, z, size=0.1, color='k', lw=1):
        """
        Plots a 3D plus marker.
        """
        ax.plot([x - size, x + size], [y, y], [z, z], color=color, lw=lw)
        ax.plot([x, x], [y - size, y + size], [z, z], color=color, lw=lw)
        ax.plot([x, x], [y, y], [z - size, z + size], color=color, lw=lw)

    plot_3d_plus(ax, p1[0, 0],  p1[0, 1],  p1[0, 2],
                 size=0.3, color='#32CD32')
    plot_3d_plus(ax, params["p_ref"][0], params["p_ref"]
                 [1], params["p_ref"][2], size=0.3, color='#E10600')

    drone = HexacopterSprite(arm_len=0.5, rotor_r=0.1).draw(ax)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # set bounds
    all_xyz = np.vstack([p1, p2])
    mins = all_xyz.min(axis=0) - 0.5
    maxs = all_xyz.max(axis=0) + 0.5
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    drone.set_axes_equal(ax)

    def animate(k):
        roll, pitch, yaw = rpy[k]
        drone.update(p=p1[k], roll=roll, pitch=pitch, yaw=yaw)
        fig_title.set_text("Slung Payload Model with Cascaded PD Controller\n\n"
                           f"$p_{{drone}} = ({p1[k, 0]:.2f}, {p1[k, 1]:.2f}, {
                               p1[k, 2]:.2f}) $ "
                           f"$p_{{ref}} = ({params['p_ref'][0]:.2f}, {params['p_ref'][1]:.2f}, {
                               params['p_ref'][2]:.2f})$\n"
                           f"$p_{{payload}} = ({p2[k, 0]:.2f}, {
                               p2[k, 1]:.2f}, {p2[k, 2]:.2f})$ "
                           f"$p_{{ref}} = (\\cdot)$")

        payload_point.set_data_3d([p2[k, 0]], [p2[k, 1]], [p2[k, 2]])
        attach_point.set_data_3d([p1[k, 0]], [p1[k, 1]], [p1[k, 2]])
        cable_line.set_data_3d([p1[k, 0], p2[k, 0]], [
            p1[k, 1], p2[k, 1]], [p1[k, 2], p2[k, 2]])
        return drone._arm_lines + drone._rotor_lines + [payload_point, attach_point, cable_line, fig_title]

    ani = FuncAnimation(fig, animate, frames=len(
        t_eval), interval=20)
    plt.show()
