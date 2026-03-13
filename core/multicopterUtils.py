import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from numpy.typing import NDArray
from viz_copter import HexacopterSprite


class Drone(object):
    """
    Defines a drone from given parameters.
    """

    def __init__(self, params):
        self.params = params

    def _YPR_to_T_EB(self, yaw, pitch, roll) -> NDArray[np.float64]:
        """
        Provides a transformation matrix from B to E (T^EB) given yaw, pitch, and roll.
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        T_EB = np.array([[cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
                         [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
                        [-sp, sr*cp, cr*cp]])

        return T_EB

    def _euler_rates_matrix(self, roll, pitch) -> NDArray[np.float64]:
        """
        Time derivative of the euler angles.
        """
        sr, cr = np.sin(roll), np.cos(roll)
        tp, cp = np.tan(pitch), np.cos(pitch)

        return np.array([[1, sr*tp, cr*tp],
                         [0, cr, -sr],
                         [0, sr/cp, cr/cp]])

    def _wrap_angle(self, a) -> float:
        """
        Wraps angle.
        """
        return (a + np.pi) % (2*np.pi) - np.pi

    def lagrange_multiplier(self, sBT, vEB, sPT, vEP, u, L, mB, mP) -> float:
        """
        Computes the Lagrange multiplier of the taught two-point-mass slung payload problem.
        """
        r = sPT - sBT
        rd = vEP - vEB

        num = (1/mB)*(r@u) - (rd@rd)
        den = 2*(L*L)*(1/mP + 1/mB)

        return num/den

    def pd_hover_controller(self, sBT, vEB, euler, omega, m, g,
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

        # outer loop PD (earth frame)
        a_cmd_E = Kp_pos@(p_ref - sBT) + Kd_pos@(0 - vEB)

        # vertial thrust
        Z = m*(g + a_cmd_E[2])

        # compute approximate desired roll and pitch
        ax = a_cmd_E[0]
        ay = a_cmd_E[1]
        c_psi = np.cos(yaw_ref)
        s_psi = np.sin(yaw_ref)

        theta_cmd = (ax*c_psi + ay*s_psi)/g
        phi_cmd = (ax*s_psi - ay*c_psi)/g

        # tilt limits
        theta_cmd = np.clip(theta_cmd, -max_tilt, max_tilt)
        phi_cmd = np.clip(phi_cmd, -max_tilt, max_tilt)

        # inner loop attitude PD
        e_eta = np.array([phi_cmd - phi,
                          theta_cmd - theta,
                          self._wrap_angle(yaw_ref - psi)])

        nB = Kp_att@e_eta + Kd_att@(0 - omega)
        L, M, N = nB

        return Z, L, M, N

    def uav_swing_payload_system(self, t, x) -> NDArray[np.float64]:
        """
        The 12-state aircraft dynamics + 6-state point mass payload model.
        """
        mB = self.params["mB"]
        mP = self.params["mP"]
        Lc = self.params["Lc"]
        g = self.params.get("g", 9.81)

        sBTE = x[0:3]
        vEBB = x[3:6]
        rpy = x[6:9]
        omegaBEB = x[9:12]
        sPTE = x[12:15]
        vEPE = x[15:18]

        phi, theta, psi = rpy
        T_EB = self._YPR_to_T_EB(psi, theta, phi)
        T_BE = T_EB.T
        e3E = np.array([0, 0, 1])  # lab frame

        # kinematics
        vEBE = T_EB@vEBB
        E = self._euler_rates_matrix(phi, theta)
        ddt_rpy = E@omegaBEB

        # control
        Z, L, M, N = self.pd_hover_controller(sBTE, vEBE, rpy, omegaBEB, mB + mP, g,
                                              p_ref=self.params["p_ref"],
                                              yaw_ref=self.params["yaw_ref"],
                                              Kp_pos=self.params["Kp_pos"],
                                              Kd_pos=self.params["Kd_pos"],
                                              Kp_att=self.params["Kp_att"],
                                              Kd_att=self.params["Kd_att"])

        uE = T_EB@np.array([0, 0, Z])
        lam = self.lagrange_multiplier(sBTE, vEBE, sPTE, vEPE, uE, Lc, mB, mP)
        sPBE = sPTE - sBTE

        F_cable_E = -2*lam*sPBE
        F_B = np.array([0, 0, Z]) + T_BE@F_cable_E

        # translational dynamics (body frame)
        OmegaBEB = np.array([[0, -omegaBEB[2], omegaBEB[1]],
                             [omegaBEB[2], 0, -omegaBEB[0]],
                             [-omegaBEB[1], omegaBEB[0], 0]])

        aEBB = -OmegaBEB@vEBB + (1/mB)*F_B - g*(T_BE@e3E)

        # rotational dynamics (body frame)
        J = self.params["J"]
        nBB = np.array([L, M, N])
        ddt_omega_BEB = np.linalg.inv(J)@(nBB - OmegaBEB@J@omegaBEB)

        # payload dynamics (lab frame)
        aEPE = -g*e3E - (1/mP)*F_cable_E

        xdot = np.zeros_like(x)
        xdot[0:3] = vEBE
        xdot[3:6] = aEBB
        xdot[6:9] = ddt_rpy
        xdot[9:12] = ddt_omega_BEB
        xdot[12:15] = vEPE
        xdot[15:18] = aEPE

        return xdot


if __name__ == '__main__':
    p10 = np.array([0, 0, 1])
    v0b = np.zeros(3)
    euler0 = np.zeros(3)
    omega0 = np.zeros(3)

    Lc = 3

    p20 = p10 + np.array([0, 0, -Lc])
    p2dot0 = v0b + np.array([0, 0, 0])

    x0 = np.hstack([p10, v0b, euler0, omega0, p20, p2dot0])

    params = {"mB": 2,
              "mP": 1,
              "Lc": Lc,
              "g": 9.81,
              "J": np.diag([0.03, 0.03, 0.05]),
              "p_ref": np.array([3, 2, 7]),
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

    drone = Drone(params)

    sol = solve_ivp(drone.uav_swing_payload_system, (t0, tf),
                    x0, t_eval=t_eval, method='RK45')

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
