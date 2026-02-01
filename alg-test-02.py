import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from core.viz_copter import HexacopterSprite


def lagrange_multiplier(p, pdot, q, qdot, u, L, md, mp) -> float:
    """
    Computes the lagrange multiplier of the taught slung payload problem.
    """
    p = np.asarray(p).reshape(3,)
    q = np.asarray(q).reshape(3,)
    pdot = np.asarray(pdot).reshape(3,)
    qdot = np.asarray(qdot).reshape(3,)
    u = np.asarray(u).reshape(3,)

    r = q - p
    rd = qdot - pdot

    num = (1/md)*(r.T@u) - (rd@rd)
    den = 2*(L*L)*(1/mp + 1/md)

    return num/den


def slung_payload_system(t, x, u, L, md, mp, g=9.81):
    """
    """
    p = x[0:3]
    pdot = x[3:6]
    q = x[6:9]
    qdot = x[9:12]

    u = control(t, x, md, mp, g)

    lam = lagrange_multiplier(p, pdot, q, qdot, u, L, md, mp)
    r = q - p

    ez = np.array([0, 0, 1])

    pddot = (1/md)*u - g*ez - (2*lam/md)*r
    qddot = -g*ez + (2*lam/mp)*r

    xdot = np.zeros_like(x)
    xdot[0:3] = pdot
    xdot[3:6] = pddot
    xdot[6:9] = qdot
    xdot[9:12] = qddot

    return xdot


def control(t, x, md, mp, g=9.81):
    """
    Just a test input
    """
    # uz = (md + mp)*g
    # ux = np.sin(2*np.pi*0.4*t)
    # uy = 0.5*np.cos(2*np.pi*0.4*t)
    uz = (md + mp)*g
    ux = 0
    uy = 0

    return np.array([ux, uy, uz])


if __name__ == '__main__':
    md = 2
    mp = 1
    L = 1
    g = 9.81

    # initial conditions
    p0 = np.array([0, 0, 0])
    v0 = np.array([0, 0, 0])
    q0 = p0 + np.array([0, 0, -L])
    # it is non-physical to apply an initial z velocity to the payload and not the drone
    w0 = v0 + np.array([1, 0.5, 0])

    x0 = np.hstack([p0, v0, q0, w0])

    # solve
    t0, tf = 0, 5
    t_eval = np.linspace(t0, tf, 200)

    def ode(t, x):
        return slung_payload_system(t, x, lambda tt, xx: control(tt, xx, md, mp, g), L, md, mp, g=g)

    sol = solve_ivp(ode, (t0, tf), x0, t_eval=t_eval)

    X = sol.y.T

    p = X[:, 0:3]
    q = X[:, 6:9]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # artists for payload + cable
    cable_ln = ax.plot([p[0, 0], q[0, 0]], [p[0, 1], q[0, 1]], [
                       p[0, 2], q[0, 2]], color="gray", lw=2)[0]
    payload_pt = ax.plot([q[0, 0]], [q[0, 1]], [q[0, 2]],
                         color='c', marker='o', markersize=5)[0]

    drone = HexacopterSprite(arm_len=0.5, rotor_r=0.1).draw(ax)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # set bounds
    all_xyz = np.vstack([p, q])
    mins = all_xyz.min(axis=0) - 0.5
    maxs = all_xyz.max(axis=0) + 0.5
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    drone.set_axes_equal(ax)

    from matplotlib.animation import FuncAnimation

    def animate(k):
        # visualization-only orientation: you can keep 0 for now
        drone.update(p=p[k], roll=0, pitch=0, yaw=0)

        payload_pt.set_data_3d([q[k, 0]], [q[k, 1]], [q[k, 2]])
        cable_ln.set_data_3d([p[k, 0], q[k, 0]], [
                             p[k, 1], q[k, 1]], [p[k, 2], q[k, 2]])
        return drone._arm_lines + drone._rotor_lines + [payload_pt, cable_ln]

    ani = FuncAnimation(fig, animate, frames=len(
        t_eval), interval=20, blit=False)
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # test trajectory
    # t = np.linspace(0, 2, 200)
    # x = np.cos(t)
    # y = np.sin(t)
    # z = t
    # ax.plot(x, y, z)
    #
    # drone1 = HexacopterSprite().draw(ax)
    #
    # drone1.update(p=(x[150], y[150], z[150]), yaw=0, pitch=0, roll=0)
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # drone1.set_axes_equal(ax)
    # plt.show()
