import numpy as np
import matplotlib.pyplot as plt


def YPR_to_R(yaw, pitch, roll):
    """
    Rotates vectors from body frame to lab frame given yaw, pitch, and roll.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R_BE = np.array([[cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
                     [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
                    [-sp, sr*cp, cr*cp]])

    return R_BE


class HexacopterSprite:
    """
    Draws an updateable hexacopter sprite.
    """

    def __init__(self, arm_len=0.15, rotor_r=0.05, n_circle=50, frame_color='k', rotor_color='b', front_color='r'):
        self.arm_len = arm_len
        self.rotor_r = rotor_r
        self.n_circle = n_circle
        self.frame_color = frame_color
        self.rotor_color = rotor_color
        self.front_color = front_color
        self.front_rotor_index = [0, 5]

        # initialize body-frame geometry
        L = self.arm_len

        # 3 arm directions in the body xy-plane 0, 60, 120 degrees
        ang = np.deg2rad([30, 90, 150])
        # matrix of arm directions 3x3
        self.body_arm_dirs = np.vstack(
            [np.cos(ang), np.sin(ang), np.zeros_like(ang)])

        # define 3 line segments 3x2
        self.body_arm_lines = []
        for i in range(3):
            body_arm_line = np.column_stack(
                (-L*self.body_arm_dirs[:, i], L*self.body_arm_dirs[:, i]))
            self.body_arm_lines.append(body_arm_line)

        # place rotor centers 3x6
        self.body_rotor_centers = np.hstack(
            [L*self.body_arm_dirs, -L*self.body_arm_dirs])

        # make circle 3xn_circle
        t = np.linspace(0, 2*np.pi, self.n_circle)
        self.body_rotor_ring = np.vstack([self.rotor_r*np.cos(t),
                                         self.rotor_r*np.sin(t),
                                         np.zeros_like(t)])

        # matplotlib artists
        self._ax = None
        self._arm_lines = []
        self._rotor_lines = []

    def draw(self, ax):
        """
        Initializes matplotlib artists for given 3D plot.
        """
        self._ax = ax

        # create the arms
        self._arm_lines = []
        for _ in range(3):
            line = ax.plot([0, 0], [0, 0], [0, 0], lw=2,
                           color=self.frame_color)[0]
            self._arm_lines.append(line)

        # create rotors
        self._rotor_lines = []
        for i in range(6):
            color = self.front_color if i in self.front_rotor_index else self.rotor_color
            line = ax.plot([0], [0], [0], lw=3, color=color)[0]
            self._rotor_lines.append(line)

        return self

    def update(self, p, R=None, roll=0, pitch=0, yaw=0):
        """
        Updates artists to given orientation.
        """
        p = np.asarray(p).reshape(3, 1)
        if R is None:
            R = YPR_to_R(yaw, pitch, roll)

        # update arms
        for i, body_arm in enumerate(self.body_arm_lines):
            world_arm = (R @ body_arm) + p
            line = self._arm_lines[i]
            line.set_data_3d(world_arm[0], world_arm[1], world_arm[2])

        # update rotors
        world_centers = (R @ self.body_rotor_centers) + p
        for i in range(6):
            world_ring = (R @ self.body_rotor_ring) + world_centers[:, i:i+1]
            line = self._rotor_lines[i]
            line.set_data_3d(world_ring[0], world_ring[1], world_ring[2])

        return self

    def set_axes_equal(self, ax):
        """
        Prevent the graphic from squashing.
        """
        try:
            ax.set_box_aspect((1, 1, 1))  # mpl >= 3.3
        except Exception:
            pass

        def set_axes_equal(ax):
            xlim = ax.get_xlim3d()
            ylim = ax.get_ylim3d()
            zlim = ax.get_zlim3d()

            xmid = 0.5*(xlim[0] + xlim[1])
            ymid = 0.5*(ylim[0] + ylim[1])
            zmid = 0.5*(zlim[0] + zlim[1])

            xr = abs(xlim[1] - xlim[0])
            yr = abs(ylim[1] - ylim[0])
            zr = abs(zlim[1] - zlim[0])
            r = 0.5*max(xr, yr, zr)

            ax.set_xlim3d(xmid - r, xmid + r)
            ax.set_ylim3d(ymid - r, ymid + r)
            ax.set_zlim3d(zmid - r, zmid + r)

        set_axes_equal(ax)


if __name__ == '__main__':

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # test trajectory
    t = np.linspace(0, 2, 200)
    x = np.cos(t)
    y = np.sin(t)
    z = t
    ax.plot(x, y, z)

    drone1 = HexacopterSprite().draw(ax)

    drone1.update(p=(x[150], y[150], z[150]), yaw=0, pitch=0, roll=0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drone1.set_axes_equal(ax)

    plt.show()
