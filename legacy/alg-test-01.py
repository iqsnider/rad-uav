import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from core.viz_copter import HexacopterSprite

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

    # Collect artists for blitting (3D blit is finicky; we’ll set blit=False)
    artists = drone1._arm_lines + drone1._rotor_lines

    def animate(i):
        # position along trajectory
        p = (x[i], y[i], z[i])

        # some attitude motion
        yaw = 2*np.pi * (i / (len(t)-1))     # 0 -> 2pi
        pitch = 0.15*np.sin(0.15*i)
        roll = 0.10*np.cos(0.12*i)

        drone1.update(p=p, roll=roll, pitch=pitch, yaw=yaw)

        # If you want the camera to slowly orbit too, uncomment:
        # ax.view_init(elev=20, azim=0.5*i)

        return artists

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(t),
        interval=30,     # ms between frames
        blit=False       # 3D + blit often doesn't behave; False is safest
    )
    plt.show()
