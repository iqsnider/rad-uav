import numpy as np
import matplotlib.pyplot as plt

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

    plt.show()
