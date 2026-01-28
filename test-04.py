import pygame
import numpy as np

WIDTH = 360
HEIGHT = 480
FPS = 60
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


def rotation(points, theta):
    """
    Applies a rotation transformation
    """
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    return points @ R.T


def polygon(screen, pts):
    """
    Draws a pygame polygon from a list of points
    """
    pygame.draw.polygon(screen, GREEN, pts.tolist(), 5)


def resolve_wall(rect_world, pos, vel, omega, mass, MoI, n, d, e=0.5):
    """
    Does collision detection where the wall is defined by d = n dot x.
    Impulse is applied if d > n dot x.
    """
    depths = d - (rect_world @ n)
    idx = np.argmax(depths)

    # if no collision yet
    if depths[idx] <= 0:
        return pos, vel, omega

    # displace body by wall penetration depth
    p = rect_world[idx]
    pos = pos + n*depths[idx]
    r = p - pos  # vector from CoM to contact point

    # impulse calcuation
    # velocity of the contact point
    v_contact = vel + np.cross([0, 0, omega],
                               [r[0], r[1], 0])[:2]

    # velocity into the wall
    v_n = np.dot(v_contact, n)
    if v_n >= 0:
        return pos, vel, omega

    rn = np.cross([r[0], r[1], 0],
                  [n[0], n[1], 0])[2]
    j = (-e - 1)*v_n/(1/mass + rn*rn/MoI)

    vel += (j/mass)*n
    omega += (j*rn)/MoI

    return pos, vel, omega


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("simple drone simulation")
clock = pygame.time.Clock()

a, b = 25, 100
rect_local = np.array([[-b,  a],
                       [-b, -a],
                       [b, -a],
                       [b,  a]])

pos = np.array([WIDTH/2, HEIGHT/4])
vel = np.array([120.0, 0.0])

mass = 2.0
J = (1/12)*mass*((2*b)**2 + (2*a)**2)
g = 900.0  # px/s^2
theta = 0
omega = 15.0

e = 0.6  # bounciness [0,1]

running = True
while running:
    dt = clock.tick(FPS)/1000
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # "dynamics"
    force = np.array([0, 0])
    force[1] += mass*g
    acc = force/mass

    vel += acc*dt
    pos += vel*dt

    # "control"
    theta += omega*dt
    rect_world = rotation(rect_local, theta) + pos

    # collision detection
    # left wall: x = 0, n = +x
    pos, vel, omega = resolve_wall(rect_world, pos, vel, omega, mass, J,
                                   n=np.array([1, 0]),
                                   d=0,
                                   e=e)
    rect_world = rotation(rect_local, theta) + pos

    # right wall: x = width, n = -x
    pos, vel, omega = resolve_wall(rect_world, pos, vel, omega, mass, J,
                                   n=np.array([-1, 0]),
                                   d=-WIDTH,
                                   e=e)
    rect_world = rotation(rect_local, theta) + pos

    # top wall: y = 0, n = +y
    pos, vel, omega = resolve_wall(rect_world, pos, vel, omega, mass, J,
                                   n=np.array([0, 1]),
                                   d=0,
                                   e=e)
    rect_world = rotation(rect_local, theta) + pos

    # bottom wall: y = height, n = -y
    pos, vel, omega = resolve_wall(rect_world, pos, vel, omega, mass, J,
                                   n=np.array([0, -1]),
                                   d=-HEIGHT,
                                   e=e)
    rect_world = rotation(rect_local, theta) + pos

    screen.fill(BLACK)
    polygon(screen, rect_world)
    pygame.display.update()

pygame.quit()
