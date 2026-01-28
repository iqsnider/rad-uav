import pygame
import numpy as np

WIDTH = 360
HEIGHT = 480
FPS = 60

GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


def rotation(points, theta):
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]])
    return points @ R.T


def polygon(screen, pts):
    pygame.draw.polygon(screen, GREEN, pts.tolist(), 5)


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("simple drone simulation")
clock = pygame.time.Clock()

a = 25
b = 100

rect_local = np.array([[-b,  a],
                       [-b, -a],
                       [b, -a],
                       [b,  a],
                       ], dtype=float)

pos = np.array([WIDTH/2, HEIGHT/2], dtype=float)
vel = np.array([0, 0], dtype=float)

g = 900  # px/s^2
mass = 2
restitution = 0.7
theta = 0
omega = 1
I = (1/12)*mass*((2*b)**2+(2*a)**2)  # rectangle inertia about center

running = True
while running:
    dt = clock.tick(FPS)/1000

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # "dynamics"
    force = np.array([0, 0], dtype=float)
    force[1] += mass*g
    acc = force/mass

    vel += acc*dt
    pos += vel*dt

    # "control"
    theta += omega*dt

    # screen orientation
    rect_world = rotation(rect_local, theta) + pos

    min_xy = rect_world.min(axis=0)
    max_xy = rect_world.max(axis=0)

    # collisions
    if min_xy[0] < 0:
        pos[0] += -min_xy[0]
        vel[0] *= -restitution
    if max_xy[0] > WIDTH:
        pos[0] -= (max_xy[0] - WIDTH)
        vel[0] *= -restitution
    if min_xy[1] < 0:
        pos[1] += -min_xy[1]
        vel[1] *= -restitution
    if max_xy[1] > HEIGHT:
        pos[1] -= (max_xy[1] - HEIGHT)
        vel[1] *= -restitution

    rect_world = rotation(rect_local, theta) + pos

    screen.fill(BLACK)
    polygon(screen, rect_world)
    pygame.display.flip()

pygame.quit()
