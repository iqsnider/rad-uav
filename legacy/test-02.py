import pygame
import random
import numpy as np

from pygame.locals import *

WIDTH = 360
HEIGHT = 480
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def polygon(x1, x2, x3, x4):
    """
    Draws a polygon from 4 2D coordinates
    """
    pygame.draw.polygon(screen, GREEN, [x1, x2, x3, x4], 5)


def rotation(rec_prev, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot_mat = np.array([[cos_theta, -sin_theta],
                        [sin_theta, cos_theta]])
    center = rec_prev.mean(axis=0)
    rec_centered = rec_prev - center
    rec_rot = rec_centered @ rot_mat

    return rec_rot + center


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("simple drone simulation")
clock = pygame.time.Clock()

running = True
a = 50
b = 100
x10 = [WIDTH/2 - b, HEIGHT/2 + a]
x20 = [WIDTH/2 - b, HEIGHT/2 - a]
x30 = [WIDTH/2 + b, HEIGHT/2 - a]
x40 = [WIDTH/2 + b, HEIGHT/2 + a]
rectangle = np.array([x10, x20, x30, x40])
theta = 0.01
while running:
    # make loop run at the same speed all the time
    clock.tick(FPS)

    for event in pygame.event.get():
        # listening for exit button
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)
    rectangle = rotation(rectangle, theta)
    x1 = rectangle[0]
    x2 = rectangle[1]
    x3 = rectangle[2]
    x4 = rectangle[3]

    polygon(x1, x2, x3, x4)
    pygame.display.update()

pygame.quit()
