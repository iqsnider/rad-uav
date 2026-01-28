import pygame

WIDTH = 360
HEIGHT = 480
FPS = 30

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class UAVUtils(object):
    """
    Initializes a pygame window
    """

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Simple drone simulation")
        self.clock = pygame.time.Clock()

    def initialize(self):
        """
        Starts game loop
        """
        running = True
        while running:
            # make loop run at the same speed all the time
            self.clock.tick(FPS)

            for event in pygame.event.get():
                # listening for exit button
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill(BLACK)
            pygame.display.flip()

        pygame.quit()
