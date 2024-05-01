import numpy as np
import pygame
import numpy
import math
import time
import torch
from itertools import count

from Assets import racing_env


# pygame setup
pygame.init()
Screen_Bounds = pygame.Vector2(1500, 1000)
screen = pygame.display.set_mode(Screen_Bounds)
Delta_Time = 0.005


Env = racing_env.raceGame(Screen_Bounds)  # Init raceGame environment object

while 1:
    Env.step(Delta_Time, draw=True, screen=screen)