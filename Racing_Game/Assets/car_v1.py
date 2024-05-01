import pygame
import math
import numpy as np

from Racing_Game.Assets import car_controller
from Racing_Game.Assets import collision



def get_rect(corners):  # returns [x, y, w, h] from Vector2[ul, br]
    return [corners[0].x, corners[0].y, abs(corners[0].x - corners[1].x), abs(corners[0].y - corners[1].y)]


class CarV1:
    def __init__(self, position, angle=0, color='red', size=0.25, speed=0):
        self.spawn_params = [position, angle, speed]
        self.position = pygame.Vector2(position)
        self.angle = angle
        self.speed = speed
        self.color = color
        self.sprite_size = size
        self.controller = car_controller.Controller(self)  # Attach a controller

        # Hitbox data
        self.size = pygame.Vector2(size * 100, size * 150)
        self.hitbox = [pygame.Vector2(), pygame.Vector2()]  # Top Left and Bottom Right
        self.update_hitbox()

        # Collision Management
        self.collide_flag = False
        self.draw_hitbox = False
        self.hitbox_color = (0, 255, 0)
        self.collide_color = (255, 0, 0)

        # Load and resize the sprite
        self.sprite = pygame.image.load("F:/PythonProjects_2024/MachineLearningTesting/Racing_Game/Assets/Sprites/CarV1.png")
        self.sprite = pygame.transform.scale_by(self.sprite, self.sprite_size)
        self.im_center = [self.sprite.get_size()[0] / 2, self.sprite.get_size()[1] / 2]

    def draw(self, surface):
        # rotate the sprite and place it so that it remains centered where an un-rotated version would be
        rotated = pygame.transform.rotate(self.sprite, self.angle)
        rotated_center = [rotated.get_size()[0] / 2, rotated.get_size()[1] / 2]
        center_diff = [self.im_center[0] - rotated_center[0], self.im_center[1] - rotated_center[1]]
        surface.blit(rotated, (self.position.x + center_diff[0], self.position.y + center_diff[1]))

        # Draw hitbox for debug
        if self.draw_hitbox:
            rot_hitbox = self.update_hitbox()
            rect = get_rect(rot_hitbox)
            if self.collide_flag:
                pygame.draw.rect(surface, self.collide_color, rect)
            else:
                pygame.draw.rect(surface, self.hitbox_color, rect)

    def update(self, dt):
        # Update position
        self.position.y -= self.speed * math.cos(math.radians(self.angle)) * dt
        self.position.x -= self.speed * math.sin(math.radians(self.angle)) * dt
        self.update_hitbox()

    def update_hitbox(self):  # returns points of the rotated hitbox
        ul = pygame.Vector2(self.position.x - self.size.x / 2, self.position.y - self.size.y / 2)
        br = pygame.Vector2(self.position.x + self.size.x / 2, self.position.y + self.size.y / 2)
        self.hitbox = [ul, br]
        rot_hitbox = self.hitbox.copy()
        for i, pt in enumerate(rot_hitbox):
            # Shift the center point to the origin
            temp_pt = pygame.Vector2(pt.x - self.position.x, pt.y - self.position.y)
            # Transform to polar coordinates
            temp_pt = collision.cart2pol(temp_pt)
            # Rotate by angle
            temp_pt = pygame.Vector2(temp_pt.x, temp_pt.y - math.radians(self.angle))
            # convert back to cartesian
            temp_pt = collision.pol2cart(temp_pt)
            # unshift center from origin
            rot_hitbox[i] = pygame.Vector2(temp_pt.x + self.position.x, temp_pt.y + self.position.y)

        return rot_hitbox

    def respawn(self):
        self.position = self.spawn_params[0]
        self.angle = self.spawn_params[1]
        self.speed = self.spawn_params[2]
        self.collide_flag = False
        self.finish = False
        self.distance_traveled = 0
        self.reward = 0
