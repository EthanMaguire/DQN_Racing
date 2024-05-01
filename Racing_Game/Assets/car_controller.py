import pygame



class Controller:
    def __init__(self, car):
        self.car = car
        self.use_model = False

        # Movement variables
        self.max_speed = 500
        self.acceleration = 150
        self.braking_accel = 400
        self.turn_power = 250

    def update(self, dt, action=None):
        if not self.use_model:  # Player driving
            keys = pygame.key.get_pressed()  # Get the state of the keyboard
            parsed_actions = [0, 0, 0, 0]  # [W, S, A, D]
            # Accelerate
            if keys[pygame.K_w]:
                parsed_actions[0] = 1
            if keys[pygame.K_s]:
                parsed_actions[1] = 1
            if keys[pygame.K_a]:
                parsed_actions[2] = 1
            if keys[pygame.K_d]:
                parsed_actions[3] = 1

        else:
            parsed_actions = self.parse_model_actions(action)

        # Accelerate
        if parsed_actions[0] > 0.5:
            self.car.speed += self.acceleration * dt
            if self.car.speed > self.max_speed:
                self.car.speed = self.max_speed
        if parsed_actions[1] > 0.5:
            self.car.speed -= self.braking_accel * dt
            if self.car.speed <= 0:
                self.car.speed = 0
        if parsed_actions[2] > 0.5:
            self.car.angle += self.turn_power * dt
        if parsed_actions[3] > 0.5:
            self.car.angle -= self.turn_power * dt

    @staticmethod
    def parse_model_actions(action):
        parsed_actions = [0, 0, 0, 0]  # [W, S, A, D]
        if action == 0:  # accelerate
            parsed_actions[0] = 1
        elif action == 1:  # Brake
            parsed_actions[1] = 1
        elif action == 2:  # Left
            parsed_actions[2] = 1
        elif action == 3:  # Right
            parsed_actions[3] = 1
        elif action == 4:  # Forward + left
            parsed_actions[0] = 1
            parsed_actions[2] = 1
        elif action == 5:  # Forward + right
            parsed_actions[0] = 1
            parsed_actions[3] = 1
        elif action == 6:  # brake + left
            parsed_actions[1] = 1
            parsed_actions[2] = 1
        elif action == 7:  # brake + right
            parsed_actions[1] = 1
            parsed_actions[3] = 1

        return parsed_actions
