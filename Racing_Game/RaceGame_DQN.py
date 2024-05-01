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
Screen_Bounds = pygame.Vector2(1280, 720)
screen = pygame.display.set_mode((1280, 720))
Delta_Time = 0.005

Env = racing_env.raceGame(use_model=True)  # Init raceGame environment object

# Training Loop
Draw_Training = True
num_episodes = 10000
max_steps = 500
rewards = np.zeros(500)
global_count = 0

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    max_steps += 1
    Env.reset_env()
    state_params = Env.get_state()
    state = torch.tensor(state_params, dtype=torch.float32, device=Car_Trainer.device).unsqueeze(0)
    for t in count():
        action = Car_Trainer.select_action(state)  # Come up with an action
        observation, reward, terminated, finished = step_game(action, draw=Draw_Training)  # Run one frame of the simulation
        reward = torch.tensor([reward], device=Car_Trainer.device)

        # Reward Tracking
        global_count += 1
        rewards[global_count % 500] = reward
        if global_count % 500 == 0:
            print("Episodes Complete: " + str(i_episode - 1))
            print("Average reward: " + str(rewards.mean()))

        done = terminated or finished
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=Car_Trainer.device).unsqueeze(0)

        Car_Trainer.memory.push(state, action, next_state, reward)  # Store the transition in memory
        state = next_state  # Move to the next state
        Car_Trainer.optimize_model()  # Perform one step of the optimization (on the policy network)
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = Car_Trainer.target_net.state_dict()
        policy_net_state_dict = Car_Trainer.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = (policy_net_state_dict[key]*Car_Trainer.TAU +
                                          target_net_state_dict[key] * (1 - Car_Trainer.TAU))
        Car_Trainer.target_net.load_state_dict(target_net_state_dict)

        if done or t > max_steps:
            Car_Trainer.episode_durations.append(t + 1)
            break
