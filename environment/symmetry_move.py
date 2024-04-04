import gym
import numpy as np
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
from gym import spaces
from helpers.algos import nested_polygon_polygon_distance

from objects.nested_shape import NestedShape
from blueprints import generate_nested_regular_polygon_blueprint

####################################################################################################
# Symmetry move is a depricated environment that was used to test some initial functionality.
####################################################################################################

ACTION_TO_DIR = {
    0: np.array([0, 1]),
    1: np.array([1, 0]),
    2: np.array([0, -1]),
    3: np.array([-1, 0]),
    4: 'Clockwise',
    5: 'CounterClockwise'
}


class SymmetryMoveEnv(gym.Env):
    def __init__(self, o1_bp, o2_bp, render_mode="human"):
        self.o1_bp = o1_bp
        self.o2_bp = o2_bp

        self.x_min = 0
        self.x_max = 10
        self.y_min = 0
        self.y_max = 10

        self.horizon = 1000

        self.v = 0.3
        self.w = 1/20
        self.tolerance = self.v / 2 + self.w / 2

        H = W = 640

        self.H = H

        self.t = 0

        self.o1 = NestedShape(self.o1_bp)
        self.o2 = NestedShape(self.o2_bp)

        self.render_mode = render_mode

        self.action_space = spaces.Discrete(6)

        # Observation space is H x W x 3 (RGB) image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(H, W, 3), dtype=np.uint8
        )

        # Initialize the pygame screen
        pygame.init()
        if render_mode == "human":
            self.screen = pygame.display.set_mode((H, W))
        else:
            self.screen = pygame.display.set_mode((H,W), flags=pygame.HIDDEN)
        # If render mode is rgb array, make the screen invisible

        # Make the screen bigger   
        # pygame.transform.scale(self.screen, (H* 100, W * 100))

        # Initialize the clock
        self.clock = pygame.time.Clock()

        # Initialize the font
        self.font = pygame.font.SysFont("Arial", 16)

        self.target = self.o1.get_target()
        self.target_count = len(self.target.shapes)

        self.target_colors = [(1, .1 * i, .1 * i) for i in range(10)]

        self.reset()


    def get_observation(self):
        # Render the scene
        self._render()

        # Get the pixels from the pygame screen
        pixels = pygame.surfarray.array3d(self.screen)

        # Convert from (H, W, C) to (C, H, W)
        pixels = np.transpose(pixels, (2, 0, 1))

        return pixels
    

    def draw_target_in_corner(self, screen):
        # Draw a rectangle in the top right corner of size H/8 x H/8
        pygame.draw.rect(screen, (0, 0, 0), (self.H - (self.H // 8), self.H - (self.H // 8), self.H, self.H), 2)

        size = self.x_max - self.x_min

        # Draw each shape in the target nested-object inside the rectangle
        target = self.o2.get_target()
        # Move the target to the top right corner
        # print(size - size // 16)
        target.move_to(np.array([size - size / 16, size - size / 16]))
        # Scale the target points to be H/8 x H/8
        target_size = target.R

        center = target.center.copy().reshape((2,1))

        # Draw a red circle at the center of the target

        #pygame.draw.circle(screen, (255, 0, 0), center, self.H // 16)

        for i in range(len(target.shapes)):
            small_points = (target.shapes[i].points.copy() - center) * (size / 16) / (target_size) + center
            target.shapes[i].points = small_points.copy()
            target.shapes[i].color = self.target_colors[i]
            
        # print("Drawing a point centered at")
        # print(center)
        target.draw(screen, size=self.H)
    
    def _render(self, mode="human"):
        # Clear the screen
        self.screen.fill((255, 255, 255))

        # Draw the shapes
        self.o1.draw(self.screen, size = self.H)
        self.o2.draw(self.screen, size = self.H)

        # Draw the text
        text = self.font.render("Symmetry Shift", True, (255, 255, 255))
        self.screen.blit(text, (5, 5))

        # Draw the target in the top right corner
        self.draw_target_in_corner(self.screen)

        # Flip the screen
        # pygame.display.flip()
        self.screen.blit(pygame.transform.flip(self.screen, False, True), (0,0))

        # Update the display
        # if mode == "human":
        pygame.display.update()


        self.clock.tick(60)

    def render(self, mode="human"):
        if mode == "human":
            self._render()
        elif mode == "rgb_array":
            return self.get_observation()

    def reset(self):
        # Move the objects to random locations between x_min and x_max
        # and between y_min and y_max

        x_bounds = (self.x_min + self.o1.R, self.x_max - self.o1.R)
        y_bounds = (self.y_min + self.o1.R, self.y_max - self.o1.R)

        o1_delta = [
            np.random.choice(np.arange(x_bounds[0], x_bounds[1], self.v)),
            np.random.choice(np.arange(y_bounds[0], y_bounds[1], self.v))
        ]

        o2_delta = [
            np.random.choice(np.arange(x_bounds[0], x_bounds[1], self.v)),
            np.random.choice(np.arange(y_bounds[0], y_bounds[1], self.v))
        ]

        self.o1.move_to(
            o1_delta
        )
        self.o2.move_to(
            o2_delta
        )

        o1_theta = np.random.choice(np.arange(0, 2 * np.pi, self.w))
        o2_theta = np.random.choice(np.arange(0, 2 * np.pi, self.w))

        o1_current_theta = self.o1.theta
        self.o1.rotate(-o1_current_theta)
        o2_current_theta = self.o2.theta
        self.o2.rotate(-o2_current_theta)

        self.o1.rotate(o1_theta)
        self.o2.rotate(o2_theta)

        print("I moved the objects to:")
        print(self.o1.center)
        print(self.o2.center)

        self.target = self.o1.get_target()

        self.t = 0

        return self.get_observation(), {}
    
    def compute_reward(self):
        # Calculate the polygon-polygon distance between the two targets
        factor = [1,1]

        t1 = self.o1.get_target()
        t2 = self.o2.get_target()

        t1_center = t1.center
        t2_center = t2.center

        dist = np.linalg.norm(t1_center - t2_center)

        t1_theta = self.o1.theta
        t2_theta = self.o2.theta

        # Find the angular difference between the two thetas
        theta_diff = min(np.abs(t1_theta - t2_theta), 2 * np.pi - np.abs(t1_theta - t2_theta))
        # theta_diff = np.abs(t1_theta - t2_theta)

        # print("Distance is", dist)
        # print("Theta difference is", theta_diff)

        # dist = nested_polygon_polygon_distance(t1, t2)
        dist = factor[0] * dist + factor[1] * theta_diff
        return -dist

    def step(self, action):
        self.t += 1
        # Action is an integer between 0 and 3

        # Move o1 in the direction specified by action
        if action in [0,1,2,3]:
            # print("Action")
            self.o1.translate(ACTION_TO_DIR[action] * self.v)
        elif action in [4,5]:
            if ACTION_TO_DIR[action] == 'Clockwise':
                self.o1.rotate(-np.pi * self.w)
            else:
                self.o1.rotate(np.pi * self.w)

        # Get the observation
        obs = self.get_observation()

        # Compute the reward
        reward = self.compute_reward()
        # print("Reward is", reward)

        # Check if the episode is over
        done = False

        if -reward < self.tolerance:
            done = True

        # Check if the episode is over
        if self.t >= self.horizon:
            done = True



        return obs, reward, done, {}


