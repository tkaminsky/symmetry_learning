import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry.polygon import Polygon as SPoly
from helpers.algos import welzl
import pygame


class Polygon(object):
    def __init__(self, blueprint):

        # Set color
        if "color" in blueprint.keys():
            self.color = blueprint["color"]
        else:
            self.color = np.random.random(size=3).tolist()

        # Set points
        self.points = np.array(blueprint["points"]).astype(float)

        if self.points.shape[0] != 2:
            self.points = self.points.T

        self.N = self.points.shape[1]

        self.center_type = "ortho"

        center, radius = self.get_smallest_containing_circle()

        self.center = center
        self.R = radius

    def get_smallest_containing_circle_com(self):
        poly = SPoly(self.points.T.tolist())
        # center = np.mean(self.points, axis=1).reshape(2, 1)
        center = poly.centroid.coords[0]
        # Turn a tuple into a numpy array
        center = np.array(center)
        # Calculate the largest distance from the center of mass to a point
        radius = 0
        for point in self.points.T:
            radius = max(radius, np.linalg.norm(point - center))
        return (center, radius)

    def get_smallest_containing_circle(self):
        if self.center_type == "com":
            return self.get_smallest_containing_circle_com()
        elif self.center_type == "ortho":
            return self.get_smallest_containing_circle_ortho()
        else:
            raise ValueError("Invalid center type: " + self.center_type)

    def get_smallest_containing_circle_ortho(self):
        point_list = self.points.copy().T.tolist()

        # Randomly permute the points
        np.random.shuffle(point_list)
        center, radius = welzl(point_list, [])
        return center, radius
    
    def plot(self, ax=None, color=None):
        if ax is None:
            fig, ax = plt.subplots()
            plt.xlim(self.center[0] - self.R - 0.5, self.center[0] + self.R + 0.5)
            plt.ylim(self.center[1] - self.R - 0.5, self.center[1] + self.R + 0.5)

        if color is None:
            color = self.color

        ax.add_patch(mpatches.Polygon(self.points.T, color=color))

        # Plot smallest containing circle
        circle = plt.Circle(self.center, self.R, color="r", fill=False)
        ax.add_artist(circle)

        return ax

    def rotate(self, theta, center=None):
        if center is None:
            center = self.center
        rot_mat = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )
        # Rotate points about center
        self.points = (
            rot_mat @ (self.points - center[:, np.newaxis]) + center[:, np.newaxis]
        )

        # rotate self.center about center
        self.center = rot_mat @ (self.center - center) + center

    def translate(self, delta):
        self.points[0, :] += delta[0]
        self.points[1, :] += delta[1]
        self.center += delta

    def draw(self, screen, size):
        # Scale the points by size
        points = self.points.copy()

        points = points * size / 10

        color_as_int = tuple([int(255 * c) for c in self.color])

        pygame.draw.polygon(screen, color_as_int, points.T.tolist())

        return screen


class RegularPolygon(Polygon):
    def __init__(self, blueprint):
        N = blueprint["N"]
        R = blueprint["R"]
        if "center" in blueprint.keys():
            c = blueprint["center"]
        else:
            c = np.array([0, 0])

        if "theta" in blueprint.keys():
            theta = blueprint["theta"]
        else:
            theta = 0

        if "color" in blueprint.keys():
            self.color = blueprint["color"]
        else:
            self.color = np.random.random(size=3).tolist()

        self.N = N
        self.points = np.zeros((N, 2))
        self.points[:, 0] = R * np.cos(np.linspace(0, 2 * np.pi, N, endpoint=False))
        self.points[:, 1] = R * np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))

        self.points[:, 0] += c[0]
        self.points[:, 1] += c[1]

        poly_blueprint = {'points': self.points.T.tolist(), 'color': self.color}

        super().__init__(poly_blueprint)

        if theta != 0:
            self.rotate(theta)
