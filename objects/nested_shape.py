from objects.shapes.polygon import Polygon, RegularPolygon
import numpy as np
import matplotlib.pyplot as plt
from helpers import welzl

VALID_SHAPES = ["polygon", "regular_polygon"]
COLORS = [
    [0.342, 0.456, 0.789],
    [0.123, 0.678, 0.901],
    [0.234, 0.567, 0.890],
    [0.345, 0.678, 0.789],
    [0.456, 0.789, 0.678],
    [0.567, 0.890, 0.567],
    [0.678, 0.901, 0.456],
    [0.789, 0.012, 0.345],
    [0.890, 0.123, 0.234],
    [0.901, 0.234, 0.123],
]


blueprint = {
    "type": "regular_polygon",
    "N": 4,
    "R": 1.0,
    "theta": 0,
    "center": [0, 0],
    "target": False,
}


class NestedShape:
    def __init__(self, blueprint):
        self.shapes = []
        self.target_indices = []
        self.blueprint = blueprint

        for item in blueprint:
            if item["type"] not in VALID_SHAPES:
                raise ValueError("Invalid shape type: " + item["type"])
            if item["type"] == "polygon":
                self.shapes.append(Polygon(item))
            elif item["type"] == "regular_polygon":
                self.shapes.append(RegularPolygon(item))

            if item["target"]:
                # Append the index of the shape to the targets list
                self.target_indices.append(len(self.shapes) - 1)

        c, r = self.get_smallest_containing_circle()
        self.center = c
        self.R = r

    def get_smallest_containing_circle(self):
        point_list = []

        for shape in self.shapes:
            point_list.extend(shape.points.copy().T.tolist())


        np.random.shuffle(point_list)

        center, radius = welzl(point_list, [])

        return (center, radius)

    def rotate(self, theta, center=None):
        if center is None:
            center = self.center

        for shape in self.shapes:
            shape.rotate(theta, center)

    def translate(self, delta):
        for shape in self.shapes:
            shape.translate(delta)
        
        self.center += delta

    def move_to(self, center):
        delta = center - self.center
        self.translate(delta)

    def draw(self, screen, size):
        for shape in self.shapes:
            shape.draw(screen, size)

    def get_target(self):
        # make a new nested shape with only the target shapes
        target_blueprints = []
        for i in self.target_indices:
            target_blueprints.append(self.blueprint[i])

        target_shape = NestedShape(target_blueprints)
        # Move target shape to current center
        target_shape.move_to(self.center)

        return target_shape

        

    def plot(self, ax=None):
        for i, shape in enumerate(self.shapes):
            ax = shape.plot(ax)

        # Plot smallest containing circle, as a thick dotted red line
        circle = plt.Circle(
            self.center, self.R, color="r", fill=False, linewidth=3, linestyle="dashed"
        )
        ax.add_artist(circle)

        # Change axis bounds
        ax.set_xlim(self.center[0] - self.R - 0.5, self.center[0] + self.R + 0.5)
        ax.set_ylim(self.center[1] - self.R - 0.5, self.center[1] + self.R + 0.5)
        return ax
