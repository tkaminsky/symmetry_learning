import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class polygon:
    def __init__(self, N=4, R=1.0):
        self.N = N
        self.R = R
        self.theta = 0
        self.verts = np.zeros((N, 2))
        self.verts[:, 0] = R * np.cos(
            self.theta + np.linspace(0, 2 * np.pi, N, endpoint=False)
        )
        self.verts[:, 1] = R * np.sin(
            self.theta + np.linspace(0, 2 * np.pi, N, endpoint=False)
        )
        self.verts = self.verts.T
        self.center = np.zeros((2, 1))

    def draw(self, canvas=None):
        if canvas == None:
            print(self.verts.shape)
            print(self.verts)
            # Greate a new figure
            rot_mat = np.array(
                [
                    [np.cos(self.theta), -np.sin(self.theta)],
                    [np.sin(self.theta), np.cos(self.theta)],
                ]
            )
            transformed_points = rot_mat @ self.verts + self.center

            fig, ax = plt.subplots()
            ax.add_patch(mpatches.Polygon(transformed_points.T, color=[1, 0.8, 0.8]))

            # Set aspect ratio to 1
            ax.set_aspect("equal")

            plt.xlim(-self.R - 0.5, self.R + 0.5)
            plt.ylim(-self.R - 0.5, self.R + 0.5)

            # plt.plot(transformed_points[0, :], transformed_points[1, :])
            # plt.show()

    def get_smallest_containing_circle(self):
        return (self.center, self.R)

    def get_largest_contained_circle(self):
        r = self.R * np.sin((self.N - 2) * np.pi / (2 * self.N))
        return (self.center, r)

    def draw_with_circles(self, canvas=None):
        if canvas == None:
            self.draw()

            (center, r) = self.get_smallest_containing_circle()
            print("Center: ", center.shape)
            print("Radius: ", r)
            circle_sm = plt.Circle(center, r, color="r", fill=False)
            plt.gca().add_patch(circle_sm)
            (center, r) = self.get_largest_contained_circle()
            print(center.shape)
            print(r)
            circle_lg = plt.Circle(center, r, color="b", fill=False)
            plt.gca().add_patch(circle_lg)

            # plt.plot(transformed_points[0, :], transformed_points[1, :])
            plt.show()
