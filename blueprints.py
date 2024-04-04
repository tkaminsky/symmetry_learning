import numpy as np


####################################################################################################
# Some functions for generating random blueprints, as well as a few fixed blueprints
####################################################################################################

# Returns a nested shape of num_objects nested regular polygons, choosing one to be the target
def generate_nested_regular_polygon_blueprint(
    num_objects, target_idx=None, size=None
):
    blueprint = []
    if size is None:
        size = np.random.uniform(0.1, 1.0)
    if target_idx is None:
        target_idx = np.random.randint(num_objects)

    curr_size = size
    curr_center = np.array([0, 0])

    # Generate random positive numbers that sum to 1
    nums = np.random.rand(num_objects)
    # Append 0 to the beginning and 1 to the end
    nums = np.append(nums, 1)
    nums /= np.sum(nums)

    for i in range(num_objects):
        curr_blueprint = {}
        curr_blueprint["type"] = "regular_polygon"
        curr_blueprint["N"] = np.random.randint(3, 10)
        curr_blueprint["R"] = size - size * (sum(nums[:i]))
        curr_blueprint["theta"] = np.random.uniform(0, 2 * np.pi)
        curr_blueprint["target"] = i == target_idx
        curr_blueprint["center"] = curr_center.copy()

        blueprint.append(curr_blueprint)

        # Move the center at most R away from the current center
        max_dist = curr_blueprint["R"] - (size - size * (sum(nums[: i + 1])))
        curr_center = curr_center + np.random.uniform(-max_dist, max_dist, 2)

    return blueprint


regular_polygon_blueprint = [
    {
        "type": "regular_polygon",
        "N": 5,
        "R": 1.0,
        "theta": 0,
        "center": [0, 0],
        "target": False,
    },
    # {'type': 'regular_polygon', 'N': 5, 'R': .7, 'theta': np.pi/2, 'center': [0, 0], 'target': False},
    # {'type': 'regular_polygon', 'N': 5, 'R': .4, 'theta': 0, 'center': [1, 0], 'target': True}
]

# Make a numpy array with points forming an L-shape
pointsL = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]).T
# Make a numpy array with points forming a T
pointsW = np.array([[-1, -2], [-0.5, 1], [0.2, 1], [0.2, 0], [1, 0], [0, -2]]).T

strange_polygon_blueprint = [
    {"type": "polygon", "points": pointsL, "target": False},
    {"type": "polygon", "points": pointsW, "target": False},
    {
        "type": "regular_polygon",
        "N": 5,
        "R": 0.4,
        "theta": 0,
        "center": [1, 0],
        "target": True,
    },
]
