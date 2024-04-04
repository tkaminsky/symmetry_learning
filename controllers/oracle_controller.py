import numpy as np


def oracle(env):
    o1_target = env.o1.get_target()
    o2_target = env.o2.get_target()

    o1_pos = o1_target.center
    o2_pos = o2_target.center

    o1_theta = o1_target.theta
    o2_theta = o2_target.theta

    if np.linalg.norm(o1_pos - o2_pos) >= 1e-4:
        # Move o1 to o2
        delta = o2_pos - o1_pos
        # Find the best action between [0,1], [0,-1], [1,0], [-1,0]
        best_action = None
        best_dist = np.inf
        for i, action in enumerate([[0, 1], [1, 0], [0, -1], [-1, 0]]):
            dist = np.linalg.norm(delta - action)
            if dist < best_dist:
                best_dist = dist
                best_action = i
        
        return best_action
    elif np.abs(o1_theta - o2_theta) >= 1e-4:
        # Rotate o1 to o2
        delta = o2_theta - o1_theta
        if delta > 0:
            return 4
        else:
            return 5
        
def angle_diff(a1, a2):
    # Return the minimum difference between the two angles
    delta = a2 - a1
    while delta > np.pi:
        delta -= 2 * np.pi
    while delta < -np.pi:
        delta += 2 * np.pi
    return delta
        
def oracle_rev(env):
    o1_target = env.o1.get_target()
    o2_target = env.o2.get_target()

    o1_pos = o1_target.center
    o2_pos = o2_target.center

    o1_theta = o1_target.theta
    o2_theta = o2_target.theta

    o1_theta = env.o1.theta
    o2_theta = env.o2.theta

    best_action = None

    if np.linalg.norm(o1_pos - o2_pos) >= env.v / 2:
        # Move o1 to o2
        delta = o2_pos - o1_pos
        # Find the best action between [0,1], [0,-1], [1,0], [-1,0]
        best_action = None
        best_dist = np.inf
        for i, action in enumerate([[0, 1], [1, 0], [0, -1], [-1, 0]]):
            dist = np.linalg.norm(delta - action)
            if dist < best_dist:
                best_dist = dist
                best_action = i
    elif np.abs(o1_theta - o2_theta) > env.w / 2:
        if o1_theta > o2_theta:
            best_action = 4
        else:
            best_action = 5
        
    return best_action


def oracleController(env):
    obs, _ = env.reset()

    while True:
        action = oracle_rev(env)
        obs, _, done, _ = env.step(action)
        if done:
            obs, _ = env.reset()
            done = False
