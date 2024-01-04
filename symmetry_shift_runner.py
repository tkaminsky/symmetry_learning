from environment.symmetry_shift import SymmetryShiftEnv
from objects.nested_shape import NestedShape
from objects.shapes.polygon import RegularPolygon
import numpy as np
import pygame
import matplotlib.pyplot as plt
from helpers import print_blueprint

sizes = np.array([.1, .2, .5, 1., 1.5, 2., 2.5, 3.])


target_idx = np.random.randint(len(sizes) - 4) + 2
target_idx = 3
target_N = np.random.randint(3, 10)

# o1_target_loc = np.random.randint(3)
o1_target_loc = 1

# o1_indices = [i + target_idx + o1_target_loc - 1 for i in range(3)]

o1_indices = [2,3,4]


# if o1_indices[0] < 0:
#     o1_indices = [0,1,2]
# elif o1_indices[-1] >= len(sizes):
#     o1_indices = [len(sizes) - 3, len(sizes) - 2, len(sizes) - 1]

o1_sizes = sizes.copy()[o1_indices]
o1_Ns = np.random.randint(3, 10, len(o1_sizes))
o1_Ns[o1_target_loc] = target_N

o1_bp = [
  {'type':'regular_polygon', 'N':o1_Ns[i], 'R':o1_sizes[i], 'theta':0, 'center':[5, 5],  'target':(i==o1_target_loc)} for i in range(len(o1_sizes))
]


# Reverse bp
o1_bp = o1_bp[::-1]

# o2_target_loc = np.random.randint(3)
o2_target_loc = 2

# o2_indices = [0, 1, 2] + target_idx + o2_target_loc - 1
# o2_indices = [i + target_idx + o2_target_loc - 1 for i in range(3)]

o2_indices = [1,2,3]

# if o2_indices[0] < 0:
#     o2_indices = [0,1,2]
# elif o2_indices[-1] >= len(sizes):
#     o2_indices = [len(sizes) - 3, len(sizes) - 2, len(sizes) - 1]


o2_sizes = sizes.copy()[o2_indices]
o2_Ns = np.random.randint(3, 10, len(o2_sizes))
o2_Ns[o2_target_loc] = target_N

o2_bp = [
  {'type':'regular_polygon', 'N':o2_Ns[i], 'R':o2_sizes[i], 'theta':0, 'center':[5, 5],  'target':(i==o2_target_loc)} for i in range(len(o2_sizes))
]

o2_bp = o2_bp[::-1]

print_blueprint(o2_bp)


env = SymmetryShiftEnv(o1_bp, o2_bp)

# fix, axs = plt.subplots(1, 3, figsize=(10, 5))

# o1_poly = RegularPolygon(o1_bp[o1_target_loc])

# o1_poly.plot(axs[2])
# axs[2].set_title("Target")

# axs[0].set_title("Object 1")
# axs[1].set_title("Object 2")

# o1 = NestedShape(o1_bp)
# o2 = NestedShape(o2_bp)

# o1.plot(axs[0])
# o2.plot(axs[1])

# # env.o1.plot(axs[0])
# # env.o2.plot(axs[1])

# axs[0].set_aspect("equal")
# axs[1].set_aspect("equal")
# axs[2].set_aspect("equal")

# plt.show()


env.reset()

# Do manual control loop with WASD keys using pygame

done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("THIS SHOULD NOT HAPPEN")
            done = True

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        obs, reward, done, info = env.step(0)
        print("reward: ", reward)
    if keys[pygame.K_d]:
        obs, reward, done, info = env.step(1)
        print("reward: ", reward)       
    if keys[pygame.K_s]:
        obs, reward, done, info = env.step(2)
        print("reward: ", reward)
    if keys[pygame.K_a]:
        obs, reward, done, info = env.step(3)
        print("reward: ", reward)

    if done == True:
        print("DONE")
        print(reward)
        done = False

    env.render()

    env.clock.tick(60)

    if done:
        break


