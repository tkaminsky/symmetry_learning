import numpy as np

def generate_valid_regular_polygon_blueprints():
    sizes = np.array([.1, .2, .5, 1., 1.5, 2., 2.5, 3.])

    target_idx = np.random.randint(len(sizes) - 4) + 2
    target_idx = 3
    target_N = np.random.randint(3, 10)

    o1_target_loc = 1

    o1_indices = [2,3,4]

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

    return o1_bp, o2_bp

triangle_1 = [{'type':'regular_polygon', 'N':5, 'R':2, 'theta':0, 'center':[5, 5],  'target':True}]
triangle_2 = [{'type':'regular_polygon', 'N':5, 'R':2, 'theta':0, 'center':[5, 5],  'target':True}]
triangle_bp = (triangle_1, triangle_2)


L_1 = [{'type':'polygon', 'points':[[0,1],[1,1], [1,0],[3,0], [3,-1], [0,-1],[0,1]], 'center':[5, 5],  'target':True}]
L_2 = [{'type':'polygon', 'points':[[0,1],[1,1], [1,0],[3,0], [3,-1], [0,-1],[0,1]], 'center':[5, 5],  'target':True}]
L_bp = (L_1, L_2)