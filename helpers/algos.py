import numpy as np

def print_blueprint(blueprint):
    print("Printing blueprint:")
    for item in blueprint:
        print(item)


# Find the circle with 3 points P on the boundary
def trivial(P):
    if len(P) == 0:
        return None
    elif len(P) == 1:
        return P[0], 0
    elif len(P) == 2:
        P_arr = np.array(P)
        return (P_arr[0] + P_arr[1]) / 2, np.linalg.norm(P_arr[0] - P_arr[1]) / 2
    else:
        # Find the circumcenter of the triangle
        # p1, p2, p3 = P
        P_arr = np.array(P)
        p1 = P_arr[0,:]
        p2 = P_arr[1,:]
        p3 = P_arr[2,:]

        r1 = p1.T - p2.T
        r2 = p2.T - p3.T
        r3 = p1.T - p3.T

        # Append a 1 to the end of each row vector
        r1 = np.append(r1, 1)
        r2 = np.append(r2, 1)
        r3 = np.append(r3, 1)

        # Stack the row vectors into a matrix
        A = 2 * np.vstack((r1, r2, r3))
        b = np.array([p1.T@p1 - p2.T@p2, p2.T@p2 - p3.T@p3, p1.T@p1 - p3.T@p3])

        # Find the circumcenter of the triangle
        # A = np.array([[p1[0], p1[1], 1], [p2[0], p2[1], 1], [p3[0], p3[1], 1]])
        # b = np.array(
        #     [p1[0] ** 2 + p1[1] ** 2, p2[0] ** 2 + p2[1] ** 2, p3[0] ** 2 + p3[1] ** 2]
        # )
        x = np.linalg.solve(A, b)
        center = x[0:2]
        radius = np.linalg.norm(center - p1)
        return center, radius

# Finds the smallest enclosing circle of a set of points P
def welzl(P, R):
    if len(P) == 0 or len(R) == 3:
        return trivial(R)

    p = P[0]
    P = P[1:]
    # R = R.copy()
    # R.append(p)
    Disc = welzl(P, R)
    if Disc is not None:
        p_arr = np.array(p)
        D_arr = np.array(Disc[0])

        if np.linalg.norm(p_arr - D_arr) <= Disc[1]:
            return Disc
    R = R.copy()
    R.append(p)
    return welzl(P, R)


# Project u onto v
def proj(u, v):
    # Project u onto v
    return (np.dot(u, v) / np.sqrt(np.dot(v, v)))


# Find the distance from a point to a line
def point_line_distance(p, p1, p2):
    # Find the distance from a point to a line
    p = np.array(p)
    p1 = np.array(p1)
    p2 = np.array(p2)

    l = p2 - p1

    # If l is the zero vector, then the distance is just the distance to p1
    if np.linalg.norm(l) < 1e-7:
        return np.linalg.norm(p - p1)

    im = proj(p - p1, l)

    if im <= 0:
        return np.linalg.norm(p - p1)
    elif im >= np.linalg.norm(p2 - p1):
        return np.linalg.norm(p - p2)
    else:
        perp = (p - p1) - im * l / np.linalg.norm(l)
        return np.linalg.norm(perp)


def point_polygon_distance(p, P):
    # Find the distance from a point to a polygon
    dists = []
    for i in range(len(P)):
        p1 = P[i]
        p2 = P[(i + 1) % len(P)]
        dists.append(point_line_distance(p, p1, p2))
    return np.min(dists)

def polygon_polygon_distance(P, R):
    # Find the distance from each point to the polygon
    dists = []
    for p in P:
        dists.append(point_polygon_distance(p, R))
    return np.max(dists)

def nested_polygon_polygon_distance(nP1, nP2):
    # Find the distance from each point to the polygon
    num_polygons = len(nP1.shapes)
    dists = []
    for i in range(num_polygons):
        p1 = nP1.shapes[i]
        p2 = nP2.shapes[i]
        dists.append(polygon_polygon_distance(p1.points.T, p2.points.T))

    return np.max(dists)
        
