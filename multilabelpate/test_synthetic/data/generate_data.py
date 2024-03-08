import numpy as np
import math
import matplotlib.pyplot as plt

'''
Generate data according to MLdatagen using hyperspheres (https://dl.acm.org/doi/10.1016/j.entcs.2014.01.025)
'''


def generate_small_hyperspheres(r_min, r_max, q, M_rel):
    '''
    Generate hyperspheres inside big hypersphere at origin with radius=1. 
    Each Hypersphere hs_i is characterized by the center C_i = (c_i1, ... ,c_iM_rel) and radius r_i
    '''
    spheres = []
    for i in range(q):
        #get Radius
        r_i = r_min + (r_max - r_min) * np.random.random()
        max_C = (1 - r_i)
        min_C = - (1 - r_i)
        #Get center coordinates
        C_i = np.zeros(M_rel)
        #Need to randomly go through indizes of center coordinates to avoid determinism
        indices_center = np.random.permutation(M_rel)
        for j in indices_center:
            c_ij = min_C + (max_C - min_C) * np.random.random()
            C_i[j] = c_ij
            #update min_C, max_C according to Equation 13
            cond = (1.0 - r_i) ** 2 - np.sum(np.square(C_i))
            min_C = - math.sqrt(cond) if cond>= 0 else 0
            max_C = math.sqrt(cond) if cond>= 0 else 0
        hs_i = (r_i, C_i)
        spheres.append(hs_i)
    return spheres


def test_spheres(points, spheres):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("auto")

    for i in range(len(spheres)):
        r, center = spheres[i]
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = r * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = r * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        ax.plot_wireframe(x, y, z, color="r")

    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color="b")

    N = len(points)
    xs = np.zeros(N)
    ys = np.zeros(N)
    zs = np.zeros(N)
    for i in range(N):
        xs[i] = points[i][0]
        ys[i] = points[i][1]
        zs[i] = points[i][2]

    ax.scatter(xs, ys, zs)

    plt.show()

def generate_points_for_sphere(N_i, sphere):
    '''
    Generates Points in the big hypersphere.
    Constraint: They need to lie at least in one smaller hypersphere
    --> Generate Points in small hyperspheres according to their volume
    N_i = round(f*r_i) and 
    '''
    r_i = sphere[0]
    C_i = sphere[1]
    M_rel = len(C_i)
    #Get points in Hypersphere
    points = np.zeros((N_i, M_rel))
    for k in range(N_i):
        indices_center = np.random.permutation(M_rel)
        for l in range(len(indices_center)):
            j = indices_center[l] 
            #minX,maxX according to equation 16
            cond = r_i**2 - np.sum(np.square(points[k] - C_i)[indices_center[:l]])
            minX = C_i[j] - math.sqrt(cond) if cond >= 0 else C_i[j]
            maxX = C_i[j] + math.sqrt(cond) if cond >= 0 else C_i[j]
            x_kj = minX + np.random.random()*(maxX-minX)
            points[k][j] = x_kj

    return points



def generate_Points(N, spheres, M_irr, M_red, noise, plot=False):
    M_rel = len(spheres[0][1])
    r_sum = 0
    for i in range(len(spheres)):
        r_sum += spheres[i][0]
    f = float(N) / r_sum

    #Get Points laying in each small Hypersphere
    points = np.array([])
    for i in range(len(spheres)):
        N_i = round(f*spheres[i][0])
        points_i = generate_points_for_sphere(N_i, spheres[i])
        points = np.vstack((points, points_i)) if points.size else points_i
    N = points.shape[0]
    if plot:
        test_spheres(points, spheres)
    labels = generate_labels(points, spheres, noise)

    #Get redundant and irrelevant features
    indices_red = np.random.randint(low=0,high=M_rel, size=M_red)
    points_red = points[:,indices_red]
    points_irr = np.random.random((N, M_irr))

    #Combine randomly into Array
    all_indices = np.random.permutation(M_rel+M_irr+M_red)
    indices_rel = all_indices[:M_rel]
    indices_irr = all_indices[M_rel:M_rel+M_irr]
    indices_red = all_indices[M_rel+M_irr:]
    combined_points = np.zeros((N,M_rel+M_irr+M_red))
    combined_points[:,indices_rel] = points
    combined_points[:,indices_irr] = points_irr
    combined_points[:,indices_red] = points_red

    return combined_points, labels


def generate_labels(points, spheres, noise):
    N = len(points)
    q = len(spheres)
    labels = np.zeros((N,q))
    for k in range(N):
        label = np.zeros(q)
        for j in range(q):
            r_j = spheres[j][0]
            C_j = spheres[j][1]
            x_k = points[k]
            cond = np.abs(x_k - C_j) < r_j
            #cond must be true for all M entries of x_k
            if np.sum(cond) >= len(x_k):
                labels[k][j] = 1
            #with probability noise flip the label
            if np.random.rand() <= noise:
                labels[k][j] = 1 - labels[k][j]
    return labels


def generate_data(r_min, r_max, q, M_rel, M_irr, M_red, N, noise, seed):
    np.random.seed(seed)
    spheres = generate_small_hyperspheres(r_min,r_max,q,M_rel)
    points, labels = generate_Points(N, spheres, M_irr, M_red, noise, plot=False)
    return points, labels


