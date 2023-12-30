import numpy as np
from scipy.spatial import ConvexHull
import geometry
from matplotlib import pyplot as plt

def get_faces_from_vertices(vertices):
    hull = ConvexHull(vertices, qhull_options="Qc")
    faces = hull.simplices
    return faces

def generate_ring(n, lat, add_offset = False, add_center = False):
    deg_offset = 5
    if(add_offset):
        deg_offset += 360/(2*n)
    lons = np.linspace(deg_offset + 0, 360 + deg_offset, n, endpoint=False)
    lats = np.ones_like(lons) * (lat)
    xyz = geometry.lat_lon_to_XYZ(lats, lons)
    if(add_center):
        z = np.sin(np.deg2rad(lat))
        xyz = np.vstack(([0, 0, z + np.sign(z) * 1e-5], xyz))
    return xyz

def plot_solid(vertices, faces):
    fig = plt.figure(figsize=(8,5))
    ax = plt.axes(projection='3d')

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    ax.plot_trisurf(x, y, z, triangles=faces,
                        cmap='viridis', alpha=0.2, edgecolor='k')
    plt.show()

def create_icosahedron():
    t = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    vertices = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ])
    
    faces = np.array([
        [0, 11,  5], [0,  5,  1], [0,  1,  7], [0,  7, 10], [0, 10, 11],
        [1,  5,  9], [5, 11,  4], [11, 10,  2], [10,  7,  6], [ 7,  1,  8],
        [3,  9,  4], [3,  4,  2], [3,  2,  6], [3,  6,  8], [3,  8,  9],
        [4,  9,  5], [2,  4, 11], [6,  2, 10], [8,  6,  7], [9,  8,  1]
    ])
    
    return vertices, faces


def fibonacci_sphere(samples=1000):

    vertices = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        vertices.append((x, y, z))

    vertices =  np.array(vertices)
    faces = get_faces_from_vertices(vertices)
    
    return vertices, faces


def waterman_polyhedron(O, R):
    r = 0.5
    if(O == 1):
        offset = np.array([0, 0, 0])
    elif(O == 2):
        offset = np.array([1/2, 1/2, 0])
    elif(O == 3):
        offset = np.array([1/3, 1/3, 2/3])
    elif(O == 4):
        offset = np.array([1/2, 1/2, 1/2])
    elif(O == 5):
        offset = np.array([0, 0, 1/2])
    elif(O == 6):
        offset = np.array(1, 0, 0)
    else:
        raise ValueError(f"Origin {O} is unknown.")

    N = np.ceil(R * 2 / r).astype(np.int32)
    i = j = K = np.arange(-N, N, dtype = np.int32)

    idx_i, idx_j = np.meshgrid(np.arange(len(i)), np.arange(len(j)))
    I = i[idx_i.flatten()]
    J = j[idx_j.flatten()]

    X = r * (2 * J + I % 2)
    Y = I * r * np.sqrt(3)
    Z = I * 0

    A = np.stack((X, Y, Z)).T

   
    B = A + r * np.array([1, np.sqrt(3)/3, 0])
    C = A + r * np.array([0, 2 * np.sqrt(3)/3, 0])
    lattice = np.zeros((0,3))
    z = np.array([0, 0, 2 * np.sqrt(6)/3]) * r
    for k in K:
        if(k % 3 == 0):
            layer = A + k * z
        elif(k % 3 == 1):
            layer = B + k * z
        elif(k % 3 == 2):
            layer = C + k * z
        lattice = np.vstack((lattice, layer))

    # sweeping
    lattice -= offset
    dist = np.linalg.norm(lattice, 2, axis = 1)
    lattice = lattice[dist < R]

    lattice += offset
    faces = get_faces_from_vertices(lattice)

    return lattice, faces
    # generate FCC lattice


def windmill():
    vertices = np.zeros((0,3))
    
    # first ring (antarctica)
    ring = generate_ring(5, -58, add_center = True)
    vertices = np.vstack((vertices, ring))

    # second ring ()
    ring = generate_ring(5, 16, add_offset = True)
    vertices = np.vstack((vertices, ring))

    # tip
    vertices = np.vstack((vertices, [0, 0, 1]))
    faces = get_faces_from_vertices(vertices)
    vertices[0, :] = [0, 0, np.sin(np.deg2rad(-58))]
    
    return vertices, faces

def snowflake():


    vertices_base = generate_ring(6, -58, add_center = True)
    center = vertices_base[0, :]
    hexagon = vertices_base[1:, :]
    hexagon_center = np.zeros_like(hexagon)
    hexagon_dir = np.zeros_like(hexagon)
    hexagon_normal = np.zeros_like(hexagon)
    faces_base = np.zeros((6, 3))
    for i in np.arange(6):
        this_edge = i
        next_edge = (i + 1) % 6
        hexagon_center[i, :] = (hexagon[this_edge, :] + hexagon[next_edge, :])/2
        hexagon_dir[i, :] = hexagon_center[i] - center
        hexagon_normal[i, :] = (hexagon[next_edge, :] - hexagon[this_edge, :])/2
        faces_base[i, :] = [this_edge + 1, next_edge + 1, 0]

 
    vertices_middle = np.zeros((12, 3))
    pp_edge = 4
    vertices_stripes = np.zeros((6 * 2 * pp_edge, 3))

    z_middle = 0
    for i in np.arange(6):
        left_edge = i * 2
        right_edge = i * 2 + 1
        center = hexagon_center[i]
        center[2] = z_middle
        center += hexagon_dir[i] * 2
        vertices_middle[left_edge, :] = center + hexagon_normal[i]
        vertices_middle[right_edge, :] = center - hexagon_normal[i]
        left_line = vertices_middle[left_edge] - hexagon[i]
        right_line = vertices_middle[right_edge] - hexagon[(i + 1) % 6]
        parts = np.linspace(0, 1, pp_edge)
        for idx, part in enumerate(parts):
            vertices_stripes[i * 2 * pp_edge+ idx, :] = hexagon[i] + part * left_line
            vertices_stripes[i * 2 * pp_edge + idx + pp_edge] = hexagon[(i + 1) % 6] + part * right_line
        for idx in np.arange(pp_edge * 2):
            vertices_stripes[i * pp_edge + idx] += vertices_stripes[i * pp_edge + idx]/1e3


    vertices_top = generate_ring(6, 58, add_center = True)

    vertices = np.vstack((vertices_base, vertices_middle, vertices_stripes, vertices_top))



    faces = get_faces_from_vertices(vertices)

    return vertices, faces

if(__name__ == "__main__"):
    
    vertices, faces = waterman_polyhedron(2, np.sqrt(3))
    plot_solid(vertices, faces)