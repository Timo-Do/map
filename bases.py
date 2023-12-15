import numpy as np
from scipy.spatial import ConvexHull
import geometry

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
        xyz = np.vstack((xyz, [0, 0, np.sin(np.deg2rad(lat)) - 1e-14]))
    return xyz

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

def timo_spezial():
    vertices = np.zeros((0,3))
    
    # first ring (antarctica)
    ring = generate_ring(5, -58, add_center = True)
    vertices = np.vstack((vertices, ring))

    # middle ring ()
    ring = generate_ring(5, 0, add_offset = True)
    vertices = np.vstack((vertices, ring))

    # third ring ()
    ring = generate_ring(5, 58)
    vertices = np.vstack((vertices, ring))

    # tip
    vertices = np.vstack((vertices, [0, 0, 1]))
    faces = get_faces_from_vertices(vertices)
    
    return vertices, faces