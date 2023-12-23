import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
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
        xyz = np.vstack(([0, 0, np.sin(np.deg2rad(lat)) - 1e-5], xyz))
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

def evaluate_triangulation(latlons, fixed, special, full = False):
    latlons = np.reshape(latlons, (-1, 2))
    vertices = geometry.lat_lon_to_XYZ(latlons[:, 0], latlons[:, 1])
    vertices = np.vstack((fixed, vertices))
    faces = get_faces_from_vertices(vertices)
    n_faces = faces.shape[0]
    lengths = np.ones((n_faces, 3)) * np.nan
    for idx_face, face in enumerate(faces):
        # get the coordinates of the edges of the face
        if(vertices[special] in face):
            continue
        A, B, C = vertices[face]
        triangle = geometry.Triangle(A, B, C)
        lengths[idx_face, :] = triangle.calculate_lengths()

    lengths = lengths.reshape(-1, 1)
    if(full):
        return lengths
    else:
        return np.nanmax(lengths) - np.nanmin(lengths)

class CallBackHandler():
    generation = 0
    
    def __init__(self, name, aggregate):
        self.name = name
        self.fun = aggregate
    def callback(self, xk, convergence):
        print(f"Generation\t : {self.generation}")
        print(f"Convergence\t : {convergence}")
        print(" - - - ")
        if(self.generation % 10 == 0):
            fname = f"solids/{self.name}-{self.generation:03}"
            points = self.fun(xk)
            np.save(fname, points, allow_pickle=False)
        self.generation += 1

def generate_triang():
    lat_ring = -58
    n_ring = 5
    eps_center = np.array([0, 0, 1e-7])
    center = np.array([0, 0, np.sin(np.deg2rad(lat_ring))])
    lons = np.linspace(0, 360, n_ring, endpoint=False)
    lats = np.ones_like(lons) * (lat_ring)
    xyz_ring = geometry.lat_lon_to_XYZ(lats, lons)
    xyz_fixed = np.vstack((center - eps_center, xyz_ring, np.array([0,0,1])))

    n_free = 10
    bounds_lat = (lat_ring, 90)
    bounds_lon = (-180, 180)
    bounds = []
    for i in np.arange(n_free * 2):
        if(i % 2):
            bounds.append(bounds_lon)
        else:
            bounds.append(bounds_lat)
    args = (xyz_fixed, 0)
    def aggregate(xk):
        xk = xk.reshape(-1, 2)
        lats = xk[:, 0]
        lons = xk[:, 1]
        xyz_free = geometry.lat_lon_to_XYZ(lats, lons)
        points = np.vstack((xyz_fixed, xyz_free))
        #points[0, :] = center
        return points
    
    cbh = CallBackHandler("ant58", aggregate)

    differential_evolution(
        evaluate_triangulation, bounds, args = args, callback = cbh.callback)


def timo_spezial2():
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



if(__name__ == "__main__"):
    generate_triang()