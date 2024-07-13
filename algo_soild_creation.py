from bases import *
import geometry
from scipy.optimize import differential_evolution


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
