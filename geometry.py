import numpy as np

from matplotlib import pyplot as plt

def r2_points(n = 10):
    # From:
    # https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    g = 1.3247179572447460259609088
    alpha = np.array([1/g, 1/g**2])
    N = np.arange(n).reshape(n, 1)
    N = np.mod(N * alpha, 1)
    return N

def get_base(w1, w2):
    # From:
    # https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    v1 = w1 / np.linalg.norm(w1)
    t2 = (w2 - np.dot(w2, v1) * v1) 
    v2 = t2 / np.linalg.norm(t2)
    return v1, v2

class Triangle():
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
        self.AC = C - A
        self.CA = -self.AC
        self.AB = B - A
        self.BA = -self.AB
        self.BC = C - B
        self.CB = -self.BC

    def generate_points(self, n):    
        # From:
        # https://extremelearning.com.au/evenly-distributing-points-in-a-triangle/
        unit_square_points = r2_points(n)
        r1 = unit_square_points[:, 0, np.newaxis]
        r2 = unit_square_points[:, 1, np.newaxis]
        # Mask points inside triangle
        mask = unit_square_points[:,0] + unit_square_points[:,1] < 1
        not_mask = np.invert(mask)
        # Parallelogram Method
        points = np.tile(self.A, (n, 1))
        points[mask] += r1[mask] * self.AB + r2[mask] * self.AC
        points[not_mask] += (1 - r1[not_mask]) * self.AB + (1 - r2[not_mask]) * self.AC
        return points
    
    def transform_to_2D(self, points):
        
        # get a ortho-normal base of the triangle
        v1, v2 = get_base(self.AB, self.AC)
        v = np.stack((v1, v2)).T
        # Set A as new origin:
        a = np.zeros(2)
        b = np.dot(self.AB, v)
        c = np.dot(self.AC, v)
        new_triangle = Triangle(a, b, c)
        # Move origin
        points -= self.A
        new_points = np.dot(points, v)
        return new_triangle, new_points
            

    
    
def xyz_to_lat_lon(x, y, z):

    # Calculate longitude
    lon = np.arctan2(y, x)

    # Calculate latitude
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))

    # Convert radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    # Wrap longitude to the range (-180, 180]
    lon = (lon + 180) % 360 - 180


    return lat, lon