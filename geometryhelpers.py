import numpy as np


def _generate_sequence(alpha, n):
    N = np.arange(1, n + 1).reshape(n, 1)
    N = np.mod(N * alpha, 1)
    return N

def r1_points(n):
    # From:
    # https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    g = 1.6180339887498948482
    alpha = 1/g
    return _generate_sequence(alpha, n)

def r2_points(n):
    # From:
    # https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    g = 1.32471795724474602596
    alpha = np.array([1/g, 1/g**2])
    return _generate_sequence(alpha, n)

def flip(vector):
    vector[0], vector[1] = vector[1], vector[0]
    return vector

def generate_rotation_matrix(rotation_angle):
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle),  np.cos(rotation_angle)]])
    return rotation_matrix

def get_base(w1, w2):
    # From:
    # https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    v1 = w1 / np.linalg.norm(w1)
    t2 = (w2 - np.dot(w2, v1) * v1) 
    v2 = t2 / np.linalg.norm(t2)
    return np.stack((v1, v2)).T

class Triangle():
    def __init__(self, A, B, C):
        self._A = np.array(A)
        self._B = np.array(B)
        self._C = np.array(C)
        self.do_calculations()
    
    @property
    def A(self):
        return self._A
    @A.setter
    def A(self, new_value):
        self._A = new_value
        self.do_calculations()

    @property
    def B(self):
        return self._B
    @B.setter
    def B(self, new_value):
        self._B = new_value
        self.do_calculations()

    @property
    def C(self):
        return self._C
    @C.setter
    def C(self, new_value):
        self._C = new_value
        self.do_calculations()

    def calculate_lengths(self):
        return np.linalg.norm([self.AB, self.BC, self.CA], axis = 1)

    def do_calculations(self):
        self.AC = self.C - self.A
        self.CA = -self.AC
        self.AB = self.B - self.A
        self.BA = -self.AB
        self.BC = self.C - self.B
        self.CB = -self.BC
        self.center = np.mean([self.A, self.B, self.C], axis = 0)
        self.normal = np.cross(self.AB, self.AC)
        self.vertices = (self.A, self.B, self.C)
        self.dim = self.center.shape[0]
        self.L = np.array([self.A, self.B, self.C]).T

    def generate_barycentric_conversion_matrix(self):
        T_upper = np.array([self.A, self.B, self.C]).T
        T_lower = np.ones(3)
        T = np.vstack([T_upper, T_lower])
        return np.linalg.inv(T)

    def flip(self):
        [self.A, self.B, self.C] = flip([self.A, self.B, self.C])

    def is_facing_inwards(self):
        return np.sign(np.dot(self.center, self.normal)) > 0

    def generate_inner_points(self, n):    
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
    
    def transform_to_2D(self):
        
        # get a ortho-normal base of the triangle
        v = get_base(self.AB, self.AC)
        # Set A as new origin:
        a = np.zeros(2)
        b = np.dot(self.AB, v)
        c = np.dot(self.AC, v)
        new_triangle = Triangle(a, b, c)
        # Move origin
        return new_triangle
            

            
    def XY_to_barycentric(self, XY):
        T_inv = self.generate_barycentric_conversion_matrix()
        n = XY.shape[0]

        if(self.dim > 2):
            raise ValueError("Can only compute barycentric coordinates for 2D triangle.")
        ones = np.ones((n, 1))
        BC = np.dot(T_inv, np.hstack((XY, ones)).T).T 

        return BC
    
    def barycentric_to_XYZ(self, BC):
        XYZ = np.dot(self.L, BC.T).T
        return XYZ
    
    def generate_edge_points(self, n):
        # n is per edge!
        assert n > 3
        points = np.array([self.A, self.B, self.C])
        n = n - 2
        k = r1_points(n)
        points = np.vstack((points, self.A + k*self.AB))
        points = np.vstack((points, self.A + k*self.AC))
        points = np.vstack((points, self.B + k*self.BC))
        return points
    
    def translate(self, translation_vector):
        self.A += translation_vector
        self.B += translation_vector
        self.C += translation_vector

    def rotate(self, rotation_matrix):
        self.A = np.dot(rotation_matrix, self.A)
        self.B = np.dot(rotation_matrix, self.B)
        self.C = np.dot(rotation_matrix, self.C)


    def align_to(self, A = None, B = None, C = None, gap = 0):
        args = [A, B, C]
        args_given = [arg is not None for arg in args]
        if(np.sum(args_given) != 2):
            raise ValueError("Can only align to two new points")
        if(A is not None):
            target = np.array(A)
            anchor = np.array(self.A)
            if(B is not None):
                current_edge = self.AB
                alignment_edge = B - A
                alignment_center = (B + A)/2
            elif(C is not None):
                current_edge = self.AC
                alignment_edge = C - A
                alignment_center = (C + A)/2
        elif(B is not None):
            target = np.array(B)
            anchor = np.array(self.B)
            current_edge = self.BC
            alignment_edge = C - B
            alignment_center = (C + B)/2

        self.translate(-anchor) 

        current_angle = np.arctan2(current_edge[1], current_edge[0])
        new_angle = np.arctan2(alignment_edge[1], alignment_edge[0])
        rotation_angle = new_angle - current_angle
        rotation_matrix = generate_rotation_matrix(rotation_angle)

        self.rotate(rotation_matrix)

        self.translate(target)

        if(gap > 0):
            rot90 = generate_rotation_matrix(np.pi/2)
            edge_normal = np.dot(rot90, alignment_edge)
            edge_normal *= np.sign(np.dot(self.center - alignment_center, edge_normal))
            edge_normal /= np.linalg.norm(edge_normal, 2)
            translate_gap = edge_normal * gap

            self.translate(translate_gap)
        
        
def XYZ_to_lat_lon(points):
    X, Y, Z = points.T

    # Calculate longitude
    lon = np.arctan2(Y, X)

    # Calculate latitude
    lat = np.arctan2(Z, np.sqrt(X**2 + Y**2))

    # Convert radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    # Wrap longitude to the range (-180, 180]
    lon = (lon + 180) % 360 - 180
    return lat, lon

def lat_lon_to_XYZ(lat, lon):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    X = np.cos(lat) * np.cos(lon)
    Y = np.cos(lat) * np.sin(lon)
    Z = np.sin(lat)
    return np.stack((X,Y,Z)).T

def getRotationMatrices(alpha, beta, gamma):
    yaw = np.array([
        [ np.cos(alpha), -np.sin(alpha), 0],
        [ np.sin(alpha),  np.cos(alpha), 0],
        [             0,              0,  1]
    ])

    pitch = np.array([
        [ np.cos(beta), 0, np.sin(beta)],
        [            0, 1,            0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    roll = np.array([
        [1,             0,              0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma),  np.cos(gamma)]
    ])
    return yaw, pitch, roll