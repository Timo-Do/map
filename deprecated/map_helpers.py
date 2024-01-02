import geometry
import image_helpers
import numpy as np

class Face():
    def __init__(self, nodes, vertices):
        self.nodes = nodes
        self.vertices = vertices
        self.points = np.zeros((0,vertices.shape[1]))
        self.RGBA = np.zeros((0, 4))
        A, B, C = vertices
        self.original_triangle = geometry.Triangle(A, B, C)
        if(self.original_triangle.is_facing_inwards()):
            self.flip()
        self.free_edges = np.array([True, True, True]) # AB, BC, CA
        self.is_aligned = False

    def _vector_to_edge(self, edge_vector):
        AB = np.array([1, 1, 0])
        BC = np.array([0, 1, 1])
        CA = np.array([1, 0, 1])
        if(edge_vector == AB):
            return 0
        elif(edge_vector == BC):
            return 1
        elif(edge_vector == CA):
            return 2
        else:
            raise ValueError("Not a valid edge vector!")

    def flip(self):
        self.original_triangle.flip()
        idx_vector = np.arange(3)
        flipped_idxs = geometry.flip(idx_vector)
        self.nodes = self.nodes[idx_vector]
        self.vertices = self.vertices[flipped_idxs, :]

    def add_border_outline(self, color = np.zeros(4)):
        border_points = self.projected_triangle.generate_edge_points(1000)
        border_RGBA = np.tile(color, (border_points.shape[0], 1))
        self.add_points(border_points, border_RGBA)

    def add_points(self, points, RGBA):
        self.points = np.vstack((self.points, points))
        self.RGBA = np.vstack((self.RGBA, RGBA))
        assert self.points.shape[0] == self.RGBA.shape[0]

    def generate_points_from_basemap(self, basemap, n = 100000):
        points = self.original_triangle.generate_inner_points(n)
        lat, lon = geometry.XYZ_to_lat_lon(points)
        RGBA = image_helpers.sample_from_image(basemap, lat, lon)
        self.add_points(points, RGBA)
    
    def project_to_plane(self):
        self.projected_triangle, self.points = self.original_triangle.transform_to_2D(self.points)

    def render(self, PP1):
        self.image = image_helpers.generate_image(self.points, self.RGBA, PP1)

    def align_to(self, face, gap = 0):
        names = ["A", "B", "C"]
        common_edge = {}
        my_vertices = np.zeros(3)
        their_vertices = np.zeros(3)
        for idx_node, node in enumerate(self.nodes):
            result_arr = np.where(face.nodes == node)[0]
            if(result_arr.shape[0] > 0):
                result = result_arr[0]
                common_edge[names[idx_node]] = face.projected_triangle.vertices[result]
                my_vertices[idx_node] = 1
                their_vertices[result] = 1
        self.points = self.projected_triangle.align_to(self.points, gap = gap, **common_edge)

        self.is_aligned = True
        # self.free_edges[self._vector_to_edge(my_vertices)] = False
        # face.free_edges[self._vector_to_edge(their_vertices)] = False


        
