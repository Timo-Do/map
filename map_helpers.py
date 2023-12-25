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
        triangle = geometry.Triangle(A, B, C)
        if(triangle.is_facing_inwards()):
            idx_vector = np.arange(3)
            triangle.flip()
            flipped_idxs = geometry.flip(idx_vector)
            self.nodes = self.nodes[idx_vector]
            self.vertices = self.vertices[flipped_idxs, :]
        self.triangle = triangle

    def add_border_outline(self, color = np.zeros(4)):
        border_points = self.triangle.generate_edge_points(1000)
        border_RGBA = np.tile(color, (border_points.shape[0], 1))
        self.add_points(border_points, border_RGBA)

    def add_points(self, points, RGBA):
        self.points = np.vstack((self.points, points))
        self.RGBA = np.vstack((self.RGBA, RGBA))
        assert self.points.shape[0] == self.RGBA.shape[0]

    def generate_points_from_basemap(self, basemap, n = 100000):
        points = self.triangle.generate_inner_points(n)
        lat, lon = geometry.XYZ_to_lat_lon(points)
        RGBA = image_helpers.sample_from_image(basemap, lat, lon)
        self.add_points(points, RGBA)
    
    def project_to_plane(self):
        self.triangle, self.points = self.triangle.transform_to_2D(self.points)

    def render(self, PP1):
        self.image = image_helpers.generate_image(self.points, self.RGBA, PP1)

    def align_to(self, face, gap = 0):
        names = ["A", "B", "C"]
        common_edge = {}

        bary = self.triangle.get_barycentric_coordinates(self.points)

        for idx_node, node in enumerate(self.nodes):
            result_arr = np.where(face.nodes == node)[0]
            if(result_arr.shape[0] > 0):
                result = result_arr[0]
                common_edge[names[idx_node]] = face.triangle.vertices[result]
        self.points = self.triangle.align_to(self.points, gap = gap, **common_edge)


        
