import geometry
import image_helpers
import numpy as np

class Face():
    def __init__(self, nodes, vertices):
        self.nodes = nodes
        self.vertices = vertices
        A, B, C = vertices
        triangle = geometry.Triangle(A, B, C)
        if(triangle.is_facing_inwards()):
            triangle.flip()
            nodes = geometry.flip(nodes)
        self.triangle = triangle

    def generate_inner_points(self):
        self.points = self.triangle.generate_points(100000)

    def get_RGBA_from_basemap(self, basemap):
        lat, lon = geometry.XYZ_to_lat_lon(self.points)
        self.RGBA = image_helpers.sample_from_image(basemap, lat, lon)
    
    def project_to_plane(self):
        self.triangle, self.points = self.triangle.transform_to_2D(self.points)

    def render(self):
        self.image = image_helpers.generate_image(self.points, self.RGBA)

    def align_to(self, face):
        names = ["A", "B", "C"]
        common_edge = {}
        for idx_node, node in enumerate(self.nodes):
            result_arr = np.where(face.nodes == node)[0]
            if(result_arr.shape[0] > 0):
                result = result_arr[0]
                common_edge[names[idx_node]] = face.triangle.vertices[result]
        self.points = self.triangle.align_to(self.points, **common_edge)

        
