import geometryhelpers

import numpy as np

class Face():
    def __init__(self, nodes, vertices):
        self.nodes = nodes
        self.vertices = vertices
        A, B, C = vertices
        self.triangle_on_solid = geometryhelpers.Triangle(A, B, C)
        if(self.triangle_on_solid.is_facing_inwards()):
            self.flip()
        self.triangle_on_plane = self.triangle_on_solid.transform_to_2D()

 
    def flip(self):
        self.triangle_on_solid.flip()
        idx_vector = np.arange(3)
        flipped_idxs = geometryhelpers.flip(idx_vector)
        self.nodes = self.nodes[flipped_idxs]
        self.vertices = self.vertices[flipped_idxs, :]

    def align_to(self, face, gap = 0):
        names = ["A", "B", "C"]
        common_edge = {}
        my_vertices = np.zeros(3)
        their_vertices = np.zeros(3)
        for idx_node, node in enumerate(self.nodes):
            result_arr = np.where(face.nodes == node)[0]
            if(result_arr.shape[0] > 0):
                result = result_arr[0]
                common_edge[names[idx_node]] = face.triangle_on_plane.vertices[result]
                my_vertices[idx_node] = 1
                their_vertices[result] = 1
        self.triangle_on_plane.align_to(gap = gap, **common_edge)


        
def get_bounds(faces):
    x_min = np.inf
    y_min = np.inf
    x_max = -np.inf
    y_max = -np.inf
    for face in faces:
        vertices = np.array(face.triangle_on_plane.vertices)
        min = np.min(vertices, axis = 0)
        x_min = min[0] if min[0] < x_min else x_min
        y_min = min[1] if min[1] < y_min else y_min

        max = np.max(vertices, axis = 0)
        
        x_max = max[0] if max[0] > x_max else x_max
        y_max = max[1] if max[1] > y_max else y_max

    return [x_min, x_max, y_min, y_max]