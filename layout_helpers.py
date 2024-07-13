import map_helpers
import numpy as np


class Layout():
    def __init__(self, faces : np.ndarray) -> None:
        self.faces = faces
        self.normals = np.array([Face.triangle.normal for Face in self.faces])
        for face in self.faces:
            face.project_to_plane()

    def get_eligble_neighbors(self, idx_face):
      
        my_nodes = face.nodes
        all_nodes = np.array([f.nodes for f in self.faces])
        common_nodes = np.array([[node in nodes for node in my_nodes] for nodes in all_nodes])

        common_nodes_count = np.sum(common_nodes, axis = 1)
        indices = np.arange(self.faces.shape[0])
        return indices[common_nodes_count == 2]
    
    def find_same_group(self, idx_face):
        
        normal = self.normals[idx_face, :]
        diff_to_normal = np.linalg.norm(self.normals - normal, 2, axis = 1)
        same_group = faces[diff_to_normal < 1e-13]


    def generate_net(self, start_face = 0):
        unfolding_chart = []
        faces = np.arange(self.faces.shape[0])
        
        current_face = start_face
        finished = False
        while(not finished):
            face = self.faces[current_face]
            elgible_neighbors = self.get_eligble_neighbors(face)
            same_group = self.find_same_group
            for neighbor in elgible_neighbors:
                if(neighbor in same_group):
                    if(neighbor in np.unique(unfolding_chart)):
                        face.align_to(self.faces[neighbor])
                        break
                if(neighbor in np.unique(unfolding_chart)):
                    face.align_to(self.faces[neighbor])
                    break
                

            

            break


def load_solid(path):
    solid = np.load(path)
    vertices = solid["vertices"]
    nfaces = solid["faces"].shape[0]
    faces = np.zeros(nfaces, dtype=map_helpers.Face)
    for idx_face in np.arange(nfaces):
        nodes = solid["faces"][idx_face, :]
        verts = vertices[nodes]
        face = map_helpers.Face(nodes, verts)
        faces[idx_face] = face
    return faces


if(__name__ == "__main__"):
    faces = load_solid("solids/windmill.npz")
    layout = Layout(faces)
    layout.generate_net(start_face=11)

