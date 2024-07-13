import numpy as np
import geometry
import image_helpers
import map_helpers
from matplotlib import pyplot as plt

PP1 = 800 # pixels per unit
n_sampling = 100000


basemap = image_helpers.load_image("images/blue_marble_august_small.png")
solid = np.load("solids/icosahedron.npz")
vertices = solid["vertices"]
#unfolding_chart = solid["unfolding_chart"]
n_faces = solid["faces"].shape[0]
faces = np.zeros(n_faces, dtype=map_helpers.Face)

unfolding_chart = np.array([
    # Antarctica
    [8, 18],
    [8, 3],
    [18, 13],
    [18, 9],
    [8, 17],
    [17, 7],
    [17, 12],
    [9, 2],
    [7, 4],
    [9, 19],
    [19, 14],
    [19, 5],
    [7, 16],
    [16, 11],
    [5, 1],
    [5, 15],
    [16, 6],
    [6, 0],
    [15, 10]
])

for idx_face in np.unique(unfolding_chart):
    nodes = solid["faces"][idx_face, :]
    verts = vertices[nodes]
    face = map_helpers.Face(nodes, verts)
    face.generate_points_from_basemap(basemap, n_sampling)
    face.project_to_plane()
    face.add_border_outline(np.array([0, 0, 0, 0]))
    faces[idx_face] = face


for fold in unfolding_chart:
    faces[fold[1]].align_to(faces[fold[0]], gap = 0.005)

points = np.zeros((0, 2))
RGBA = np.zeros((0, 4))
for idx_face in np.unique(unfolding_chart):
    points = np.vstack((points, faces[idx_face].points))
    RGBA = np.vstack((RGBA, faces[idx_face].RGBA))

map = image_helpers.generate_image(points, RGBA, PP1)
image_helpers.save_image("icosahedron.png", map)