import numpy as np
import geometry
import image_helpers
import map_helpers
from matplotlib import pyplot as plt

PP1 = 800 # pixels per unit
n_sampling = 100000


basemap = image_helpers.load_image("images/blue_marble_august_small.png")
solid = np.load("solids/timo_spezial.npz")
vertices = solid["vertices"]

n_faces = solid["faces"].shape[0]
faces = np.zeros(20, dtype=map_helpers.Face)

unfolding_chart = np.array([
    # Antarctica
    [19,  7],
    [ 7, 10],
    [10, 11],
    [11, 18],
    # Africa and Europe
    [19, 17],
    [17, 16],
    [16,  4],
    # Asia
    [ 7,  8],
    [ 8,  9],
    [ 9,  5],
    # Pacific East
    [10, 12],
    [12,  3],
    [ 3,  0],
    # Pacific West
    [11, 14],
    [14, 13],
    [13,  1],
    # Americas
    [18, 15],
    [15,  6],
    [ 6,  2]
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
image_helpers.save_image("layout.png", map)