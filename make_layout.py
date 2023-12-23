import numpy as np
import geometry
import image_helpers
import map_helpers
from matplotlib import pyplot as plt




basemap = image_helpers.load_image("images/blue_marble_august_small.png")
solid = np.load("solids/timo_spezial.npz")
vertices = solid["vertices"]
n_faces = solid["faces"].shape[0]
faces = np.zeros(20, dtype=map_helpers.Face)

folding_chart = np.array([
    [19,  7],
    [ 7, 10],
    [10, 11],
    [11, 18]
])

for idx_face in np.unique(folding_chart):
    nodes = solid["faces"][idx_face, :]
    verts = vertices[nodes]
    face = map_helpers.Face(nodes, verts)
    face.generate_inner_points()
    face.get_RGBA_from_basemap(basemap)
    face.project_to_plane()
    faces[idx_face] = face


for fold in folding_chart:
    faces[fold[1]].align_to(faces[fold[0]])

points = np.zeros((0, 2))
RGBA = np.zeros((0, 4))
for idx_face in np.unique(folding_chart):
    points = np.vstack((points, faces[idx_face].points))
    RGBA = np.vstack((RGBA, faces[idx_face].RGBA))

map = image_helpers.generate_image(points, RGBA/255)
plt.imshow(map)
plt.show()