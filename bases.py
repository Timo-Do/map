import numpy as np
from scipy.spatial import ConvexHull
import geometryhelpers
from matplotlib import pyplot as plt

def get_faces_from_vertices(vertices):
    hull = ConvexHull(vertices, qhull_options="Qc")
    faces = hull.simplices
    return faces



def plot_solid(vertices, faces):
    fig = plt.figure(figsize=(8,5))
    ax = plt.axes(projection='3d')

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    ax.plot_trisurf(x, y, z, triangles=faces,
                        cmap='viridis', alpha=0.2, edgecolor='k')
    plt.show()

def createOctahedron():
    vertices = np.array([
        # tips
        [  0,  0,  1],
        [  0,  0, -1],
        # base
        [  1,  0,  0],
        [  0,  1,  0],
        [ -1,  0,  0],
        [  0, -1,  0]
    ])
    return vertices