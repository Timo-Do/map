from matplotlib import pyplot as plt
import numpy as np

def plot_triangle_with_points(triangle, points):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(points[:, 0], points[:, 1], points[:, 2], "r.")
    vertices = np.vstack([triangle.A, triangle.B, triangle.C, triangle.A])
    ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    plt.show()