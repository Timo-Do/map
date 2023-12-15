import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def create_icosahedron():
    t = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    vertices = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ])
    
    faces = np.array([
        [0, 11,  5], [0,  5,  1], [0,  1,  7], [0,  7, 10], [0, 10, 11],
        [1,  5,  9], [5, 11,  4], [11, 10,  2], [10,  7,  6], [ 7,  1,  8],
        [3,  9,  4], [3,  4,  2], [3,  2,  6], [3,  6,  8], [3,  8,  9],
        [4,  9,  5], [2,  4, 11], [6,  2, 10], [8,  6,  7], [9,  8,  1]
    ])
    
    return vertices, faces


def fibonacci_sphere(samples=1000):

    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)



def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    See:
        https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def get_faces(n=15):
    #points = fibonacci_sphere(samples=n)
    #points = create_icosahedron()
    #hull = ConvexHull(points)
    #faces = points[hull.simplices]
    #return points, hull.simplices
    points, vertices = create_icosahedron()
    return points, vertices

if __name__ == "__main__":

    points = fibonacci_sphere(samples=11)

    hull = ConvexHull(points)

    #vertices, faces = create_icosahedron()

    # Get the face vertices
    faces = points[hull.simplices]
    print(faces[0, :, :])

    # Compute the centroids
    centroids = np.mean(faces, axis=1)
    print(centroids[0, :])

    # Compute the normal vectors
    normals = np.cross(faces[:, 0] - faces[:, 1], faces[:, 1] - faces[:, 2])


    centroid_hull = np.zeros(3)
    centroid_vector = centroids - centroid_hull
    normals *= np.sign(np.sum(normals * centroid_vector, axis=1))[:, np.newaxis]

    fig = plt.figure(figsize=(8,5))
    ax = plt.axes(projection='3d')
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    s = ax.plot_trisurf(X, Y, Z, triangles=hull.simplices,
                        cmap='viridis', alpha=0.2, edgecolor='k')

    for centroid, normal in zip(centroids, normals):
        ax.quiver(centroid[0], centroid[1], centroid[2], normal[0], normal[1], normal[2], color='red')
    ax.plot(centroids[:, 0], centroids[:, 1], centroids[:, 2], 'ro', markersize=4)

    set_axes_equal(ax)
    plt.show()