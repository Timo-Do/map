import geometry
import bases
import imageio.v3 as iio
import image_helpers
import plot_helpers
import numpy as np
from matplotlib import pyplot as plt

DO_DEBUG_PLOTS = False

def generate_map():
    basemap = iio.imread("images/blue_marble_august_small.png")

    # Load a solid to project on
    vertices, faces = bases.create_icosahedron()

    for idx_face, face in enumerate(faces):
        # get the coordinates of the edges of the face
        A, B, C = vertices[face]
    
        triangle = geometry.Triangle(A, B, C)
        # get the coordinates of the inner points
        X, Y, Z = triangle.generate_points(1000).T
        if(DO_DEBUG_PLOTS):
            plot_helpers.plot_triangle_with_points(triangle, np.stack((X, Y, Z)).T)
        lat, lon = geometry.xyz_to_lat_lon(X, Y, Z)

        image_helpers.plot_points_on_map(basemap, lat, lon)
        # get the colors
        RGB = image_helpers.sample_from_image(basemap, lat, lon)
        # transform points onto "triangle" coordinate system
        flat_triangle, points = triangle.transform_to_2D(np.stack((X, Y, Z)).T)


        map_part = image_helpers.generate_image(flat_triangle, points, RGB)

        plt.imshow(map_part)
        plt.show()


generate_map()