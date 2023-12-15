import geometry
import bases
import imageio.v3 as iio
import image_helpers
import plot_helpers
import numpy as np
from matplotlib import pyplot as plt

DO_DEBUG_PLOTS = False

def generate_map():
    basemap = iio.imread("images/blue_marble_august_small.png", mode = "RGBA")

    # Load a solid to project on
    vertices, faces = bases.create_icosahedron()
    border_lat = np.zeros(0)
    border_lon = np.zeros(0)
    for idx_face, face in enumerate(faces):
        # get the coordinates of the edges of the face
        A, B, C = vertices[face]
    
        triangle = geometry.Triangle(A, B, C)
        # get the coordinates of the inner points
        #points = triangle.generate_points(10)
        points = triangle.generate_points_on_border(1000)

        lat, lon = geometry.XYZ_to_lat_lon(points)
        if(DO_DEBUG_PLOTS):
            plot_helpers.plot_triangle_with_points(triangle, points)
            image_helpers.plot_points_on_map(basemap, lat, lon)
        
        border_lat = np.hstack((border_lat, lat))
        border_lon = np.hstack((border_lon, lon))

        continue
        # get the colors
        RGBA = image_helpers.sample_from_image(basemap, lat, lon)

        # transform points onto "triangle" coordinate system
        flat_triangle, points = triangle.transform_to_2D(points)


        map_part = image_helpers.generate_image(flat_triangle, points, RGBA)
        map_part = map_part.astype(np.uint8)
        iio.imwrite(f"images/generated/{idx_face}.png", map_part)
        
    image_helpers.plot_points_on_map(basemap, border_lat, border_lon)


generate_map()