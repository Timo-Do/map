import geometry
import bases
import imageio.v3 as iio
import image_helpers
import plot_helpers
import numpy as np
from matplotlib import pyplot as plt

DO_DEBUG_PLOTS = False
DO_BORDER_ONLY = False
PP1 = 600

def generate_map():
    basemap = image_helpers.load_image("images/blue_marble_august_small.png")

    # Load a solid to project on
    #vertices, faces = bases.create_icosahedron()
    #vertices, faces = bases.fibonacci_sphere(samples=35)
    #vertices, faces = bases.fibonacci_sphere(12)
    vertices, faces = bases.create_icosahedron()
    np.savez("solids/icosahedron.npz", vertices=vertices, faces=faces)
    n_faces = faces.shape[0]
    border_lats = np.zeros(0)
    border_lons = np.zeros(0)
    for idx_face, face in enumerate(faces):
        print(f"Processing triangle {idx_face + 1}/{n_faces}")
        # get the coordinates of the edges of the face
        A, B, C = vertices[face]
    
        triangle = geometry.Triangle(A, B, C)
        if(triangle.is_facing_inwards()):
            triangle.flip()
        # get the coordinates of the inner points
        points = triangle.generate_inner_points(100000)
        lat, lon = geometry.XYZ_to_lat_lon(points)
        border = triangle.generate_edge_points(1000)
        border_lat, border_lon = geometry.XYZ_to_lat_lon(border)
        if(DO_DEBUG_PLOTS):
            plot_helpers.plot_triangle_with_points(triangle, points)
            image_helpers.plot_points_on_map(basemap, lat, lon)
        
        border_lats = np.hstack((border_lats, border_lat))
        border_lons = np.hstack((border_lons, border_lon))
        if(DO_BORDER_ONLY):
            continue
        # get the colors
        RGBA = image_helpers.sample_from_image(basemap, lat, lon)

        # transform points onto "triangle" coordinate system
        flat_triangle, points = triangle.transform_to_2D(points)


        map_part = image_helpers.generate_image(points, RGBA, PP1)
        map_part = map_part.astype(np.uint8)
        iio.imwrite(f"images/generated/{idx_face}.png", map_part)
        
    image_helpers.plot_points_on_map(basemap, border_lats, border_lons)


generate_map()