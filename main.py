import bases
import image_helpers
import map_helpers


from matplotlib import pyplot as plt


DO_DEBUG_PLOTS = False
DO_BORDER_ONLY = False
PP1 = 600

def generate_map():
    basemap = image_helpers.load_image("images/blue_marble_august_small.png")

    vertices, faces = bases.create_icosahedron()
    n_faces = faces.shape[0]

    for idx_face, face in enumerate(faces):
        print(f"Processing face {idx_face + 1}/{n_faces}")
        triangle = map_helpers.Face(face, vertices[face])
        canvas = image_helpers.render([triangle], basemap, 600, overspill = 0)
        
        plt.imshow(canvas)
        plt.show()
        break
        image_helpers.save_image(f"images/generated/{idx_face}.png", canvas)



generate_map()