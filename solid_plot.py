from matplotlib import pyplot as plt
import numpy as np
import bases

points = np.load("solids/ant58-990.npy")


fig = plt.figure(figsize=(8,5))
ax = plt.axes(projection='3d')


faces = bases.get_faces_from_vertices(points)

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

ax.plot_trisurf(x, y, z, triangles=faces,
                    cmap='viridis', alpha=0.2, edgecolor='k')
plt.show()