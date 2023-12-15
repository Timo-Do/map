import base
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import imageio.v3 as iio
from scipy.interpolate import LinearNDInterpolator
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 466560000

g = 1.3247179572447460259609088
alpha = np.array([1/g, 1/g**2])
pp1 = 1000

def get_base(w1, w2):
    v1 = w1 / np.linalg.norm(w1)
    t2 = (w2 - np.dot(w2, v1) * v1) 
    v2 = t2 / np.linalg.norm(t2)
    return v1, v2

def r2_points(n = 10):
    N = np.arange(n).reshape(n, 1)
    N = np.mod(N * alpha, 1)
    return N

def lat_2_y(lat, h):
    y = np.round(h * (-lat + 90)/180).astype(np.int32)
    return y

def lon_2_x(lon, w):
    x = np.round(w * (lon + 180)/360).astype(np.int32)
    return x

def sample_from_image(image, lat, lon):
    h = image.shape[0]
    w = image.shape[1]
    y = lat_2_y(lat, h)
    x = lon_2_x(lon, w)
    y[y >= h] = h - 1
    x[x >= w] = w - 1
    return image[y, x, :]

def plot_triangle_on_world(map, lats, lons):
    plt.imshow(map)
    h = image.shape[0]
    w = image.shape[1]
    ys = lat_2_y(lats, h)
    xs = lon_2_x(lons, w)
    ys = np.append(ys, ys[0])
    xs = np.append(xs, xs[0])
    plt.plot(xs, ys, "r.", alpha = 0.3)
    plt.show()

def generate_points_on_face(face, n = 10):
    A = face[0, :]
    B = face[1, :]
    C = face[2, :]
    AC = C - A
    AB = B - A
    unit_square_points = r2_points(n)
    r1 = unit_square_points[:, 0, np.newaxis]
    r2 = unit_square_points[:, 1, np.newaxis]
    # Mask points inside triangle
    mask = unit_square_points[:,0] + unit_square_points[:,1] < 1
    not_mask = np.invert(mask)
    # Parallelogram Method
    new_points = np.tile(A, (n, 1))
    new_points[mask] += r1[mask] * AB + r2[mask] * AC
    new_points[not_mask] += (1 - r1[not_mask]) * AB + (1 - r2[not_mask]) * AC
    
    return new_points

def generate_image(A, B, C, X, Y, r, g, b):
    max_w = B[0]
    max_h = C[1]
    H = (max_h * pp1).astype(np.int32)
    W = (max_w * pp1).astype(np.int32)
    NX = np.linspace(0, max_w, W)
    NY = np.linspace(0, max_h, H)
    xgrid, ygrid = np.meshgrid(NX, NY)
    RGB = np.stack([r, g, b]).T
    XY = np.stack((X,Y)).T
    interp = LinearNDInterpolator(XY, RGB/255)
    z = interp(xgrid, ygrid)
    z[z > 1] = 1
    z[z < 0] = 0
    plt.imshow(z)
    plt.show()

def xyz_to_lat_lon(x, y, z):


    # Calculate longitude
    lon = np.arctan2(y, x)

    # Calculate latitude
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))

    # Convert radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    # Wrap longitude to the range (-180, 180]
    lon = (lon + 180) % 360 - 180


    return lat, lon


coord_points, vertices = base.get_faces(n =7)


image = iio.imread("images/blue_marble_august_small.png")

for idx_face, idx_vertices in enumerate(vertices):
    face = coord_points[idx_vertices]
    points = generate_points_on_face(face, n = 100000)
    df = pd.DataFrame()
    df["x"] = points[:, 0]
    df["y"] = points[:, 1]
    df["z"] = points[:, 2]
    df["lat"], df["lon"] = xyz_to_lat_lon(df["x"], df["y"],df["z"])

    verts_lat, verts_lon = xyz_to_lat_lon(face[:, 0], face[:, 1], face[:, 2])
    print("Vertices:")
    print(verts_lat)
    print(verts_lon)
    
    plot_triangle_on_world(image, df["lat"], df["lon"])
    
    df[["r", "g", "b"]] = sample_from_image(image, df["lat"], df["lon"])
    a = face[0, :]
    b = face[1, :]
    c = face[2, :]
    points -= a
    b -= a
    c -= a
    a -= a
    v1, v2 = get_base(b-a, c-a)
    A = np.zeros(2)
    B = np.array([np.dot(b, v1), np.dot(b, v2)])
    C = np.array([np.dot(c, v1), np.dot(c, v2)])
    print(A, B, C)
    df["X"] = np.dot(points, v1)
    df["Y"] = np.dot(points, v2)
    generate_image(A, B, C, df["X"], df["Y"], df["r"], df["g"], df["b"])