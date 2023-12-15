import numpy as np
from scipy.interpolate import LinearNDInterpolator
from matplotlib import pyplot as plt

pp1 = 1000

# Transform to picture coordinates
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

def generate_image(triangle, points, RGB):
    max_w = triangle.B[0]
    max_h = triangle.C[1]
    H = (max_h * pp1).astype(np.int32)
    W = (max_w * pp1).astype(np.int32)
    NX = np.linspace(0, max_w, W)
    NY = np.linspace(0, max_h, H)
    xgrid, ygrid = np.meshgrid(NX, NY)
    interp = LinearNDInterpolator(points, RGB/255)
    z = interp(xgrid, ygrid)
    z[z > 1] = 1
    z[z < 0] = 0
    return z

def plot_points_on_map(map, lats, lons):
    plt.imshow(map)
    h = map.shape[0]
    w = map.shape[1]
    ys = lat_2_y(lats, h)
    xs = lon_2_x(lons, w)
    ys = np.append(ys, ys[0])
    xs = np.append(xs, xs[0])
    plt.plot(xs, ys, "r.", alpha = 0.3)
    plt.show()

