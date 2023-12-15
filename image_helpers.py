import numpy as np
from scipy.interpolate import LinearNDInterpolator
from matplotlib import pyplot as plt
import imageio.v3 as iio

pp1 = 600

# Transform to picture coordinates
def lat_2_y(lat, h):
    y = np.round(h * (-lat + 90)/180).astype(np.int32)
    return np.clip(y, 0, h - 1)

def lon_2_x(lon, w):
    x = np.round(w * (lon + 180)/360).astype(np.int32)
    return np.clip(x, 0, w - 1)

def sample_from_image(image, lat, lon):
    h = image.shape[0]
    w = image.shape[1]
    y = lat_2_y(lat, h)
    x = lon_2_x(lon, w)
    y[y >= h] = h - 1
    x[x >= w] = w - 1
    return image[y, x, :]

def generate_image(triangle, points, RGBA):
    max_w = triangle.B[0]
    max_h = triangle.C[1]
    H = (max_h * pp1).astype(np.int32)
    W = (max_w * pp1).astype(np.int32)
    NX = np.linspace(0, max_w, W)
    NY = np.linspace(0, max_h, H)
    xgrid, ygrid = np.meshgrid(NX, NY)
    interp = LinearNDInterpolator(points, RGBA, fill_value=0)
    z = interp(xgrid, ygrid)

    return z

def plot_points_on_map(map, lats, lons):
    h = map.shape[0]
    w = map.shape[1]
    ys = lat_2_y(lats, h)
    xs = lon_2_x(lons, w)
    ys = np.append(ys, ys[0])
    xs = np.append(xs, xs[0])
    map[ys, xs] = [1, 1, 1, 1]    
    plt.imshow(map)
    plt.show()

def stitch_together_map(n):
    fname = lambda n: f"images/generated/{n}.png"
    n_cols = 3
    col_width = 700
    row_height = 700
    carpet = np.zeros((row_height * 10, col_width * n_cols, 4))
    for i in range(n):
        row = np.floor_divide(i, n_cols)
        col = np.mod(i, n_cols)
        image = iio.imread(fname(i), mode = "RGBA")
        h = image.shape[0]
        w = image.shape[1]
        row_start = row_height * row
        col_start = col_width * col
        carpet[row_start : row_start + h, col_start : col_start + w, :] = image

    iio.imwrite("carpet.png", carpet.astype(np.uint8))

stitch_together_map(30)