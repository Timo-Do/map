import numpy as np
from scipy.interpolate import LinearNDInterpolator
from matplotlib import pyplot as plt
import imageio.v3 as iio
from glob import glob
import os
import random


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

def generate_image(points, RGBA, PP1):
    min_w = np.min(points[:, 0])
    max_w = np.max(points[:, 0])
    min_h = np.min(points[:, 1])
    max_h = np.max(points[:, 1])
    range_w = max_w - min_w
    range_h = max_h - min_h
    H = (range_h * PP1).astype(np.int32)
    W = (range_w * PP1).astype(np.int32)
    NX = np.linspace(min_w, max_w, W)
    NY = np.linspace(min_h, max_h, H)
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

def stitch_together_map(max_width = 2500):
    png_files = glob(os.path.join("images", "generated", "*.png"))
    n = len(png_files)
    current_col_height = 0
    current_x_position = 0
    current_y_position = 0
    carpet = np.zeros((1000, max_width, 4))
    random.shuffle(png_files)
    for fname in png_files:
        image = iio.imread(fname, mode = "RGBA")
        h = image.shape[0]
        w = image.shape[1]

        if(current_x_position + w > max_width):
            current_y_position += current_col_height + 10
            current_x_position = 0
            current_col_height = 0
        while(current_y_position + h > carpet.shape[0]):
            carpet = np.vstack((carpet, np.zeros_like(carpet)))
        if(h > current_col_height):
            current_col_height = h
        carpet[current_y_position : current_y_position + h,
            current_x_position : current_x_position + w, :] = image
        current_x_position += w

    carpet = carpet[0 : current_y_position + current_col_height, :, :]

    iio.imwrite("carpet.png", carpet.astype(np.uint8))

def load_image(path):
    return iio.imread(path, mode = "RGBA")

def save_image(path, image):
    image = image.astype(np.uint8)
    iio.imwrite(path, image)


if(__name__ == "__main__"):
    
    stitch_together_map()