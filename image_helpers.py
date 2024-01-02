import numpy as np
from scipy.interpolate import LinearNDInterpolator
from matplotlib import pyplot as plt
import imageio.v3 as iio
from glob import glob
import os
import random
import geometry
import map_helpers


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


def generate_canvas(bounds, PP1, padding = 0):
    padding = padding / PP1
    [x_min, x_max, y_min, y_max] = bounds
    inc = 1/PP1
    x = np.arange(x_min - padding, x_max + padding + inc, inc)
    y = np.arange(y_min - padding, y_max + padding + inc, inc)
    ww, hh = np.meshgrid(x, y)
    canvas = np.zeros((hh.shape[0], hh.shape[1], 4), dtype = np.uint8)
    return canvas, hh, ww

def render(faces, baseimage, PP1, overspill = 0):
    bounds = map_helpers.get_bounds(faces)
    canvas, hh, ww = generate_canvas(bounds, PP1, padding = 10)
    h = canvas.shape[0]
    w = canvas.shape[1]
    xx = ww.reshape(-1, 1)
    yy = hh.reshape(-1, 1)

    XY = np.hstack((xx, yy))
    canvas = canvas.reshape((XY.shape[0], 4))
 
    for face in faces:
        BC = face.triangle_on_plane.XY_to_barycentric(XY)
        inner_triangle = np.min(BC, axis = 1) > -overspill
        XYZ = face.triangle_on_solid.barycentric_to_XYZ(BC[inner_triangle, :])
        lats, lons = geometry.XYZ_to_lat_lon(XYZ)
        RGBA = sample_from_image(baseimage, lats, lons)
        canvas[inner_triangle, :] = RGBA
    
    canvas = canvas.reshape(h, w, 4)
    return canvas


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