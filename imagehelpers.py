import PIL.Image
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from matplotlib import pyplot as plt
import imageio.v3 as iio
from glob import glob
import os
import random
import geometryhelpers
import maphelpers
from scipy.stats import multivariate_normal
import scipy.signal
import drawsvg

import PIL

PIL.Image.MAX_IMAGE_PIXELS = None

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

def sample_from_images(images, lat, lon):
    cols, dx = divmod(lon + 180, 90)
    rows, dy = divmod(-lat + 90, 90)

    RGBA = np.zeros((len(lat), 4))
    parts = np.vstack((cols, rows)).T
    for part in np.unique(parts, axis = 0):
        image = images[part[0]][part[1]]
        h = image.shape[0]
        w = image.shape[1]
        partMask = np.all(parts == part, axis = 1)
        dxs = dx[partMask]
        dys = dy[partMask]
        x = np.clip(w * dxs / 90, 0, w - 1).astype(np.int32)
        y = np.clip(h * dys / 90, 0, h - 1).astype(np.int32)
        RGBA[partMask] = image[y, x, :]
    return RGBA


def generate_canvas(bounds, PP1, padding = 0, background = [0,0,0,0]):
    padding = padding / PP1
    [x_min, x_max, y_min, y_max] = bounds
    inc = 1/PP1
    x = np.arange(x_min - padding, x_max + padding + inc, inc)
    y = np.arange(y_min - padding, y_max + padding + inc, inc)
    ww, hh = np.meshgrid(x, y)
    canvas = np.zeros((hh.shape[0], hh.shape[1], 4), dtype = np.uint8)
    canvas[:, :, :] = background
    return canvas, hh, ww

def render(faces, baseimage, PP1, overspill = 0, background = [0,0,0,0], blur = True, outline = None):
    bounds = maphelpers.get_bounds(faces)
    canvas, hh, ww = generate_canvas(bounds, PP1, padding = 150, background = background)
    h = canvas.shape[0]
    w = canvas.shape[1]
    xx = ww.reshape(-1, 1)
    yy = hh.reshape(-1, 1)

    XY = np.hstack((xx, yy))
    canvas = canvas.reshape((XY.shape[0], 4))
    outside = np.ones(XY.shape[0], dtype=bool)
    for face in faces:
        BC = face.triangle_on_plane.XY_to_barycentric(XY)
        inner_triangle = np.min(BC, axis = 1) > -overspill
        XYZ = face.triangle_on_solid.barycentric_to_XYZ(BC[inner_triangle, :])
        lats, lons = geometryhelpers.XYZ_to_lat_lon(XYZ)
        if(isinstance(baseimage, np.ndarray)):
            RGBA = sample_from_image(baseimage, lats, lons)
        elif(isinstance(baseimage, dict)):
            RGBA = sample_from_images(baseimage, lats, lons)
        else:
            raise NotImplementedError(f"Parsed baseimage of type {type(baseimage)} is not supported.")
        canvas[inner_triangle, :] = RGBA
        outside[inner_triangle] = False
    
    canvas = canvas.reshape(h, w, 4)
    if(blur):
        outside = outside.reshape(h, w)

        # blur
        kernel = getGaussianKernel()
        blurred = np.zeros_like(canvas)
        for channel in np.arange(canvas.shape[2]):
            blurred[:, :, channel] = scipy.signal.convolve2d(canvas[:, :, channel], kernel, mode = "same")
        canvas[outside] = blurred[outside]

    if(outline is None):
        return canvas
    else:
        #svg_width = abs(ww[0, 0] - ww[-1, -1])
        #svg_height = abs(hh[0, 0] - hh[-1, -1])
        svg_width = 500 #mm
        #svg_height = h * scale
        scale =  abs(ww[0, 0] - ww[-1, -1])/svg_width
        svg_height = abs(hh[0, 0] - hh[-1, -1]) / scale
        #scale = np.mean([w/svg_width, h/svg_height])        
        topLeft = (ww[0,0], hh[0,0])
        movedOutline = outline - topLeft
        svg = drawsvg.Drawing(svg_width, svg_height)
        path = 'M ' + ' L '.join(f'{x},{y}' for x, y in movedOutline/scale) + ' Z'
        svg.append(drawsvg.Path(d = path, stroke = "black"))
        #svg.save_svg("wetesting.svg")
    return canvas, svg


def getGaussianKernel(size = 5, cov = 0.5):
    x = np.linspace(0, size, size, endpoint=False)
    y = multivariate_normal.pdf(x, mean=size//2, cov=cov)
    y = y.reshape(1,size)
    return np.dot(y.T,y)

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
    png_files = glob(os.path.join("images", "butterfly", "*.png"))
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

def load_basemap(fnameFun):
    cols = ["A", "B", "C", "D"]
    rows = ["1", "2"]
    images = {}
    for col in range(len(cols)):
        images[col] = {}
        for row in range(len(rows)):
            images[col][row] = load_image(fnameFun(col, row))
    return images

def save_image(path, image):
    image = image.astype(np.uint8)
    iio.imwrite(path, image)

