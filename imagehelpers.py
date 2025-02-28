import PIL.Image
import numpy as np
from matplotlib import pyplot as plt
import imageio.v3 as iio
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
        svg_width = 500 #mm
        scale =  abs(ww[0, 0] - ww[-1, -1])/svg_width
        svg_height = abs(hh[0, 0] - hh[-1, -1]) / scale   
        topLeft = (ww[0,0], hh[0,0])
        movedOutline = outline - topLeft
        svg = drawsvg.Drawing(svg_width, svg_height)
        path = 'M ' + ' L '.join(f'{x},{y}' for x, y in movedOutline/scale) + ' Z'
        svg.append(drawsvg.Path(d = path, stroke = "black"))
    return canvas, svg


def getGaussianKernel(size = 5, cov = 0.5):
    x = np.linspace(0, size, size, endpoint=False)
    y = multivariate_normal.pdf(x, mean=size//2, cov=cov)
    y = y.reshape(1,size)
    return np.dot(y.T,y)

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

