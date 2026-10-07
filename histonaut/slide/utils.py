"""Coordinate rescaling helpers between pyramid levels."""

# Image libraries
from cv2 import convertScaleAbs
from PIL import Image

def get_size_to(size, downsample_from, downsample_to, integer=True):
    """Rescale a size from one downsample factor to another.

    Parameters
    ----------
    size : tuple[float, float]
        `(size_x, size_y)` size expressed at the `downsample_from` level, in
        pixels.

    downsample_from : float
        Downsample factor (relative to level 0) of the level `size` is
        expressed at.

    downsample_to : float
        Downsample factor (relative to level 0) of the target level.

    integer : bool, optional
        If True, round the rescaled values with `round`. Default is True.

    Returns
    -------
    size_to : tuple[int, int] | tuple[float, float]
        `(size_x, size_y)` rescaled by `downsample_from / downsample_to`.
    """
    size_x, size_y = size
    scal = float(downsample_from / downsample_to)
    if integer:
        func_round = round
    else:
        func_round = lambda x: x
    size_x_new = func_round(float(size_x) * scal)
    size_y_new = func_round(float(size_y) * scal)
    size_to = size_x_new, size_y_new
    return size_to


# rename get_xy_to
def get_x_y_to(point, dim_from, dim_to, integer=True):
    """Rescale a point from one level's dimensions to another's.

    Parameters
    ----------
    point : tuple[float, float]
        `(x, y)` coordinates expressed in the `dim_from` reference frame, in
        pixels.

    dim_from : tuple[int, int]
        `(width, height)` of the source level, in pixels.

    dim_to : tuple[int, int]
        `(width, height)` of the target level, in pixels.

    integer : bool, optional
        If True, round the converted coordinates with `round`. Default is True.

    Returns
    -------
    point_l : tuple[int, int] | tuple[float, float]
        `(x, y)` coordinates in the `dim_to` reference frame.
    """
    x_0, y_0 = point
    size_x_l = float(dim_to[0])
    size_y_l = float(dim_to[1])
    size_x_0 = float(dim_from[0])
    size_y_0 = float(dim_from[1])

    x_l = x_0 * size_x_l / size_x_0
    y_l = y_0 * size_y_l / size_y_0
    if integer:
        point_l = (round(x_l), round(y_l))
    else:
        point_l = (x_l, y_l)
    return point_l


def vips_to_numpy(img, format=None):
    arr = img.numpy()
    if format == "ushort":
        arr = convertScaleAbs(arr, alpha=(255.0 / 65535.0))
    return arr


def arr_to_pil(img):
    return Image.fromarray(img, mode="RGB")