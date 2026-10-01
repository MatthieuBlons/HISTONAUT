"""Patch grid helpers.
"""

# General libraries
import itertools

# Data libraries
import numpy as np

# Image reader
import openslide
import PIL

# Project modules
from histonaut.slide.reader import OpenWSI, get_slide_reader


def grid_blob(point_start, point_end, space):
    """Build the top-left coordinates of a regular grid between two points.

    Parameters
    ----------
    point_start : tuple[int, int]
        `(x, y)` of the first grid point (included).

    point_end : tuple[int, int]
        `(x, y)` upper bounds of the grid (excluded).

    space : tuple[int, int]
        Grid step `(step_x, step_y)`.

    Returns
    -------
    grid : list[tuple[int, int]]
        `(x, y)` grid points, ordered by `x` then `y`.
    """
    size_x, size_y = space
    list_x = range(point_start[0], point_end[0], size_x)
    list_y = range(point_start[1], point_end[1], size_y)
    return list(itertools.product(list_x, list_y))


def get_bag_of_tiles(slide, xywh, res_to_view=0):
    """Read the patches located at the given coordinates.

    Parameters
    ----------
    slide : str | OpenWSI | openslide.OpenSlide
        Slide, or path to a slide opened with `get_slide_reader`.

    xywh : np.ndarray | list[tuple[int, int, int, int]]
        Patch coordinates `(N, 4)` as `(x, y, w, h)`; `w` and `h` are read at
        level `res_to_view`.

    res_to_view : int, optional
        Pyramid level to read the patches from. Default is 0.

    Returns
    -------
    tiles : np.ndarray | list[np.ndarray]
        `(h, w, 3)` RGB patches; a single array when only one patch is read.
    """
    bag = []
    if isinstance(slide, str):
        reader = get_slide_reader(slide)
        slide = reader(slide)

    for pos in xywh:
        x, y, w, h = pos
        tile = slide.read_region(location=(x, y), level=res_to_view, size=(w, h))
        if not isinstance(tile, np.ndarray):
            tile = np.array(tile[:, :, :3])
        bag.append(tile)

    if len(bag) == 1:
        return bag[0]
    else:
        return bag


def get_patch_coverage(
    slide: str | OpenWSI | openslide.OpenSlide,
    patches: np.ndarray,
    region: tuple[int, int] = (0, 0),
    size: tuple[int, int] | None = None,
    analyse_level: int = 0,
    level_to_view: int = 0,
    numpy: bool = True,
    coverage_only: bool = True,
):
    """Draw a binary mask of the area covered by patches.

    Patches intersecting the region are shifted to region coordinates, rescaled
    from `analyse_level` to `level_to_view` and painted with 1 on a zero mask.

    Parameters
    ----------
    slide : str | OpenWSI | openslide.OpenSlide
        Slide, or path to a slide opened with `get_slide_reader`.

    patches : np.ndarray
        Patch coordinates, shape `(N, 4)` xywh, expressed at `analyse_level`.

    region : tuple[int, int], optional
        Top-left `(x, y)` of the region to view, at `analyse_level`. Default is
        `(0, 0)`.

    size : tuple[int, int] | None, optional
        `(width, height)` of the region at `analyse_level`. Default is None
        (whole level `analyse_level`).

    analyse_level : int, optional
        Pyramid level in which `patches`, `region` and `size` are expressed.
        Default is 0.

    level_to_view : int, optional
        Pyramid level of the output mask; clipped to the last available level.
        Default is 0.

    numpy : bool, optional
        Passed to `read_region` to get the region as an array. Default is True.

    coverage_only : bool, optional
        If True, return only the coverage mask. Default is True.

    Returns
    -------
    coverage : np.ndarray
        `float32` mask of shape `(height, width)` at `level_to_view`, 1 where
        patches lie and 0 elsewhere.
        
    wsi : np.ndarray | PIL.Image.Image
        Region read at `level_to_view`; only returned if `coverage_only` is
        False.

    Raises
    ------
    AssertionError
        If `patches` is an array whose second dimension is not 4.
    """
    # read slide if path is provided
    if isinstance(slide, str):
        reader = get_slide_reader(slide)
        slide = reader(slide)

    # check downsampling level
    if level_to_view >= slide.level_count:
        print(
            f"downsampling level={level_to_view} is not accessible, use level={slide.level_count-1} instead"
        )
        level_to_view = slide.level_count - 1

    down_analyse = slide.level_downsamples[analyse_level]
    down_view = slide.level_downsamples[level_to_view]

    if not size:
        size = slide.level_dimensions[analyse_level]
        dim_view = slide.level_dimensions[level_to_view]
    else:
        dim_view = (
            int(size[0] * (down_analyse / down_view)),
            int(size[1] * (down_analyse / down_view)),
        )

    # get whole image
    wsi = slide.read_region(
        location=region, level=level_to_view, size=dim_view, numpy=numpy
    )


    if isinstance(patches, np.ndarray): 
        assert patches.shape[1] == 4, "custom_xywh must be a (n, 4) array of int [[x, y, w, h]]"
    
    # Vectorized coordinates and sizes
    xs = patches[:, 0]
    ys = patches[:, 1]
    ws = patches[:, 2]
    hs = patches[:, 3] 

    # Filter tiles that intersect the selected region
    x0, y0 = region
    W, H = size

    mask = (xs + ws >= x0) & (xs < x0 + W) & (ys + hs >= y0) & (ys < y0 + H)

    xs, ys, ws, hs = (
        xs[mask],
        ys[mask],
        ws[mask],
        hs[mask],
    )

    # Shift to region coordinates
    xs = xs - x0
    ys = ys - y0

    # or apply get_size_to() get_x_y_to()
    x_scaled = (xs * dim_view[0] / size[0]).astype(int)
    y_scaled = (ys * dim_view[1] / size[1]).astype(int)
    w_scaled = np.maximum((ws * down_analyse / down_view).astype(int), 1)
    h_scaled = np.maximum((hs * down_analyse / down_view).astype(int), 1)

    # Create transparent overlay
    coverage = np.zeros((dim_view[1], dim_view[0]), dtype=np.float32)  # gray scale
    for x, y, w, h in zip(x_scaled, y_scaled, w_scaled, h_scaled):
        coverage[y : y + h + 1, x : x + w + 1] = 1

    if not numpy:
        overlay = PIL.Image.fromarray(coverage, mode="L")

    if coverage_only:
        return coverage
    else:
        return coverage, wsi