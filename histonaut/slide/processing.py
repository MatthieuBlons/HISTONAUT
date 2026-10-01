"""Image processing: Tissue masking and patch filtering.

This module provides CPU tools to compute binary tissue masks filter candidate
patches (HOG texture, colour statistics, gray-pixel detection). 

GPU (torch) accelerated counterparts live in `histonaut.slide.torch_processing`.
"""

# General libraries
from functools import partial
import numpy as np
import warnings
import openslide


# Image processing modules
from skimage.color import rgb2gray
from skimage.feature._hog import _hog_channel_gradient, hog
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, square

# Project modules
from histonaut.slide.reader import OpenWSI
from histonaut.slide.utils import get_x_y_to

# Defining valid statistic function
VALID_STAT_FUNC = ["mean", "quantile", "min", "max", "channel_std"]
VALID_AGG_FUNC = ["sum", "mean", "max", "min", "quantile"]


def clear_border(
    img: np.ndarray, margin: tuple[int, int] | int, border_value: int = 1
) -> np.ndarray:
    """Set the border of a 2D image to a constant value.

    The border width is given by `margin` (in pixels) along each axis. 
    
    The input image is copied, not modified in place.

    Parameters
    ----------
    img : np.ndarray
        2D image of shape `(H, W)` (e.g. a grayscale slide thumbnail).

    margin : tuple[int, int] | int
        Border width in pixels along rows and columns. An int applies the same
        width to both axes.

    border_value : int, optional
        Value written into the border. Default is 1 (white in a `[0, 1]`
        grayscale image, i.e. background).

    Returns
    -------
    clean_img : np.ndarray
        Copy of `img`, shape `(H, W)`, with the border set to `border_value`.
    """

    # Parsing margin argument
    if isinstance(margin, int):
        margin = (margin, margin)

    # Retrieving the shapes
    r, c = img.shape
    mr, mc = margin

    # Clearing the border
    clean_img = img.copy()
    clean_img[:mr, :] = border_value
    clean_img[r - mr :, :] = border_value
    clean_img[:, :mc] = border_value
    clean_img[:, c - mc :] = border_value

    # Returning the cleaned image
    return clean_img


def check_borders_correct(
    img: np.ndarray, point: tuple[int, int] | int
) -> tuple[int, int]:
    """Clip a point's coordinates to the image extent.

    If the point lies outside `[0, H] x [0, W]`, each coordinate is clipped to
    that range and a warning is emitted.

    Parameters
    ----------
    img : np.ndarray
        Image whose first two dimensions `(H, W)` define the valid range.

    point : tuple[int, int] | int
        `(row, col)` coordinates to check, in array index order.

    Returns
    -------
    point : tuple[int, int]
        The input point if valid, otherwise the clipped `(row, col)` point.

    Notes
    -----
    The upper bound is inclusive (`row == H` is accepted), which suits slice
    end points but not single-element indexing.
    """

    # Retrieving image shape
    shape = img.shape

    # Check if point is inside img otherwise relocate within image
    if point[0] < 0 or point[1] < 0 or point[0] > shape[0] or point[1] > shape[1]:
        x, y = point
        x = max(0, x)
        y = max(0, y)
        x = min(shape[0], x)
        y = min(shape[1], y)
        warnings.warn("Invalid point: {}, corrected to {}".format(point, (x, y)))
        point = (x, y)

    # Returning point
    return point


def slice_img(
    img: np.ndarray,
    point_0: tuple[int, int] | list[int],
    point_1: tuple[int, int] | list[int] = None,
) -> np.ndarray:
    """Extract the region of an image between two points, or the value at one point.

    Both points are first clipped to the image extent with: `check_borders_correct`.

    Parameters
    ----------
    img : np.ndarray
        Image to slice, indexed as `(row, col, ...)`.

    point_0 : tuple[int, int] | list[int]
        `(row, col)` start point (inclusive).

    point_1 : tuple[int, int] | list[int], optional
        `(row, col)` end point (exclusive). Default is None, in which case the
        value at `point_0` is returned.

    Returns
    -------
    sliced_img : np.ndarray
        `img[row_0:row_1, col_0:col_1]`, or `img[row_0, col_0]` when `point_1`
        is None.

    Raises
    ------
    AssertionError
        If `point_0` is after `point_1` along either axis.
    """

    # Check first point's coordinates
    x_0, y_0 = check_borders_correct(img=img, point=point_0)

    # If no second point, evaluate the image at first image
    if point_1 is None:
        sliced_img = img[x_0, y_0]

    # Otherwise, extract the image slice between the two points
    else:

        # Check last point's coordinates
        x_1, y_1 = check_borders_correct(img=img, point=point_1)

        # Check if first point is before second point
        assert x_0 <= x_1, "Invalid x_axis slicing, point_0: {} and point_1: {}".format(
            point_0, point_1
        )
        assert y_0 <= y_1, "Invalid y_axis slicing, point_0: {} and point_1: {}".format(
            point_0, point_1
        )

        # Extracting image slice
        sliced_img = img[x_0:x_1, y_0:y_1]

    # Returning the sliced image
    return sliced_img


def mask_percentage(
    mask: np.ndarray,
    point: tuple[int, int] | list[int],
    radius: int | tuple[int] | np.ndarray,
    mask_tolerance: float = 0.5,
) -> bool:
    """Check whether a patch overlaps the tissue mask by more than a tolerance.

    The patch is the square window `[point - radius, point + radius]`
    (inclusive) in mask pixels, clipped to the mask extent. 
    
    The overlap score is the fraction of mask pixels set inside this window.

    Parameters
    ----------
    mask : np.ndarray
        Binary tissue mask of shape `(H, W)`.

    point : tuple[int, int] | list[int]
        `(row, col)` centre of the patch, in mask pixels.

    radius : int | tuple[int] | np.ndarray
        Half-size of the patch in mask pixels, either shared or per axis.

    mask_tolerance : float, optional
        Minimum fraction of tissue pixels (in `[0, 1]`) required to accept the
        patch. Default is 0.5.

    Returns
    -------
    is_accepted : bool
        True if the tissue fraction is strictly greater than `mask_tolerance`.

    Raises
    ------
    AssertionError
        Propagated from `slice_img` if the clipped window is invalid.
    """
    # Extracting mask value at patch location
    sub_mask = slice_img(
        img=mask,
        point_0=(np.array(point) - np.array(radius)).tolist(),
        point_1=(np.array(point) + np.array(radius) + 1).tolist(),
    )

    # Compute percentage of overlapping in the patch
    score = sub_mask.sum() / (sub_mask.shape[0] * sub_mask.shape[1])

    # Returning if the patch is valid or not
    return score > mask_tolerance


def compute_luminosity_mask(
    slide: OpenWSI | openslide.OpenSlide,
    mask_level: int = 0,
    margin: int | tuple[int, int] = 0,
    intensity_thresh: tuple[float, float] = (0.1, 0.8),
    neighborhood_size: int = 2,
) -> np.ndarray:
    """Compute a tissue mask by thresholding grayscale luminosity.

    The whole slide is read at `mask_level`, converted to grayscale in
    `[0, 1]`, its border of width `margin` is set to white, and pixels whose
    luminosity lies strictly between the two values of `intensity_thresh` are
    kept as tissue. 
    
    The mask is then cleaned with a morphological closing followed by an opening.

    Parameters
    ----------
    slide : OpenWSI | openslide.OpenSlide
        Slide to process. Must expose `read_region` and `level_dimensions`.

    mask_level : int, optional
        Pyramid level read to compute the mask. Default is 0 (full
        resolution).

    margin : int | tuple[int, int], optional
        Border width, in pixels at `mask_level`, cleared before thresholding.
        Default is 0.

    intensity_thresh : tuple[float, float], optional
        Lower and upper grayscale bounds in `[0, 1]`; very dark pixels
        (artefacts) and bright pixels (glass) are excluded. Default is
        `(0.1, 0.8)`.

    neighborhood_size : int, optional
        Side, in pixels, of the square footprint used for closing and opening.
        Default is 2.

    Returns
    -------
    mask : np.ndarray
        Binary tissue mask of shape `(H, W)` (bool) at `mask_level`.

    Raises
    ------
    ValueError
        If the computed mask is empty.
    """
    # Extracting region from WSI slide at right level
    img = slide.read_region(
        location=(0, 0), level=mask_level, size=slide.level_dimensions[mask_level]
    )

    # Convert to numpy array
    if not isinstance(img, np.ndarray):
        img = np.array(img)[:, :, :3]

    # Parsing to grayscale and clearing border
    img_gray = rgb2gray(img)
    img_gray = clear_border(img_gray, margin=margin)

    mask = opening(
        closing(
            np.logical_and(
                img_gray < max(intensity_thresh), img_gray > min(intensity_thresh)
            ),
            footprint=square(neighborhood_size),
        ),
        footprint=square(neighborhood_size),
    )

    # check it's not empty
    if mask.sum() == 0:
        msg = "Empty tissue mask computed."
        raise ValueError(msg)

    return mask


def compute_otsu_mask(
    slide: OpenWSI | openslide.OpenSlide,
    mask_level: int,
    margin: int | tuple[int, int] = 0,
    intensity_thresh: tuple[float, float] = (0.1, 0.98),
    neighborhood_size: int = 2,
) -> np.ndarray:
    """Compute a tissue mask with Otsu thresholding of the grayscale slide.

    The whole slide is read at `mask_level`, converted to grayscale in
    `[0, 1]` and its border of width `margin` is set to white. The Otsu
    threshold `t` is computed only on pixels strictly inside
    `intensity_thresh`; tissue pixels are those with
    `min(intensity_thresh) < gray < t`. 
    
    The mask is then cleaned with a morphological closing followed by an opening.

    Parameters
    ----------
    slide : OpenWSI | openslide.OpenSlide
        Slide to process. Must expose `read_region` and `level_dimensions`.

    mask_level : int
        Pyramid level read to compute the mask.

    margin : int | tuple[int, int], optional
        Border width, in pixels at `mask_level`, cleared before thresholding.
        Default is 0.

    intensity_thresh : tuple[float, float], optional
        Lower and upper grayscale bounds in `[0, 1]` of the pixels used to fit
        the Otsu threshold. Default is `(0.1, 0.98)`.

    neighborhood_size : int, optional
        Side, in pixels, of the square footprint used for closing and opening.
        Default is 2.

    Returns
    -------
    mask : np.ndarray
        Binary tissue mask of shape `(H, W)` (bool) at `mask_level`.

    Raises
    ------
    ValueError
        If the computed mask is empty.
    """
    # Extracting region from WSI slide at right level
    img = slide.read_region(
        location=(0, 0), level=mask_level, size=slide.level_dimensions[mask_level]
    )

    # Convert to numpy array
    if not isinstance(img, np.ndarray):
        img = np.array(img)[:, :, :3]

    # Parsing to grayscale and clearing border
    img_gray = rgb2gray(img)
    img_gray = clear_border(img_gray, margin=margin)
    img_gray = img_gray.flatten()

    # Computing Otsu threshold on filtered intensity
    pixels_int = img_gray[
        np.logical_and(
            img_gray > min(intensity_thresh), img_gray < max(intensity_thresh)
        )
    ]
    t = threshold_otsu(pixels_int)

    # Create a mask for filtering the image based on Otsu threshold with opening (remove bright spots and dark cracks)
    # And closing (remove dark spots and bright cracks) post-processing
    mask = opening(
        closing(
            np.logical_and(img_gray < t, img_gray > min(intensity_thresh)).reshape(
                img.shape[:-1]
            ),
            footprint=square(neighborhood_size),
        ),
        footprint=square(neighborhood_size),
    )

    # check it's not empty
    if mask.sum() == 0:
        msg = "Empty tissue mask computed."
        raise ValueError(msg)

    # Returning the mask
    return mask


def get_patch_mask(
    coords: np.ndarray, mask_dim: tuple[int, int], slide_dim: tuple[int, int]
) -> np.ndarray:
    """Build a label map of patches at mask resolution.

    Each patch is rescaled from slide to mask dimensions with `get_x_y_to` and
    painted with its index in `coords`; uncovered pixels are -1 (background).
    Overlapping patches are overwritten by later ones.

    Parameters
    ----------
    coords : np.ndarray
        Patch coordinates of shape `(N, 4)`, `(x, y, size_x, size_y)` in pixels
        of the slide space given by `slide_dim` (typically level 0).

    mask_dim : tuple[int, int]
        Dimensions of the output mask; also used as the output array shape.

    slide_dim : tuple[int, int]
        Dimensions of the slide space of `coords`, in the same axis order as
        `mask_dim`.

    Returns
    -------
    patch_mask : np.ndarray
        Float array of shape `mask_dim` holding the patch index at each pixel,
        or -1 for background.

    Notes
    -----
    `x` indexes the first array axis and `y` the second, so `mask_dim` and
    `slide_dim` must follow the same `(x, y)` order as `coords`.
    """

    # Retrieving downsample factor
    downsample_factor = (slide_dim[0] / mask_dim[0], slide_dim[1] / mask_dim[1])

    # Initialization of the mask for the patches, -1 corresponds to background
    patch_mask = -np.ones(mask_dim)

    # For each patch, put its index at its corresponding position in the mask
    for i, (x, y, size_x, size_y) in enumerate(coords):
        mask_patch_size = (
            max(1, int(size_x / downsample_factor[0])),
            max(1, int(size_y / downsample_factor[1])),
        )
        x, y = get_x_y_to(
            point=(x, y), dim_from=slide_dim, dim_to=mask_dim, integer=True
        )
        patch_mask[x : (x + mask_patch_size[0]), y : (y + mask_patch_size[1])] = i

    # Returning mask for the patches
    return patch_mask


def compute_channel_gradient(image: np.ndarray, channel_axis: int = None):
    """Compute row and column gradients of an image, HOG style.

    For a multichannel image, the gradient is computed per channel with
    scikit-image's `_hog_channel_gradient` and, at each pixel, the gradient of
    the channel with the largest magnitude is kept.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image `(H, W)` or multichannel image `(H, W, C)`; cast to
        float.

    channel_axis : int, optional
        Channel axis of `image`. Default is None, meaning a single-channel
        image.

    Returns
    -------
    g_row : np.ndarray
        Gradient along rows, shape `(H, W)`.

    g_col : np.ndarray
        Gradient along columns, shape `(H, W)`.

    Notes
    -----
    The multichannel branch indexes channels as the last axis, whatever the
    value of `channel_axis`.
    """

    # Parsing image as type float
    image = image.astype("float", copy=False)

    # Retrieving multichannel args
    multichannel = False
    if channel_axis is not None:
        multichannel = True

    # Computing gradient per channel
    if multichannel:

        # Initialization of arrays for image gradients
        g_row_by_ch = np.empty_like(image, dtype=image.dtype)
        g_col_by_ch = np.empty_like(image, dtype=image.dtype)
        g_magn = np.empty_like(image, dtype=image.dtype)

        # Compute the gradients for each channel and the global gradient's magnitude
        for idx_ch in range(image.shape[channel_axis]):
            (
                g_row_by_ch[:, :, idx_ch],
                g_col_by_ch[:, :, idx_ch],
            ) = _hog_channel_gradient(image[:, :, idx_ch])
            g_magn[:, :, idx_ch] = np.hypot(
                g_row_by_ch[:, :, idx_ch], g_col_by_ch[:, :, idx_ch]
            )

        # For each pixel select the channel with the highest gradient magnitude
        idcs_max = g_magn.argmax(axis=channel_axis)
        rr, cc = np.meshgrid(
            np.arange(image.shape[0]),
            np.arange(image.shape[1]),
            indexing="ij",
            sparse=True,
        )
        g_row = g_row_by_ch[rr, cc, idcs_max]
        g_col = g_col_by_ch[rr, cc, idcs_max]

    # Otherwise, compute overall gradient
    else:
        g_row, g_col = _hog_channel_gradient(image)

    # Returning the gradients
    return g_row, g_col


def compute_hog(
    image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (16, 16),
    cells_per_block: tuple[int, int] = (1, 1),
    use_grayscale: bool = False,
    hog_image: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Compute HOG descriptors, and optionally the HOG image, of an image.

    Thin wrapper around `skimage.feature.hog`.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape `(H, W, 3)`.

    orientations : int, optional
        Number of orientation bins. Default is 9.

    pixels_per_cell : tuple[int, int], optional
        Cell size in pixels. Default is `(16, 16)`.

    cells_per_block : tuple[int, int], optional
        Number of cells per normalisation block. Default is `(1, 1)`.

    use_grayscale : bool, optional
        If True, convert `image` to grayscale first; otherwise use the RGB
        channels (last axis). Default is False.

    hog_image : bool, optional
        If True, also return the HOG visualisation image. Default is False.

    Returns
    -------
    features : np.ndarray
        Flattened HOG descriptors. Returned alone when `hog_image` is False.

    hog_image : np.ndarray
        HOG visualisation image of shape `(H, W)`, returned as the second
        element of a tuple `(features, hog_image)` when `hog_image` is True.
    """

    # Convert image to grayscale and define channel axis
    if use_grayscale:
        image = rgb2gray(image)
        channel_axis = None
    else:
        channel_axis = -1

    # Parsing visualize default arg
    visualize = False
    if hog_image:
        visualize = True

    # Compute HOG descriptors
    res = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=visualize,
        channel_axis=channel_axis,
    )

    # Returning the HOG features and image
    return res


def hog_selection(
    img_list: list[np.ndarray],
    hog_thresh: float = 0.5,
    hog_std_thresh: float = None,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (16, 16),
    cells_per_block: tuple[int, int] = (1, 1),
    use_grayscale: bool = False,
    *args,
    **kwargs,
):
    """Select images with enough texture according to their HOG image.

    For each image, the HOG visualisation image is computed with
    `compute_hog`; an image is kept if the mean of its HOG image is
    `>= hog_thresh` and, when `hog_std_thresh` is given, its standard
    deviation is `>= hog_std_thresh`.

    Parameters
    ----------
    img_list : list[np.ndarray]
        Images to filter, each of shape `(H, W, 3)` and all of the same shape.

    hog_thresh : float, optional
        Minimum mean value of the HOG image. Default is 0.5.

    hog_std_thresh : float, optional
        Minimum standard deviation of the HOG image. Default is None (no std
        criterion).

    orientations : int, optional
        Number of orientation bins. Default is 9.

    pixels_per_cell : tuple[int, int], optional
        Cell size in pixels. Default is `(16, 16)`.

    cells_per_block : tuple[int, int], optional
        Number of cells per normalisation block. Default is `(1, 1)`.

    use_grayscale : bool, optional
        If True, compute HOG on grayscale images. Default is False.

    *args
        Ignored; accepted for a common selection-function signature.

    **kwargs
        Ignored; accepted for a common selection-function signature.

    Returns
    -------
    idx_selected : np.ndarray
        Indices of the selected images, shape `(n_selected,)` (squeezed, so a
        0-d array when exactly one image is selected).
    """

    # Initialization of a list to store hog values
    hog_list = []

    # Computing HOG descriptors
    for i in range(len(img_list)):
        _, hog_img = compute_hog(
            image=img_list[i],
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            use_grayscale=use_grayscale,
            hog_image=True,
        )
        hog_list.append(hog_img)

    # Filtering the patches based on hog mean values
    hog_filter = (
        np.array(hog_list).reshape(len(hog_list), -1).mean(axis=-1) >= hog_thresh
    )

    # Filtering the patches based on hog std values as well
    if hog_std_thresh is not None:
        hog_filter &= (
            np.array(hog_list).reshape(len(hog_list), -1).std(axis=-1) >= hog_std_thresh
        )

    # Returning the indices of the selected patch
    return np.argwhere(hog_filter).squeeze()


def patch_mean_pooling(
    img: np.ndarray, patch_size: int | tuple[int, int]
) -> np.ndarray:
    """Average-pool an image over non-overlapping windows, ignoring NaN pixels.

    Pixels with a NaN in any channel are excluded from the window mean; a
    window with only NaN pixels yields NaN. Trailing rows / columns that do
    not fill a full window are dropped.

    Parameters
    ----------
    img : np.ndarray
        Image of shape `(H, W, C)`, possibly containing NaN (e.g. removed
        background).

    patch_size : int | tuple[int, int]
        Window size in pixels, shared or `(rows, cols)`.

    Returns
    -------
    mean_img : np.ndarray
        Float array of shape `(H // patch_rows, W // patch_cols, C)` with the
        per-window channel means.
    """

    # Parsing patch size
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    # Retrieving number of patches per row and columns
    n_patch_row, n_patch_col = (
        img.shape[0] // patch_size[0],
        img.shape[1] // patch_size[1],
    )

    # Initialization of the local average image
    mean_img = np.zeros((n_patch_row, n_patch_col, img.shape[-1]))

    # For each patch, compute the average of the pixels
    for i in range(n_patch_row):
        for j in range(n_patch_col):
            patch = img[
                i * patch_size[0] : min((i + 1) * patch_size[0], img.shape[0]),
                j * patch_size[1] : min((j + 1) * patch_size[1], img.shape[1]),
            ].reshape(-1, img.shape[-1])
            if sum(~np.isnan(patch).any(axis=-1)) > 0:
                mean_img[i, j, :] = patch[~np.isnan(patch).any(axis=-1), :].mean(axis=0)
            else:
                mean_img[i, j, :] = np.nan

    # Returning the averaged image per patch
    return mean_img


def compute_channel_std(
    img_list: list[np.ndarray] | np.ndarray, aggregate: str = "mean", q: float = 0.5
):
    """Compute an aggregated across-channel standard deviation per image.

    For every pixel, the standard deviation across channels is computed (low
    values mean gray pixels); these per-pixel values are then aggregated per
    image with the NumPy function named by `aggregate`, ignoring NaN pixels.

    Parameters
    ----------
    img_list : list[np.ndarray] | np.ndarray
        Images stacked along the first axis, typically flattened as
        `(N, n_pixels, C)`.

    aggregate : str, optional
        Name of the NumPy aggregation function, one of `VALID_AGG_FUNC`.
        Default is ``"mean"``.

    q : float, optional
        Quantile in `[0, 1]` used when `aggregate` is ``"quantile"``. Default
        is 0.5.

    Returns
    -------
    stat_values : np.ndarray
        Aggregated channel standard deviation, shape `(N, 1)`.

    Raises
    ------
    AssertionError
        If `aggregate` is not an attribute of `numpy`.
    """
    # Retrieving the aggregate function
    assert hasattr(np, aggregate), (
        "Aggregate function {} not implemented.".format(aggregate)
        + f"Please choose a valid argument between: {', '.join(VALID_AGG_FUNC)}."
    )
    agg_func = getattr(np, aggregate)
    if aggregate == "quantile":
        agg_func = partial(agg_func, q=q)

    # Computing channel std
    channel_std = np.array(img_list).std(axis=-1)

    # Aggregated the results across images in the list
    if np.isnan(channel_std).any():
        stat_values = np.apply_along_axis(
            arr=channel_std, func1d=lambda x: agg_func(x[~np.isnan(x)]), axis=1
        )
    else:
        stat_values = agg_func(
            channel_std[~np.isnan(channel_std)].reshape(len(img_list), -1), axis=-1
        )

    # Returning the results
    return stat_values.reshape(len(img_list), -1)


def color_filter_selection(
    img_list: list[np.ndarray] | np.ndarray,
    color_thresh: tuple[int, int, int] | int,
    stat_fct: str = "quantile",
    q: float = 0.05,
    remove_background: bool = True,
    background_thresh: tuple[int, int, int] | int = 245,
    local_average: bool = True,
    filter_size: tuple[int, int] | int = 16,
    filter_all: bool = False,
    background_average: bool = True,
    aggregate: str = "mean",
) -> list[int]:
    """Select images whose per-channel colour statistic reaches a threshold.

    Pipeline per image:

    1. If `remove_background` and not `background_average`, pixels with any
       channel `>= background_thresh` are set to NaN.
    2. If `local_average`, the image is mean-pooled over `filter_size`
       windows (`patch_mean_pooling`).
    3. If `remove_background` and `background_average`, pixels (or pooled
       windows) with any channel `>= background_thresh` are dropped.
    4. The statistic `stat_fct` is computed over pixels, per channel (or as
       the aggregated channel std for ``"channel_std"``), and compared with
       `color_thresh` (`>=`).

    Parameters
    ----------
    img_list : list[np.ndarray] | np.ndarray
        RGB images of shape `(H, W, 3)` (uint8 range), all of the same shape.

    color_thresh : tuple[int, int, int] | int
        Minimum value of the statistic per channel (R, G, B); an int applies
        to all channels.

    stat_fct : str, optional
        Statistic to compute, one of `VALID_STAT_FUNC`. Default is
        ``"quantile"``.

    q : float, optional
        Quantile in `[0, 1]` used when `stat_fct` is ``"quantile"``. Default
        is 0.05.

    remove_background : bool, optional
        Whether to exclude background pixels before computing the statistic.
        Default is True.

    background_thresh : tuple[int, int, int] | int, optional
        Per-channel intensity at or above which a pixel is background; an int
        applies to all channels. Default is 245.

    local_average : bool, optional
        Whether to mean-pool the image before computing the statistic.
        Default is True.

    filter_size : tuple[int, int] | int, optional
        Window size in pixels for local averaging. Default is 16.

    filter_all : bool, optional
        If True, all channels must meet the threshold; otherwise at least one.
        Default is False.

    background_average : bool, optional
        If True, background is removed after local averaging (on pooled
        windows); otherwise on raw pixels before averaging. Default is True.

    aggregate : str, optional
        Aggregation function (one of `VALID_AGG_FUNC`) passed to
        `compute_channel_std` when `stat_fct` is ``"channel_std"``. Default is
        ``"mean"``.

    Returns
    -------
    idx_selected : list[int]
        Indices of the selected images in `img_list`.

    Raises
    ------
    AssertionError
        If `color_thresh` is None or `stat_fct` is not in `VALID_STAT_FUNC`.
    """

    # Parsing color threshold argument
    if isinstance(color_thresh, int):
        color_thresh = (color_thresh, color_thresh, color_thresh)
    if remove_background and isinstance(background_thresh, int):
        background_thresh = (background_thresh, background_thresh, background_thresh)

    # Checking valid value of color_thresh
    assert (
        color_thresh is not None
    ), f"No color threshold was given to filter the patches. \nPlease specify a `color_thresh` value."

    # Retrieving the statistic to compute
    assert stat_fct in VALID_STAT_FUNC, (
        f"The statistic function given is not implemented. "
        + f"\nPlease specify a valid `stat_fct` value among: {', '.join(VALID_STAT_FUNC)}."
    )
    # Retrieving the statistic function
    if stat_fct == "quantile":
        stat_fn = partial(np.quantile, q=q, axis=-2)
    elif stat_fct == "channel_std":
        stat_fn = partial(compute_channel_std, aggregate=aggregate)
    else:
        stat_fn = partial(getattr(np, stat_fct), axis=-2)

    # Removing background on image before computing the local average
    if remove_background and not background_average:
        clean_imgs = []
        for img in img_list:
            clean_img = img.copy().astype(float)
            clean_img[(clean_img >= background_thresh).any(axis=-1), :] = None
            clean_imgs.append(clean_img)
        img_list = clean_imgs

    # Local averaging of the input
    if local_average:
        mean_imgs = []
        for img in img_list:
            mean_imgs.append(patch_mean_pooling(img=img, patch_size=filter_size))
        img_list = mean_imgs

    # Removing background and compute the statistic for each cleaned image due to different dimensions
    if remove_background and background_average:
        stat_values = []
        for img in img_list:
            clean_img = img[(img < background_thresh).all(axis=-1), :]
            stat_values.append(stat_fn(clean_img.reshape(1, -1, clean_img.shape[-1])))
        stat_values = np.array(stat_values).squeeze(-2)

    # Otherwise, compute the statistic on all images in one shot since they have same dimensions
    else:
        stat_values = stat_fn(
            np.array(img_list).reshape(len(img_list), -1, img_list[0].shape[-1])
        )

    # Filter for the images based on the channels' minimum color threshold
    img_filter = stat_values >= np.array(color_thresh)
    if filter_all:
        img_filter = img_filter.all(axis=-1)
    else:
        img_filter = img_filter.any(axis=-1)

    # Retrieving the indices selected
    idx_selected = np.argwhere(img_filter).squeeze(-1).tolist()

    # Returning selected indices
    return idx_selected


def filter_gray_img(
    img_list: list[np.ndarray] | np.ndarray,
    std_thresh: int,
    remove_background: bool = True,
    background_thresh: int | tuple[int, int, int] = 245,
    local_average: bool = True,
    filter_size: int | tuple[int, int] = 16,
    aggregate: str = "mean",
):
    """Discard mostly gray images using the across-channel standard deviation.

    Calls `color_filter_selection` with `stat_fct="channel_std"`: an image is
    kept if its aggregated channel std (after optional background removal and
    local averaging) is `>= std_thresh`.

    Parameters
    ----------
    img_list : list[np.ndarray] | np.ndarray
        RGB images of shape `(H, W, 3)` (uint8 range), all of the same shape.

    std_thresh : int
        Minimum aggregated across-channel standard deviation to keep an image.

    remove_background : bool, optional
        Whether to exclude background pixels before computing the channel std.
        Default is True.

    background_thresh : int | tuple[int, int, int], optional
        Per-channel intensity at or above which a pixel is background. Default
        is 245.

    local_average : bool, optional
        Whether to mean-pool the image before computing the channel std.
        Default is True.

    filter_size : int | tuple[int, int], optional
        Window size in pixels for local averaging. Default is 16.
        
    aggregate : str, optional
        Aggregation function (one of `VALID_AGG_FUNC`) applied to the per-pixel
        channel std. Default is ``"mean"``.

    Returns
    -------
    idx_selected : list[int]
        Indices of the non-gray images in `img_list`.

    Raises
    ------
    AssertionError
        Propagated from `color_filter_selection` / `compute_channel_std`.
    """

    # Color filter based on channel std
    idx_selected = color_filter_selection(
        img_list=img_list,
        color_thresh=std_thresh,
        stat_fct="channel_std",
        remove_background=remove_background,
        background_thresh=background_thresh,
        local_average=local_average,
        filter_size=filter_size,
        aggregate=aggregate,
    )

    # Returning selected indices
    return idx_selected
