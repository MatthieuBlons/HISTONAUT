"""Slide and patch visualization.

This module provides helpers to inspect whole-slide images such as:
slide thumbnails, overlays of patch grids and tissue segmentation masks,
maps of patch-level features, RGB encodings on top of the slide,
image mosaics and text annotation boxes.
"""

# General libraries
import os
from typing import Any
from operator import itemgetter
from pathlib import Path

# Data libraries
import numpy as np
import pandas as pd

# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib import axes, patches

# Image transform
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter, gaussian_filter1d

# Image readers
import cv2
from PIL import Image
import openslide

# Project modules
from histonaut.slide.utils import get_x_y_to, get_size_to
from histonaut.slide.reader import OpenWSI, get_slide_reader, get_slide_whole


def mosaic(
    imgs: list[str] | list[Path] | list[np.ndarray],
    size: tuple[int, int] | None = None,
    max_img: int | None = None,
    ax: Any | None = None,
    title: str | None = None,
):
    """Display a list of images as a square mosaic on a matplotlib axis.

    Images are placed row by row on a `ceil(sqrt(max_img))` x `ceil(sqrt(max_img))`
    grid.

    Images given as file paths are opened with PIL (by default), converted to RGB and
    resized to `size` when needed; NumPy arrays are pasted as is.

    Parameters
    ----------
    imgs : list[str] | list[np.ndarray]
        Images to display, either paths to image files or RGB arrays of shape
        `(H, W, 3)`. The type of the first element decides how the list is read.

    size : tuple[int, int], optional
        Size of each cell of the mosaic. Default is None, meaning the largest first
        and second dimensions found in `imgs` (PIL `(width, height)` for paths,
        array `shape[:2]` for arrays).

    max_img : int, optional
        Maximum number of images to display. Default is None, meaning
        `len(imgs)`.

    ax : matplotlib.axes.Axes, optional
        Axis to draw on. Default is None, meaning a new 15 x 15 inches figure is
        created.

    title : str, optional
        Title of the axis. Default is None (no title).

    Returns
    -------
    None
        The mosaic is drawn on `ax` (`plt.show` is not called).

    Raises
    ------
    Exception
        If `imlist` is empty.
    TypeError
        If the elements of `imlist` are neither `str` nor `np.ndarray`.
    """
    if not imgs:
        raise Exception("List of images is empty")

    if not size:
        imsizes = []
        if isinstance(imgs[0], str):
            for name in imgs:
                with Image.open(name) as img:
                    imsizes.append(img.size)
        elif isinstance(imgs[0], np.ndarray):
            for array in imgs:
                imsizes.append(array.shape)
        max_width, max_height = (
            max(imsizes, key=itemgetter(0))[0],
            max(imsizes, key=itemgetter(1))[1],
        )
        size = (max_width, max_height)
    if not max_img:
        max_img = len(imgs)
    if not ax:
        fig = plt.figure(figsize=(15, 15))
        ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1, fig=fig)
    ax.set_axis_off()
    grid_size = (
        np.ceil(np.sqrt(max_img)).astype(int),
        np.ceil(np.sqrt(max_img)).astype(int),
        3,
    )
    grid = np.zeros(
        (grid_size[0] * size[0], grid_size[1] * size[1], grid_size[2])
    ).astype(int)
    # populate the mosaic
    gridpos = 0
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            if gridpos < max_img:
                # make it possible to use list of numpy as the image list
                if isinstance(imgs[0], str):
                    with Image.open(imgs[gridpos]) as img:
                        if img.size != size:
                            img_resized = img.resize(size)
                            np_img = np.array(img_resized.convert(mode="RGB"))
                        else:
                            np_img = np.array(img.convert(mode="RGB"))
                        grid[
                            x * size[0] : (x + 1) * size[0],
                            y * size[1] : (y + 1) * size[1],
                            :,
                        ] = np_img[:, :, :]
                        gridpos += 1
                elif isinstance(imgs[0], np.ndarray):
                    np_img = imgs[gridpos]
                    grid[
                        x * size[0] : (x + 1) * size[0],
                        y * size[1] : (y + 1) * size[1],
                        :,
                    ] = np_img[:, :, :]
                    gridpos += 1
                else:
                    raise TypeError(
                        "imlist should be a list of str or list of ndarrays"
                    )
    # show image
    if title:
        ax.set_title(title)
    ax.imshow(grid)



def add_annotation_box(
    img: np.ndarray,
    text: list[str],
    title: str = "",
    text_y_spacing: int = 25,
    text_box_height: int = 150,
    text_box_width: int = 350,
    text_box_color: tuple[int, int, int] | str = None,
    alpha=0.25,
    title_fontscale: float = 0.75,
    thickness: int = 1,
    text_fontscale: float = 0.5,
    color: str | tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Add a text annotation box in the top-left corner of an image.

    The box covers `img[:text_box_height, :text_box_width]`. The title is written on
    the first line and each element of `text` on the following lines, using OpenCV
    `cv2.putText` (Hershey simplex font, anti-aliased).

    Parameters
    ----------
    img : np.ndarray
        Image of shape `(H, W, 3)`, dtype `uint8`. Modified in place.

    text : list[str]
        Lines of text to write in the box, one element per line.

    title : str, optional
        Title written on the first line of the box. Default is "".

    text_y_spacing : int, optional
        Vertical spacing between lines, in pixels. Default is 25.

    text_box_height : int, optional
        Height of the box, in pixels. Default is 150.

    text_box_width : int, optional
        Width of the box, in pixels. Default is 350.

    text_box_color : tuple[int, int, int] | str, optional
        Background color of the box. Default is None, meaning no colored background:
        the box region is darkened by multiplying it by `alpha`.

    alpha : float, optional
        Weight of the original image in the box region (the background color gets
        `1 - alpha`). Default is 0.25.

    title_fontscale : float, optional
        Font scale of the title. Default is 0.75.

    thickness : int, optional
        Thickness of the text strokes. Default is 1.

    text_fontscale : float, optional
        Font scale of the text lines. Default is 0.5.

    color : str | tuple[int, int, int], optional
        Color of the text. Default is (255, 255, 255) (white).

    Returns
    -------
    img : np.ndarray
        The annotated image (same object as the input `img`).

    Notes
    -----
    The horizontal text offset is 3% of `img.shape[0]` (the image height).
    `text_box_height` is not adapted to the number of lines of `text`.
    """

    # Computing x offset
    text_x_offset = int(img.shape[0] * 0.03)

    # Adding background transparency
    if text_box_color is None:
        img[:text_box_height, :text_box_width] = (
            img[:text_box_height, :text_box_width] * alpha
        ).astype(np.uint8)
    else:
        # Create an overlay of the same size as the ROI, filled with the background color
        roi = img[:text_box_height, :text_box_width].copy()
        text_box = np.full_like(roi, text_box_color, dtype=np.uint8)
        overlay = cv2.addWeighted(text_box, 1 - alpha, roi, alpha, 0)
        img[:text_box_height, :text_box_width] = overlay

    # Adding title
    cv2.putText(
        img,
        text=title,
        org=(text_x_offset, text_y_spacing),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=title_fontscale,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    # Add annotation
    for i, line in enumerate(text):
        cv2.putText(
            img,
            text=line,
            org=(text_x_offset, text_y_spacing * (i + 2)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=text_fontscale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    # Returning the annotated image
    return img


def add_slide_annotation_box(
    img: np.ndarray, slide_attrs: dict, title: str = "", *args, **kwargs
):
    """Format slide attributes as text lines and add them in an annotation box.

    Known keys of `slide_attrs` are formatted into readable lines (case
    insensitive):

    - `size`: "width=... px, height=... px" (an int is used for both).
    - `mpp`: "mpp=...", followed by the magnification if `mag_0` is present.
    - `downsample`: "downsample=...", followed by "from <mag_0>x" if present.
    - `strategy`: "masking strategy=...".
    - `tissue_tolerance`: shown as a percentage.
    - `patch_size_target`: "ask: patch=...", with `overlap_target` and `mag_target`
      if present.
    - `patch_size_mask`: "out: patch=...", with `overlap_level` and `level` if
      present.

    `mag_0`, `mag_target`, `overlap_target`, `overlap_level` and `level` are only
    used as complements of the lines above. Any other key is written as
    "key=value" (underscores replaced by spaces).

    Parameters
    ----------
    img : np.ndarray
        Image of shape `(H, W, 3)`, dtype `uint8`, on which to add the box.

    slide_attrs : dict
        Slide information to write in the box.

    title : str, optional
        Title of the annotation box. Default is "".

    *args
        Additional positional arguments forwarded to `add_annotation_box`.

    **kwargs
        Additional keyword arguments forwarded to `add_annotation_box`
        (`text_box_height`, `text_box_color`, `alpha`...).

    Returns
    -------
    img : np.ndarray
        The annotated image.
    """

    # Creating text annotation list
    text = []
    for key, value in slide_attrs.items():
        if key.lower() == "size":
            if not isinstance(value, tuple):
                value = (value, value)
            text.append("width={} px, height={} px".format(*value))
        elif key.lower() == "mpp":
            text.append("mpp={:.4f}".format(value))
            if "mag_0" in slide_attrs.keys():
                text[-1] += ", {}x".format(slide_attrs["mag_0"])
        elif key.lower() == "downsample":
            text.append("downsample={}".format(value))
            if "mag_0" in slide_attrs.keys():
                text[-1] += " from {}x".format(slide_attrs["mag_0"])
        elif key.lower() == "strategy":
            text.append("masking strategy={}".format(value))
        elif key.lower() == "tissue_tolerance":
            text.append("tissue tolerance={:.1f}".format(value * 100) + "%")
        elif key.lower() == "patch_size_target":
            text.append("ask: patch={}".format(value))
            if "overlap_target" in slide_attrs.keys():
                text[-1] += " w. overlap={}".format(slide_attrs["overlap_target"])
            if "mag_target" in slide_attrs.keys():
                text[-1] += " at {}x".format(slide_attrs["mag_target"])
        elif key.lower() == "patch_size_mask":
            text.append("out: patch={}".format(value))
            if "overlap_level" in slide_attrs.keys():
                text[-1] += " w. overlap={}".format(slide_attrs["overlap_level"])
            if "level" in slide_attrs.keys():
                text[-1] += " at lvl={}".format(slide_attrs["level"])
        elif key.lower() not in [
            "mag_0",
            "mag_target",
            "overlap_target",
            "overlap_level",
            "level",
        ]:
            text.append("{}={}".format(key.replace("_", " "), value))

    # Adding annotations in a box on the image
    img = add_annotation_box(img=img, text=text, title=title, *args, **kwargs)

    # Returning the annotated image
    return img


def map_cell_feat(
    slide: str | OpenWSI | openslide.OpenSlide,
    data: pd.DataFrame,
    feat: str,
    analyse_level: int = 0,
    level_to_view: int = 0,
    discrete: bool = True,
    color: dict | list | None = None,
    cmap: plt.cm = plt.cm.jet,
    alpha: float = 0.6,
    limits: tuple[float, float] = (0, 1),
    ax: axes.Axes | None = None,
    title: str | None = None,
    loc: str = "right",
    show: bool = True,
):
    """Overlay a cell- or tile-level feature on the whole slide.

    The whole slide is read at `level_to_view` and each row of `data` is a cell drawn as a
    filled rectangle (bounding box), rescaled from `analyse_level` to `level_to_view`. 
    
    Discrete cell features get one color per unique value and a legend; continuous features are
    colored with `cmap` normalized to `limits` and get a colorbar.

    Parameters
    ----------
    slide : str | OpenWSI | openslide.OpenSlide
        Slide to display, or path to a slide opened with `get_slide_reader`.

    data : pd.DataFrame
        One row per cell / tile, with columns `x`, `y`, `w`, `h` (pixels at
        `analyse_level`) and `feat`.

    feat : str
        Name of the column of `data` to display.

    analyse_level : int, optional
        Pyramid level at which the coordinates in `data` were extracted. Default
        is 0.

    level_to_view : int, optional
        Pyramid level used for display. Clipped to the last level if not
        available. Default is 0.

    discrete : bool, optional
        Whether `feat` is discrete (one color per value) or continuous. Default is
        True.

    color : dict | list | None, optional
        Colors of the discrete values. Only a list is used (one color per non-NaN
        unique value); otherwise evenly spaced hues of `cmap` are used. Default is
        None.

    cmap : matplotlib.colors.Colormap, optional
        Colormap for continuous features, or for discrete ones when `color` is not
        a list. Default is `plt.cm.jet`.

    alpha : float, optional
        Opacity of the rectangles. Default is 0.6.

    limits : tuple[float, float], optional
        `(min, max)` of the color normalization for continuous features. Default
        is (0, 1).

    ax : matplotlib.axes.Axes, optional
        Axis to draw on. Default is None, meaning a new 10 x 10 inches figure.

    title : str, optional
        Axis title. Default is None.

    loc : str, optional
        Location of the legend (discrete) or colorbar (continuous). Default is
        "right".

    show : bool, optional
        Whether to call `plt.show()`. Default is True.

    Returns
    -------
    None
        The overlay is drawn on `ax`.

    Raises
    ------
    AssertionError
        If `color` is a list whose length differs from the number of values.
    """
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
    ax.set_title(title, size=20)
    ax.axis("off")

    if isinstance(slide, str):
        reader = get_slide_reader(slide)
        slide = reader(slide)

    if level_to_view >= slide.level_count:
        print(
            f"downsampling level={level_to_view} is not accessible, use level={slide.level_count-1} instead"
        )
        level_to_view = slide.level_count - 1

    dim_at_analyse_level = slide.level_dimensions[analyse_level]
    down_at_analyse_level = slide.level_downsamples[analyse_level]
    dim_at_level_to_view = slide.level_dimensions[level_to_view]
    down_at_level_to_view = slide.level_downsamples[level_to_view]

    wsi = slide.read_region(
        (0, 0), level_to_view, slide.level_dimensions[level_to_view]
    )
    if not isinstance(wsi, np.ndarray):
        wsi = np.array(wsi)[:, :, :3]

    ax.imshow(wsi, aspect="equal")
    if discrete:
        # color code the different labels
        values = data[feat].unique().tolist()
        values = [value for value in values if ~np.isnan(value)]
        if isinstance(color, list):
            colors = color
            assert not len(colors) != len(
                values
            ), "color list must match number of non nan values"
        else:
            hues = np.linspace(0, 1, len(values), endpoint=False)  # Evenly spaced hues
            colors = cmap(hues)
        # add cells
        for i, value in enumerate(values):
            for _, row in data.loc[data[feat] == value].iterrows():
                x, y = get_x_y_to(
                    (row["x"], row["y"]),
                    dim_at_analyse_level,
                    dim_at_level_to_view,
                    integer=True,
                )
                w, h = get_size_to(
                    (row["w"], row["h"]),
                    down_at_analyse_level,
                    down_at_level_to_view,
                    integer=True,
                )
                plot_seed = (x, y)
                patch = patches.Rectangle(
                    plot_seed,
                    w,
                    h,
                    fill=True,
                    facecolor=colors[i],
                    alpha=alpha,
                )
                ax.add_patch(patch)
        # add legend
        ax.legend(
            handles=[
                patches.Patch(color=colors[i], label=label)
                for i, label in enumerate(values)
            ],
            loc=loc,
            borderaxespad=1,
            frameon=True,
            facecolor="white",
            framealpha=0.7,
        )
    else:
        norm = plt.Normalize(limits[0], limits[1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        for _, row in data.iterrows():
            x, y = get_x_y_to(
                (row["x"], row["y"]),
                dim_at_analyse_level,
                dim_at_level_to_view,
                integer=True,
            )
            w, h = get_size_to(
                (row["w"], row["h"]),
                down_at_analyse_level,
                down_at_level_to_view,
                integer=True,
            )
            plot_seed = (x, y)
            value = row[feat]
            colors = cmap(norm(value))
            patch = patches.Rectangle(
                plot_seed,
                w,
                h,
                fill=True,
                facecolor=colors,
                alpha=alpha,
            )
            ax.add_patch(patch)
        plt.colorbar(sm, ax=ax, label=feat, location=loc)

    if show:
        plt.show()

def draw_cut(
    slide: str | OpenWSI | openslide.OpenSlide,
    xywh: list | np.ndarray,
    analyse_level: int = 0,
    level_to_view: int = 0,
    color: str = "red",
    title: str | None = None,
    ax: axes.Axes | None = None,
    show: bool = True,
):
    """Draw patch boxes on the whole slide with matplotlib.

    The whole slide is read at `level_to_view` and each patch is drawn as an
    unfilled rectangle, rescaled from `analyse_level` to `level_to_view`.

    Parameters
    ----------
    slide : str | OpenWSI | openslide.OpenSlide
        Slide to display, or path to a slide opened with `get_slide_reader`.

    xywh : list | np.ndarray
        Patch coordinates, one `(x, y, w, h)` per patch, in pixels at
        `analyse_level`.

    analyse_level : int, optional
        Pyramid level at which the coordinates were extracted. Default is 0.

    level_to_view : int, optional
        Pyramid level used for display. Clipped to the last level if not
        available. Default is 0.

    color : str, optional
        Edge color of the boxes. Default is "red".

    title : str, optional
        Axis title. Default is None.

    ax : matplotlib.axes.Axes, optional
        Axis to draw on. Default is None, meaning a new 10 x 10 inches figure.

    show : bool, optional
        Whether to call `plt.show()`. Default is True.

    Returns
    -------
    None
        The boxes are drawn on `ax`.
    """
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")

    if isinstance(slide, str):
        reader = get_slide_reader(slide)
        slide = reader(slide)

    if level_to_view >= slide.level_count:
        print(
            f"downsampling level={level_to_view} is not accessible, use level={slide.level_count-1} instead"
        )
        level_to_view = slide.level_count - 1

    dim_at_analyse_level = slide.level_dimensions[analyse_level]
    down_at_analyse_level = slide.level_downsamples[analyse_level]

    dim_at_level_to_view = slide.level_dimensions[level_to_view]
    down_at_level_to_view = slide.level_downsamples[level_to_view]

    wsi = slide.read_region(
        (0, 0), level_to_view, slide.level_dimensions[level_to_view]
    )
    if not isinstance(wsi, np.ndarray):
        wsi = np.array(wsi)[:, :, :3]

    ax.imshow(wsi, aspect="equal")
    for para in xywh:
        x, y, w, h = para
        x, y = get_x_y_to(
            (x, y), dim_at_analyse_level, dim_at_level_to_view, integer=True
        )
        w, h = get_size_to(
            (w, h), down_at_analyse_level, down_at_level_to_view, integer=True
        )
        plot_seed = (x, y)
        patch = patches.Rectangle(plot_seed, w, h, fill=False, edgecolor=color)
        ax.add_patch(patch)
    ax.set_title(title, size=20)
    ax.axis("off")

    if show:
        plt.show()


def visualize_cut(
    wsi_slide: str | OpenWSI | openslide.OpenSlide,
    coords: np.ndarray | list[list | tuple],
    size: tuple[int, int],
    patch_size_target: int,
    mag_target: int,
    slide_attrs: dict = None,
    save_cut: str = None,
    show_plot: bool = False,
    annotation: bool = True,
    *args,
    **kwargs,
) -> str:
    """Draw the patch grid on a slide thumbnail, annotate it and optionally save it.

    The best pyramid level for `mag_target` is found with
    `get_best_level_for_downsample`, the patch size at that level is computed, and
    each patch is drawn as a red square (OpenCV) on a thumbnail of the slide. An
    annotation box summarizing slide and patching attributes can be added.

    Parameters
    ----------
    wsi_slide : str | OpenWSI
        Slide, or path to a slide opened with `get_slide_reader`. Must expose
        `magnification`, `mpp`, `name` and `get_best_level_for_downsample`.

    coords : np.ndarray | list[list | tuple]
        Patch coordinates, shape `(n_tiles, 4)`, one `(x, y, w, h)` per patch;
        `(x, y)` are rescaled from the dimensions of the selected pyramid level to
        the thumbnail, `w` and `h` are ignored.

    size : tuple[int, int]
        Maximum `(width, height)` of the thumbnail, in pixels.

    patch_size_target : int
        Patch size, in pixels at `mag_target`.

    mag_target : int
        Target magnification of the patches (e.g. 20).

    slide_attrs : dict, optional
        Extra attributes to write in the annotation box; they override the default
        ones (`size`, `mpp`, `mag_0`, `patch_size_target`, `mag_target`,
        `patch_size_mask`). Default is None.

    save_cut : str, optional
        Directory where to save the image as `<wsi_slide.name>.jpg` (created if
        needed). Default is None (not saved).

    show_plot : bool, optional
        Whether to display the image with matplotlib. Default is False.

    annotation : bool, optional
        Whether to add the annotation box. Default is True.

    *args
        Additional positional arguments forwarded to `add_slide_annotation_box`.

    **kwargs
        Additional keyword arguments forwarded to `add_slide_annotation_box`.
        `text_box_height` defaults to 180 and `text_box_color` to
        (204, 139, 189).

    Returns
    -------
    cut_path : str | None
        Path of the saved image, or None if `save_cut` is None.
    """

    # Parsing default slide attrs
    if slide_attrs is None:
        slide_attrs = {}

    # Retrieve slide if not given but only path
    if isinstance(wsi_slide, str):
        reader_ = get_slide_reader(wsi_slide)
        wsi_slide = reader_(wsi_slide)
    else:
        wsi_slide = wsi_slide

    # Computing best level, patch size, overlap in original dimensions
    downsample = wsi_slide.magnification / mag_target
    level, downsample_level, resize_factor = wsi_slide.get_best_level_for_downsample(
        downsample
    )
    patch_size_level = round(patch_size_target / resize_factor)

    # Get thumbnail image of the slide and compute downsample factor and corresponding patch size
    thumbnail = np.array(wsi_slide.get_thumbnail(size))[:, :, :3]
    thumbnail_height, thumbnail_width, _ = thumbnail.shape
    downsample_factor = max(
        wsi_slide.level_dimensions[level][0] / thumbnail_width,
        wsi_slide.level_dimensions[level][1] / thumbnail_height,
    )
    thumbnail_patch_size = max(1, int(patch_size_level / downsample_factor))

    # Draw rectangles for patches
    for x_, y_, _, _ in coords:
        x_, y_ = get_x_y_to(
            point=(x_, y_),
            dim_from=wsi_slide.level_dimensions[level],
            dim_to=(thumbnail_width, thumbnail_height),
            integer=True,
        )
        thickness = max(1, thumbnail_patch_size // 16)
        thumbnail = cv2.rectangle(
            img=thumbnail,
            pt1=(x_, y_),
            pt2=(x_ + thumbnail_patch_size, y_ + thumbnail_patch_size),
            color=(255, 0, 0),
            thickness=thickness,
        )

    # Adding annotations
    all_slide_attrs = {
        "size": wsi_slide.dimensions,
        "mpp": wsi_slide.mpp,
        "mag_0": wsi_slide.magnification,
        "patch_size_target": patch_size_target,
        "mag_target": mag_target,
        "patch_size_mask": patch_size_level,
    }
    all_slide_attrs.update(slide_attrs)
    if "text_box_height" not in kwargs.keys():
        kwargs["text_box_height"] = 180
    if "text_box_color" not in kwargs.keys():
        kwargs["text_box_color"] = (204, 139, 189)
    if annotation:
        thumbnail = add_slide_annotation_box(
            thumbnail,
            slide_attrs=all_slide_attrs,
            title=f"{len(coords)} patches",
            *args,
            **kwargs,
        )

    # Showing the figure
    if show_plot:
        _, axis = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
        axis.set_axis_off()
        axis.imshow(thumbnail, aspect="equal")
        plt.show()

    # Saving the visualization
    if save_cut is not None:
        os.makedirs(save_cut, exist_ok=True)
        cut_path = os.path.join(save_cut, f"{wsi_slide.name}.jpg")
        Image.fromarray(thumbnail).save(cut_path)
    else:
        cut_path = None

    # Returning path to saved image
    return cut_path

def visualise_tile_feat(
    slide: str | OpenWSI | openslide.OpenSlide,
    data: pd.DataFrame,
    feat: str,
    region=(0, 0),
    size=None,
    analyse_level: int = 0,
    level_to_view: int = 0,
    cmap: plt.cm = plt.cm.jet,
    alpha: float = 0.6,
    smooth: int | None = None,
    limits: tuple[float, float] | None = None,
    ax: axes.Axes | None = None,
    legend: list | None = None,
    title: str | None = None,
    loc: str = "right",
    show: bool = True,
):
    """Overlay a continuous tile-level feature on a region of the slide.

    The region is read at `level_to_view`. Tiles of `data` intersecting the region
    are rasterized (vectorized rescaling) into an RGBA overlay colored with `cmap`,
    optionally smoothed with a Gaussian filter, and displayed on top of the slide.

    Parameters
    ----------
    slide : str | OpenWSI | openslide.OpenSlide
        Slide to display, or path to a slide opened with `get_slide_reader`.

    data : pd.DataFrame
        One row per tile, with columns `x`, `y`, `w`, `h` (pixels at
        `analyse_level`) and `feat`.

    feat : str
        Name of the column of `data` to display.

    region : tuple[int, int], optional
        Top-left corner `(x, y)` of the region to display. Used both to filter /
        shift the tiles (frame of `analyse_level`) and as `location` of
        `slide.read_region`. Default is (0, 0).

    size : tuple[int, int], optional
        Size `(width, height)` of the region at `analyse_level`. Default is None,
        meaning the full slide dimensions at `analyse_level`.

    analyse_level : int, optional
        Pyramid level at which the coordinates in `data` were extracted. Default
        is 0.

    level_to_view : int, optional
        Pyramid level used for display. Clipped to the last level if not
        available. Default is 0.

    cmap : matplotlib.colors.Colormap, optional
        Colormap of the feature. Default is `plt.cm.jet`.

    alpha : float, optional
        Opacity of the overlay. Default is 0.6.

    smooth : int, optional
        Sigma (in display pixels) of the Gaussian filter applied to the overlay.
        Default is None (no smoothing).

    limits : tuple[float, float], optional
        `(min, max)` of the color normalization. Default is None, meaning the min
        and max of `data[feat]`.

    ax : matplotlib.axes.Axes, optional
        Axis to draw on. Default is None, meaning a new 10 x 10 inches figure.

    legend : list, optional
        Unused. Default is None.

    title : str, optional
        Axis title. Default is None.

    loc : str, optional
        Location of the colorbar; no colorbar if empty or None. Default is
        "right".

    show : bool, optional
        Whether to call `plt.show()`. Default is True.

    Returns
    -------
    None
        The overlay is drawn on `ax`.
    """
    # here numpy can be false and I get the dim with (pyvips.image.get("width"), pyvips.image.get("height"))
    # should also allow pyvips images as slide
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
    ax.set_title(title, size=20)
    ax.axis("off")

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
    wsi = slide.read_region(location=region, level=level_to_view, size=dim_view)
    if not isinstance(wsi, np.ndarray):
        wsi = np.array(wsi)[:, :, :3]

    # normalize features
    values = data[feat].values
    if limits is not None:
        vmin, vmax = limits
    else:
        vmin, vmax = values.min(), values.max()
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Vectorized coordinates and sizes
    xs = data["x"].values
    ys = data["y"].values
    ws = data["w"].values
    hs = data["h"].values

    # Filter tiles that intersect the selected region
    x0, y0 = region
    W, H = size

    mask = (xs + ws >= x0) & (xs < x0 + W) & (ys + hs >= y0) & (ys < y0 + H)

    xs, ys, ws, hs, values = xs[mask], ys[mask], ws[mask], hs[mask], values[mask]

    # Shift to region coordinates
    xs = xs - x0
    ys = ys - y0

    # or apply get_size_to() get_x_y_to()
    x_scaled = (xs * dim_view[0] / size[0]).astype(int)
    y_scaled = (ys * dim_view[1] / size[1]).astype(int)
    w_scaled = np.maximum((ws * down_analyse / down_view).astype(int), 1)
    h_scaled = np.maximum((hs * down_analyse / down_view).astype(int), 1)

    # Create transparent overlay
    overlay = np.zeros((dim_view[1], dim_view[0], 4), dtype=np.float32)  # HWC RGBA
    for x, y, w, h, val in zip(x_scaled, y_scaled, w_scaled, h_scaled, values):
        color = cmap(norm(val), alpha=alpha)  # RGBA in [0,1]
        overlay[y : y + h + 1, x : x + w + 1, :] = color

    # gaussian smoothing
    if smooth is not None:
        overlay = gaussian_filter(overlay, sigma=(smooth, smooth, 0))

    ax.imshow(wsi, aspect="equal")
    ax.imshow(overlay, aspect="equal")
    ax.set_title(title, size=16)
    ax.axis("off")

    if loc:
        plt.colorbar(sm, ax=ax, label=feat, location=loc)
    if show:
        plt.show()


def make_feature_thumbnail(
    slide: str | OpenWSI | openslide.OpenSlide,
    data: pd.DataFrame,
    feat: str,
    region: tuple[int, int] = (0, 0),
    size: tuple[int, int] | None = None,
    analyse_level: int = 0,
    level_to_view: int = 0,
    normalize: str | None = None,
    smooth: int | None = None,
    rgb: bool = False,
    numpy: bool = True,
    feat_only: bool = True,
):
    """Rasterize a tile-level feature into a gray-scale thumbnail of a region.

    Tiles of `data` intersecting the region are rescaled from `analyse_level` to
    `level_to_view` and painted with their (optionally normalized) feature value
    into a single-channel `float32` image, optionally smoothed with a Gaussian
    filter.

    Parameters
    ----------
    slide : str | OpenWSI | openslide.OpenSlide
        Slide to read, or path to a slide opened with `get_slide_reader`.

    data : pd.DataFrame
        One row per tile, with columns `x`, `y`, `w`, `h` (pixels at
        `analyse_level`) and `feat`.

    feat : str
        Name of the column of `data` to rasterize.

    region : tuple[int, int], optional
        Top-left corner `(x, y)` of the region. Used both to filter / shift the
        tiles (frame of `analyse_level`) and as `location` of `slide.read_region`.
        Default is (0, 0).

    size : tuple[int, int], optional
        Size `(width, height)` of the region at `analyse_level`. Default is None,
        meaning the full slide dimensions at `analyse_level`.

    analyse_level : int, optional
        Pyramid level at which the coordinates in `data` were extracted. Default
        is 0.

    level_to_view : int, optional
        Pyramid level of the output thumbnail. Clipped to the last level if not
        available. Default is 0.

    normalize : str, optional
        Feature normalization: "minmax" (`(f - min) / (max - min)`), "max"
        (`f / max`) or None (raw values). Default is None.

    smooth : int, optional
        Sigma (in output pixels) of the Gaussian filter. Default is None (no
        smoothing).

    rgb : bool, optional
        Unused. Default is False.

    numpy : bool, optional
        If True, return NumPy arrays; otherwise the feature map is converted to a
        `PIL.Image.Image` (mode "L") and the slide region is read as PIL. Passed to
        `slide.read_region`. Default is True.

    feat_only : bool, optional
        If True, return only the feature map; otherwise also return the slide
        region. Default is True.

    Returns
    -------
    overlay : np.ndarray | PIL.Image.Image
        Feature map of shape `(H, W)` at `level_to_view`, `float32` when NumPy.
    wsi : np.ndarray | PIL.Image.Image
        Slide region read at `level_to_view`. Only returned when `feat_only` is
        False.
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

    # normalize features
    feats = data[feat].values
    if normalize == "minmax":
        feats_norm = (feats - feats.min()) / (feats.max() - feats.min())
    elif normalize == "max":
        feats_norm = feats / feats.max()
    else:
        feats_norm = feats
    # Vectorized coordinates and sizes
    xs = data["x"].values
    ys = data["y"].values
    ws = data["w"].values
    hs = data["h"].values

    # Filter tiles that intersect the selected region
    x0, y0 = region
    W, H = size

    mask = (xs + ws >= x0) & (xs < x0 + W) & (ys + hs >= y0) & (ys < y0 + H)

    xs, ys, ws, hs, feats_norm = (
        xs[mask],
        ys[mask],
        ws[mask],
        hs[mask],
        feats_norm[mask],
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
    overlay = np.zeros((dim_view[1], dim_view[0]), dtype=np.float32)  # gray scale
    for x, y, w, h, val in zip(x_scaled, y_scaled, w_scaled, h_scaled, feats_norm):
        overlay[y : y + h + 1, x : x + w + 1] = val
    # gaussian smoothing
    if smooth is not None:
        overlay = gaussian_filter(overlay, sigma=(smooth, smooth))

    if not numpy:
        overlay = Image.fromarray(overlay, mode="L")

    if feat_only:
        return overlay
    else:
        return overlay, wsi


def visualise_tile_rgb(
    slide: str | OpenWSI | openslide.OpenSlide,
    data: pd.DataFrame,
    region=(0, 0),
    size=None,
    analyse_level: int = 0,
    level_to_view: int = 0,
    alpha: float = 0.6,
    smooth: int | None = None,
    ax: axes.Axes | None = None,
    title: str | None = None,
    show: bool = True,
):
    """Overlay per-tile RGB colors on a region of the slide.

    Typically used to display a 3-component embedding (e.g. PCA of tile features)
    mapped to RGB. Tiles of `data` intersecting the region are rasterized into an
    RGBA overlay with constant opacity `alpha`, optionally smoothed, and displayed
    on top of the slide region read at `level_to_view`.

    Parameters
    ----------
    slide : str | OpenWSI | openslide.OpenSlide
        Slide to display, or path to a slide opened with `get_slide_reader`.

    data : pd.DataFrame
        One row per tile, with columns `x`, `y`, `w`, `h` (pixels at
        `analyse_level`) and `r`, `g`, `b` (values in [0, 1]).

    region : tuple[int, int], optional
        Top-left corner `(x, y)` of the region to display. Used both to filter /
        shift the tiles (frame of `analyse_level`) and as `location` of
        `slide.read_region`. Default is (0, 0).

    size : tuple[int, int], optional
        Size `(width, height)` of the region at `analyse_level`. Default is None,
        meaning the full slide dimensions at `analyse_level`.

    analyse_level : int, optional
        Pyramid level at which the coordinates in `data` were extracted. Default
        is 0.

    level_to_view : int, optional
        Pyramid level used for display. Clipped to the last level if not
        available. Default is 0.

    alpha : float, optional
        Opacity of the overlay. Default is 0.6.

    smooth : int, optional
        Sigma (in display pixels) of the Gaussian filter applied to the overlay.
        Default is None (no smoothing).

    ax : matplotlib.axes.Axes, optional
        Axis to draw on. Default is None, meaning a new 10 x 10 inches figure.

    title : str, optional
        Axis title. Default is None.

    show : bool, optional
        Whether to call `plt.show()`. Default is True.

    Returns
    -------
    None
        The overlay is drawn on `ax`.

    Raises
    ------
    AssertionError
        If `data` lacks one of the `r`, `g`, `b` columns.
    """
    # get rgb image
    assert all([k in data.keys() for k in ["r", "g", "b"]]), print(
        "data must containt r, g, b features"
    )
    rgbs = data[["r", "g", "b"]].to_numpy()

    # here numpy can be false and I get the dim with (pyvips.image.get("width"), pyvips.image.get("height"))
    # should also allow pyvips images as slide
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
    ax.set_title(title, size=20)
    ax.axis("off")

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
    wsi = slide.read_region(location=region, level=level_to_view, size=dim_view)
    if not isinstance(wsi, np.ndarray):
        wsi = np.array(wsi)[:, :, :3]

    # Vectorized coordinates and sizes
    xs = data["x"].values
    ys = data["y"].values
    ws = data["w"].values
    hs = data["h"].values

    # Filter tiles that intersect the selected region
    x0, y0 = region
    W, H = size

    mask = (xs + ws >= x0) & (xs < x0 + W) & (ys + hs >= y0) & (ys < y0 + H)

    xs, ys, ws, hs, values = xs[mask], ys[mask], ws[mask], hs[mask], rgbs[mask]

    # Shift to region coordinates
    xs = xs - x0
    ys = ys - y0

    # or apply get_size_to() get_x_y_to()
    x_scaled = (xs * dim_view[0] / size[0]).astype(int)
    y_scaled = (ys * dim_view[1] / size[1]).astype(int)
    w_scaled = np.maximum((ws * down_analyse / down_view).astype(int), 1)
    h_scaled = np.maximum((hs * down_analyse / down_view).astype(int), 1)

    # Create transparent overlay
    overlay = np.zeros((dim_view[1], dim_view[0], 4), dtype=values.dtype)  # HWC RGBA
    for x, y, w, h, rgb in zip(x_scaled, y_scaled, w_scaled, h_scaled, values):
        rgba = np.concatenate([rgb, np.ones(1) * alpha], axis=0)
        overlay[y : y + h + 1, x : x + w + 1, :] = rgba

    # gaussian smoothing
    if smooth is not None:
        overlay = gaussian_filter(overlay, sigma=(smooth, smooth, 0))

    ax.imshow(wsi, aspect="equal")
    ax.imshow(overlay, aspect="equal")
    ax.set_title(title, size=16)
    ax.axis("off")

    if show:
        plt.show()



def visualize_tissue_seg(
    mask: np.ndarray,
    size: tuple[int, int],
    slide_name: str,
    slide_attrs: dict = None,
    annotation: bool = True,
    save_seg: str = None,
    show: bool = False,
    *args,
    **kwargs,
) -> str:
    """Resize a tissue mask to a thumbnail, annotate it and optionally save it.

    The mask is converted to 3 channels and resized (linear interpolation) so that
    its longest side matches the corresponding entry of `size`, keeping the aspect
    ratio.

    Parameters
    ----------
    mask : np.ndarray
        Tissue segmentation mask of shape `(H, W)`, accepted by `cv2.cvtColor`
        (typically `uint8` with values 0 / 255).

    size : tuple[int, int]
        Target `(width, height)` of the thumbnail, in pixels.

    slide_name : str
        Name of the slide, used as file name of the saved image.

    slide_attrs : dict, optional
        Slide attributes written in the annotation box (see
        `add_slide_annotation_box`). Default is None.

    annotation : bool, optional
        Whether to add the annotation box titled "Tissue mask". Default is True.

    save_seg : str, optional
        Directory where to save the image as `<slide_name>.jpg` (created if
        needed). Default is None (not saved).

    show : bool, optional
        Whether to display the image with matplotlib. Default is False.

    *args
        Additional positional arguments forwarded to `add_slide_annotation_box`.

    **kwargs
        Additional keyword arguments forwarded to `add_slide_annotation_box`.

    Returns
    -------
    seg_path : str | None
        Path of the saved image, or None if `save_seg` is None.
    """

    # Computing thumbnail size
    mask_height, mask_width = mask.shape
    if mask_width > mask_height:
        thumbnail_width = size[0]
        thumbnail_height = int(size[1] * mask_height / mask_width)
    else:
        thumbnail_height = size[1]
        thumbnail_width = int(size[0] * mask_width / mask_height)
    thumbnail_dim = (thumbnail_width, thumbnail_height)

    # Converting to grayscale and resizing mask
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = cv2.resize(mask, thumbnail_dim, interpolation=cv2.INTER_LINEAR)

    # Adding annotations
    if annotation:
        mask = add_slide_annotation_box(
            mask, slide_attrs=slide_attrs, title="Tissue mask", *args, **kwargs
        )

    # Showing the tissue segmentation image
    if show:
        _, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
        ax.set_axis_off()
        ax.imshow(mask, aspect="equal")
        plt.show()

    # Saving the tissue segmentation mask visualization
    if save_seg is not None:
        os.makedirs(save_seg, exist_ok=True)
        seg_path = os.path.join(save_seg, f"{slide_name}.jpg")
        Image.fromarray(mask).save(seg_path)
    else:
        seg_path = None

    # Returning path to the saved image
    return seg_path



def multiple_openslide_thumbnail(
    data_dir: str,
    outputs_dir: str = None,
    level: int = 0,
    extension: str = "ndpi",
    grayscale: bool = False,
    verbose: bool = True,
):
    """Save a full-slide thumbnail of every slide found in a directory.

    Each slide is read whole at `level` with `get_slide_whole`, plotted with
    matplotlib (titled with the file stem and level) and saved as
    `<stem>_slide_level_<level>.png`.

    Parameters
    ----------
    data_dir : str
        Directory containing the slides (not searched recursively).

    outputs_dir : str, optional
        Output directory. If not None, it is replaced by
        `<data_dir>/visualization` (created if needed). Default is None.

    level : int, optional
        Pyramid level to read. Default is 0.

    extension : str, optional
        Extension to look for; a file is kept when `extension` is a substring of
        its suffix. Default is "ndpi".

    grayscale : bool, optional
        If True, convert to inverted gray scale (`1 - rgb2gray(wsi)`). Default is
        False.
        
    verbose : bool, optional
        Whether to print the number of slides found and save messages. Default is
        True.

    Returns
    -------
    None
        Thumbnails are saved to disk.
    """
    # Finding files corresponding to the format in data dir
    files = sorted(
        str(path)
        for path in Path(data_dir).iterdir()
        if path.is_file() and extension in path.suffix
    )
    cnt = len(files)
    if verbose:
        print(f"{cnt} WSI.{extension} were found in source dir: {data_dir}")

    # Make outputs directory if it does not exist
    if outputs_dir is not None:
        outputs_dir = os.path.join(data_dir, "visualization")
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir, exist_ok=True)

    # Make thumbnail visualization for all files found in data dir
    # call make_openslide_thumbnail
    for file in files:

        # Retrieve filename and file taf
        filename, _ = os.path.splitext(file)
        tag = os.path.basename(filename)

        # Get corresponding WSI and convert to grayscale if necessary
        wsi_np = get_slide_whole(file, level=level)
        if grayscale:
            wsi_np = (rgb2gray(wsi_np).astype("float32") - 1) * -1

        # Creating thumbnail visualization and saving it in outputs dir
        wsi_aspect = wsi_np.shape[1] / wsi_np.shape[0]
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(12, 12 / wsi_aspect), layout="constrained"
        )
        ax.set_title(f"{tag} at level={level}")
        ax.imshow(wsi_np)
        ax.axis("off")
        fig.savefig(os.path.join(outputs_dir, f"{tag}_slide_level_{level}.png"))
        plt.close()
        if verbose:
            print(f"Slide thumbnail saved in {outputs_dir}.")


def dynamic_display_range(image, smoothing_sigma=1.0, min_density_ratio=0.001):
    """
    Compute the dynamic display range for an image channel using histogram analysis.

    Parameters:
        image (ndarray): Input image channel as a NumPy array.
        smoothing_sigma (float): Sigma for Gaussian smoothing of the histogram.
        min_density_ratio (float): Minimum histogram density (relative to peak) to include in the range.

    Returns:
        tuple: (lower_bound, upper_bound) for the display range.
    """
    # Compute the histogram
    hist, bin_edges = np.histogram(
        image, bins=256, range=(np.min(image), np.max(image)), density=True
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Smooth the histogram to reduce noise
    smoothed_hist = gaussian_filter1d(hist, sigma=smoothing_sigma)
    # Determine the peak density
    peak_density = np.max(smoothed_hist)

    # Find the bounds where the histogram density exceeds a threshold
    threshold = peak_density * min_density_ratio
    valid_bins = bin_centers[smoothed_hist > threshold]

    # Set the lower and upper bounds
    lower_bound = valid_bins[0] if len(valid_bins) > 0 else np.min(image)
    upper_bound = valid_bins[-1] if len(valid_bins) > 0 else np.max(image)

    return lower_bound, upper_bound


def percentil_range(image, p_low=1, p_high=99):
    lo = np.percentile(image, p_low)
    hi = np.percentile(image, p_high)
    return lo, hi


def blend_colors(
    img,
    colors,
    scale_by="hist",
    density_ratio=0.0001,
    gammas: float | list[float] = 1,
    alpha: float | list[float] = 1,
    blending="add",
    eps=1e-8,
):

    if len(colors.shape) > 1:
        n_channel_color = colors.shape[1]
    else:
        n_channel_color = len(colors)
        colors = np.expand_dims(colors, axis=0)

    if img.ndim > 2:
        r, c, nc = img.shape[:3]
    else:
        nc = 1
        r, c = img.shape[:2]
        img = np.expand_dims(img, axis=-1)

    relative_imgs = []
    for i in range(nc):
        channel = img[..., i]
        # relative image is how bright the channel will be
        if scale_by == "hist":
            lower_bound, upper_bound = dynamic_display_range(
                channel, min_density_ratio=density_ratio
            )
            channel = np.clip(channel, a_min=lower_bound, a_max=upper_bound)
        elif scale_by == "percentile":
            lower_bound, upper_bound = percentil_range(channel)
            channel = np.clip(channel, a_min=lower_bound, a_max=upper_bound)
        relative_img = (channel - channel.min()) / (channel.max() - channel.min() + eps)

        if isinstance(gammas, list):
            assert (
                len(gammas) == nc
            ), "if you want per channel gammas, provide a list that matches"
            relative_img = np.power(relative_img, gammas[i])
        else:
            relative_img = np.power(relative_img, gammas)
        relative_imgs.append(relative_img)
    relative_imgs = np.stack(relative_imgs, axis=-1)

    # blending (switch between method)
    blended_img = np.zeros((r, c, n_channel_color))
    if blending == "add":
        # Additive mixing
        for i in range(nc):
            for j in range(n_channel_color):
                channel_color = colors[i, j]
                blended_img[..., j] += channel_color * relative_imgs[..., i]
        blended_img = np.clip(blended_img, a_min=0, a_max=255)
        blended_img = blended_img.astype(np.uint8)

    elif blending == "max":
        argmax = np.argmax(relative_imgs, axis=-1)
        blended_img = np.zeros((r, c, n_channel_color))
        for i in range(nc):
            mask = argmax == i
            for j in range(n_channel_color):
                channel_color = colors[i, j]
                blended_img[..., j][mask] = channel_color * relative_imgs[..., i][mask]
        blended_img = blended_img.astype(np.uint8)

    # Add Max projection style

    return blended_img