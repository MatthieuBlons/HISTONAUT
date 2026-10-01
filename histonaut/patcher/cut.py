"""Cut a WSI into square patches.

`SlidePatcher` computes a tissue mask on a downsampled pyramid level, lays a
regular patch grid at the pyramid level closest to the target magnification,
keeps the patches lying on enought tissue, optionally filters patches (texture / colour)
and saves their coordinates as `xywh` rows (pixels at the pyramid level
`SlidePatcher.level`, see `SlidePatcher` Notes).
"""

# General libraries
import os
from functools import partial

# Data libraries
import numpy as np
import PIL
import cv2
from torch.utils.data import Dataset

# Tensor op

# ML libraries

# Plotting libraries
import matplotlib.pyplot as plt

# Project modules
from histonaut.slide.utils import (
    get_size_to,
    get_x_y_to,
)
from histonaut.slide.processing import (
    mask_percentage,
    compute_otsu_mask,
    compute_luminosity_mask,
)
from histonaut.slide.torch_processing import (
    hog_selection_torch,
    color_filter_selection_torch,
)
from histonaut.slide.draw import (
    mosaic,
    visualize_cut,
    visualize_tissue_seg,
)
from histonaut.slide.reader import OpenWSI
from histonaut.patcher.utils import grid_blob
from histonaut.patcher.io import save_h5


class SlidePatcher:
    """Extract tissue patches from a whole-slide image.

    On construction, the patcher resolves the pyramid level closest to
    `mag_target`, picks the right level for the tissue mask, builds the tissue mask
    and (optional) a tile selection functions.
    
    If `lazy` (default) is True, immediately computes the valid patch coordinates 
    (and saves them when `dst` is given).
    
    The instance is then indexable and iterable over the valid patches.

    Parameters
    ----------
    slide : OpenWSI
        Slide to patch.

    pixel_size_0 : int, optional
        Pixel size at level 0, in microns per pixel. Default is None, meaning
        `slide.mpp` is used.

    pixel_size_target : int, optional
        Pixel size at the target magnification, in microns per pixel. Default
        is None, meaning `pixel_size_0 * mag_0 / mag_target`.

    mag_0 : int, optional
        Magnification at level 0. Default is None, meaning `slide.magnification`
        is used.

    mag_target : int, optional
        Target magnification of the patches. Default is 20.

    patch_size : int, optional
        Side of the returned patches, in pixels at the target magnification.
        Default is 256.

    overlap : int, optional
        Overlap between neighbouring patches, in pixels at the target
        magnification. Default is 0.

    mask_downsample : int, optional
        Requested downsample factor (relative to level 0) giving the level on which
        the tissue mask is computed. Default is 16. If None, the mask is computed
        at the patching level `level`.

    margin : int | None, optional
        Border, in pixels of the mask level, removed on each side of the mask. 
        Default is None, meaning 1% of the mask height and width.

    mask_tolerance : int, optional
        Tissue fraction threshold passed to `mask_percentage` to accept a patch.
        Default is None, meaning 0.

    mask_strategy : str, optional
        Tissue masking strategy. Default is ``"otsu"``.

    custom_mask : np.ndarray, optional
        Precomputed binary tissue mask used instead of `mask_strategy`. Default
        is None.

    custom_xywh : np.ndarray, optional
        Precomputed `(N, 4)` array of `xywh` coordinates; when given, masking and
        selection are skipped. Default is None.

    overwrite : bool, optional
        Whether to save the coordinates to `dst` during lazy patching. 
        Default is True.

    selection_strategy : str, optional
        Tile selection strategy, a key of `tile_selection_mapping` (``"hog"``).
        Default is None, meaning no selection.

    color_thresh : tuple[int, int, int], optional
        Colour threshold per channel (R, G, B) for colour filtering. Default is
        None, meaning no colour filtering.

    stat_fct : str, optional
        Statistic compared to `color_thresh`. Default is ``"channel_std"``.

    q : float, optional
        Quantile used by quantile-based colour statistics. Default is 0.2.

    remove_background : bool, optional
        Whether to remove background pixels before computing the colour
        statistic. Default is True.

    background_thresh : int | tuple[int, int, int], optional
        Colour threshold (per channel or shared) above which a pixel is
        considered background. Default is 245.

    background_average : bool, optional
        Whether background is detected on the channel-averaged pixels rather
        than per channel. Default is True.

    local_average : bool, optional
        Whether to apply local averaging before computing the statistic. Default
        is True.

    filter_size : tuple[int, int] | int, optional
        Window size, in pixels, of the local averaging. Default is 16.

    filter_all : bool, optional
        Whether the colour criterion must be met on all channels (True) or on at
        least one (False). Default is False.

    aggregate : str, optional
        Function aggregating the per-pixel statistic over the patch. Default is
        ``"mean"``.

    xywh_only : bool, optional
        Whether indexing/iteration returns only coordinates (True) or
        `(tile, coords)` tuples (False). Default is True.

    pil : bool, optional
        Whether tiles are returned as `PIL.Image.Image` (True) or RGB
        `np.ndarray` (False). Default is False.

    dst : str, optional
        Output directory; coordinates are written to
        ``<dst>/patches/<slide.name>_patches.<save_as>``. Default is None,
        meaning nothing is saved.

    save_as : str, optional
        Output format. Only ``"h5"`` is supported. Default is ``"h5"``.

    lazy : bool, optional
        Whether to compute (and save) the patch coordinates in the constructor.
        Default is True.

    *args
        Extra positional arguments bound (via `functools.partial`) to both the
        mask function and the tile selection function.

    **kwargs
        Extra keyword arguments bound to both the mask function and the tile
        selection function.

    Attributes
    ----------
    mask_strategy_mapping : dict[str, callable]
        Class-level registry: mask strategy name -> tissue mask function.
    tile_selection_mapping : dict[str, callable]
        Class-level registry: selection strategy name -> tile selection function.
    slide : OpenWSI
        The slide being patched.
    width, height : int
        Slide dimensions at level 0, in pixels.
    mag_0, pixel_size_0 : float
        Magnification and pixel size (microns per pixel) at level 0.
    mag_target, pixel_size_target : float
        Target magnification and pixel size (microns per pixel).
    level : int
        Pyramid level read to build the patches.
    downsample_level : float
        Downsample factor of `level` relative to level 0.
    resize_factor : float
        Scale between `level` and the target magnification, as returned by
        `slide.get_best_level_for_downsample` (1 for an exact level match).
    patch_size_target, overlap_target : int
        Patch size and overlap, in pixels at the target magnification.
    patch_size_level, overlap_level : int
        Patch size and overlap, in pixels at `level`.
    level_mask : int
        Pyramid level on which the tissue mask is computed.
    mask_downsample : float
        Downsample factor assumed for the mask level.
    margin : tuple[int, int] | None
        Mask border removed on each axis, in pixels at `level_mask`.
    mask_selection : functools.partial
        Tissue mask function resolved from `mask_strategy`.
    tile_selection : functools.partial | None
        Tile selection function resolved from `selection_strategy`, or None.
    valid_patches : list[list[int]] | np.ndarray | None
        Valid patch coordinates, `(N, 4)` `xywh` rows; None until patching.
    nb_valid_patches : int | None
        Number of valid patches; None until patching.
    idx_selected : np.ndarray | None
        Indices, among tissue patches, kept by selection / colour filtering.
    patch_path : str | None
        Path of the saved coordinate file, if any.
    i : int
        Iteration counter used by `__next__`.

    Notes
    -----
    Coordinates are built on the grid of pyramid level `level` and stored as
    `[x, y, w, h]` with `w = h = patch_size_level`, in pixels of that level
    (they coincide with level-0 pixels when `level == 0`). `get_tile` reads
    them back with `slide.read_region(..., level=level)` and resizes each tile
    to `patch_size_target`. The h5 file stores the `level` attribute needed to
    interpret them.

    Examples
    --------
    >>> from histonaut.slide.reader import OpenWSI
    >>> from histonaut.patcher.cut import SlidePatcher
    >>> slide = OpenWSI("slide.svs")
    >>> patcher = SlidePatcher(
    ...     slide,
    ...     mag_target=20,
    ...     patch_size=256,
    ...     mask_strategy="otsu",
    ...     xywh_only=False,
    ...     dst="outputs",
    ... )
    >>> len(patcher)  # number of valid patches
    >>> patcher.patch_path  # 'outputs/patches/<slide name>_patches.h5'
    >>> for tile, (x, y, w, h) in patcher:
    ...     pass  # tile: (256, 256, 3) np.ndarray
    """

    # Creating a dictionary to store the tissue mask selection strategies mapping with the corresponding function
    mask_strategy_mapping = {
        "luminosity": compute_luminosity_mask,
        "otsu": compute_otsu_mask,
    }

    # Creating a dictionary to store the tile selection strategies mapping with the corresponding function
    tile_selection_mapping = {"hog": hog_selection_torch}

    def __init__(
        self,
        slide: OpenWSI,
        pixel_size_0: int = None,
        pixel_size_target: int = None,
        mag_0: int = None,
        mag_target: int = 20,
        patch_size: int = 256,
        overlap: int = 0,
        mask_downsample: int = 16,
        margin: int | None = None,
        mask_tolerance: int = None,
        mask_strategy: str = "otsu",
        custom_mask: np.ndarray = None,
        custom_xywh: np.ndarray = None, # Could also be a str -> path to coords h5 file
        overwrite: bool = True,
        selection_strategy: str = None,
        color_thresh: tuple[int, int, int] = None,
        stat_fct: str = "channel_std",
        q: float = 0.2,
        remove_background: bool = True,
        background_thresh: int | tuple[int, int, int] = 245,
        background_average: bool = True,
        local_average: bool = True,
        filter_size: tuple[int, int] | int = 16,
        filter_all: bool = False,
        aggregate: str = "mean",
        xywh_only: bool = True,
        pil: bool = False,
        dst: str = None,
        save_as: str = "h5",
        lazy: bool = True,
        *args,
        **kwargs,
    ):
        """Initialise the patcher; see the class docstring for the parameters."""

        # Storing useful attributes
        self.slide = slide
        self.width, self.height = slide.dimensions
        self.patch_size_target = patch_size
        self.mag_target = mag_target
        self.overlap_target = overlap
        self.custom_xywh = custom_xywh
        self.overwrite = overwrite
        self.xywh_only = xywh_only
        self.pil = pil
        self.dst = dst
        self.save_as = save_as
        self.selection_strategy = selection_strategy
        self.patch_path = None
        self.valid_patches = None
        self.nb_valid_patches = None
        self.idx_selected = None

        # Storing color filtering args
        self.color_thresh = color_thresh
        self.stat_fct = stat_fct
        self.q = q
        self.remove_background = remove_background
        self.background_thresh = background_thresh
        self.background_average = background_average
        self.local_average = local_average
        self.filter_size = filter_size
        self.filter_all = filter_all
        self.aggregate = aggregate

        # Parsing source image default args using the slide attributes
        self.mag_0 = mag_0 if mag_0 is not None else slide.magnification
        self.pixel_size_0 = pixel_size_0 if pixel_size_0 is not None else slide.mpp

        # Parsing tolerance for tissue mask default args
        self.mask_tolerance = mask_tolerance if mask_tolerance is not None else 0

        # Computing target level's associated values
        downsample = self.mag_0 / self.mag_target
        self.pixel_size_target = pixel_size_target
        if pixel_size_target is None:
            self.pixel_size_target = self.pixel_size_0 * downsample
        self.level, self.downsample_level, self.resize_factor = (
            self.slide.get_best_level_for_downsample(downsample)
        )

        # Parsing mask downsample and get associated level
        if mask_downsample is None:
            self.mask_downsample = 1.0
            self.level_mask = self.level
        else:
            self.mask_downsample = mask_downsample
            self.level_mask, _, _ = self.slide.get_best_level_for_downsample(
                self.mask_downsample
            )
        self.margin = (margin, margin) if margin is not None else None
        self.mask_strategy = mask_strategy
        self.mask_selection = self.get_mask_strategy(
            mask_strategy=self.mask_strategy, *args, **kwargs
        )
        self.custom_mask = custom_mask

        # Computing patch size and overlap in original dimensions
        self.patch_size_level = round(self.patch_size_target / self.resize_factor)
        self.overlap_level = round(self.overlap_target / self.resize_factor)

        # Getting the tile selection mapping function
        if self.selection_strategy is not None:
            self.tile_selection = self.get_selection_strategy(
                selection_strategy=self.selection_strategy, *args, **kwargs
            )
        else:
            self.tile_selection = None

        # Performing lazy patching
        if lazy:
            self.lazy_patch()

        # Initialization of a counter for filtering valid patches
        self.i = 0

    def __len__(self) -> int:
        """Return the number of valid patches in the slide.

        Returns
        -------
        nb_valid_patches : int
            Number of valid patches (`nb_valid_patches`; None before patching,
            which makes `len` raise `TypeError`).
        """
        return self.nb_valid_patches

    def __iter__(self):
        """Reset the iteration counter and return the patcher itself.

        Returns
        -------
        self : SlidePatcher
            The patcher, used as its own iterator (see `__next__`).
        """
        self.i = 0
        return self

    def __next__(
        self,
    ) -> tuple[PIL.Image.Image, tuple[int, int, int, int]] | tuple[int, int, int, int]:
        """Return the next valid patch and advance the counter.

        Returns
        -------
        tile : PIL.Image.Image | np.ndarray
            Next valid tile, only returned when `xywh_only` is False.

        coords : list[int]
            Coordinates `[x, y, w, h]` of the patch (pixels at `level`).

        Raises
        ------
        StopIteration
            When all valid patches have been returned.
        """

        # Check if it is a valid patch
        if self.i >= self.nb_valid_patches:
            raise StopIteration

        # Get the patch and update counter
        x = self.__getitem__(self.i)
        self.i += 1

        # Return the patch
        return x

    def __getitem__(
        self, index: int
    ) -> tuple[PIL.Image.Image, tuple[int, int, int, int]] | tuple[int, int, int, int]:
        """Return the valid patch at `index`.

        Parameters
        ----------
        index : int
            Index in `valid_patches`, in ``[0, len(self))``. Negative indices are
            not supported.

        Returns
        -------
        tile : PIL.Image.Image | np.ndarray
            Tile read with `get_tile`, only returned when `xywh_only` is False.

        coords : list[int]
            Coordinates `[x, y, w, h]` of the patch (pixels at `level`).

        Raises
        ------
        IndexError
            If `index` is out of range.
        """
        # Check if it is a valid patch
        if 0 <= index < len(self):

            # Retrieve coordinates
            coords = self.valid_patches[index]

            # Retrieve the tile
            if not self.xywh_only:
                res = (self.get_tile(*coords), coords)
            else:
                res = coords

            # Returning the results
            return res

        # Otherwise raise an error
        else:
            raise IndexError("Index out of range")

    def get_mask_strategy(self, mask_strategy: str, *args, **kwargs) -> callable:
        """Resolve a tissue masking function from `mask_strategy_mapping`.

        Parameters
        ----------
        mask_strategy : str
            Mask strategy name (case-insensitive), a key of
            `mask_strategy_mapping`: ``"otsu"`` or ``"luminosity"``.

        *args
            Positional arguments bound to the mask function.

        **kwargs
            Keyword arguments bound to the mask function.

        Returns
        -------
        mask_func : functools.partial
            Mask function called as ``mask_func(slide, mask_level, margin=...)``
            and returning a binary tissue mask.

        Raises
        ------
        NotImplementedError
            If `mask_strategy` is not a key of `mask_strategy_mapping`.
        """
        # Fetch the corresponding selection strategy
        if mask_strategy.lower() in SlidePatcher.mask_strategy_mapping.keys():
            return partial(
                SlidePatcher.mask_strategy_mapping[mask_strategy.lower()],
                *args,
                **kwargs,
            )

        # Otherwise raise an error
        else:
            raise NotImplementedError(
                "mask strategy {} is not implemented.".format(mask_strategy)
                + f"\nPlease choose a valid strategy from: {' ,'.join(SlidePatcher.mask_strategy_mapping.keys())}."
            )

    def get_selection_strategy(
        self, selection_strategy: str, *args, **kwargs
    ) -> callable:
        """Resolve a tile selection function from `tile_selection_mapping`.

        Parameters
        ----------
        selection_strategy : str
            Selection strategy name (case-insensitive), a key of
            `tile_selection_mapping`: ``"hog"``.

        *args
            Positional arguments bound to the selection function.

        **kwargs
            Keyword arguments bound to the selection function.

        Returns
        -------
        selection_func : functools.partial
            Selection function called as ``selection_func(img_list=tiles)`` and
            returning the indices of the selected tiles.

        Raises
        ------
        NotImplementedError
            If `selection_strategy` is not a key of `tile_selection_mapping`.
        """
        # Fetch the corresponding selection strategy
        if selection_strategy.lower() in SlidePatcher.tile_selection_mapping.keys():
            return partial(
                SlidePatcher.tile_selection_mapping[selection_strategy.lower()],
                *args,
                **kwargs,
            )

        # Otherwise raise an error
        else:
            raise NotImplementedError(
                "Tile selection strategy {} is not implemented.".format(
                    selection_strategy
                )
                + f"\nPlease choose a valid strategy from: {' ,'.join(SlidePatcher.tile_selection_mapping.keys())}."
            )

    def lazy_patch(self):
        """Compute the valid patches and save them if requested.

        Raises
        ------
        ValueError
            If `custom_xywh` is not an `(N, 4)` array.
        """

        # Check and load the custom coordinates
        if isinstance(self.custom_xywh, np.ndarray):
            if self.custom_xywh.shape[1] != 4:
                raise ValueError(
                    "custom_xywh must be a (n, 4) array of int [[x, y, w, h]]"
                )
            self.nb_valid_patches, self.valid_patches = (
                len(self.custom_xywh),
                self.custom_xywh,
            )

        # Otherwise, segment and patch the slide
        else:
            self.nb_valid_patches, self.valid_patches = self.patch_sampling()
            # Saving the outputs
        if (self.dst is not None) & self.overwrite:
            self.patch_path = self.save_patch(self.dst, self.save_as)

    def get_seg_mask(self, margin: tuple[int, int] | None = None) -> np.array:
        """Compute the tissue segmentation mask at `level_mask`.

        Parameters
        ----------
        margin : tuple[int, int] | None, optional
            Border removed on each axis (rows, columns), in pixels at
            `level_mask`. Default is None, meaning 1% of the mask height and
            width.

        Returns
        -------
        mask : np.ndarray
            Binary tissue mask of shape `(height, width)` at `level_mask`.

        margin : tuple[int, int]
            Margin actually used.

        Notes
        -----
        If `custom_mask` is set, it is returned alone (not as a tuple) and
        `margin` is ignored.
        """
        if self.custom_mask is not None:
            return self.custom_mask

        # Parsing margin default args
        if margin is None:
            shape_mask = (
                self.slide.level_dimensions[self.level_mask][1],
                self.slide.level_dimensions[self.level_mask][0],
            )
            margin = (shape_mask[0] // 100, shape_mask[1] // 100)

        # Computing segmentation mask
        mask = self.mask_selection(self.slide, self.level_mask, margin=margin)

        # Returning segmentation mask
        return mask, margin

    def patch_sampling(
        self,
        select_tiles: bool = None,
        color_filter: bool = None,
        store_idx: bool = True,
    ) -> tuple[int, list[list[int]]]:
        """Segment the tissue, grid the slide and keep the valid patches.

        Computes the tissue mask (`get_seg_mask`), converts the mask bounding
        box (minus `margin`) to `level` coordinates, builds a grid with a step of
        `patch_size_level - overlap_level` (`grid_blob`), keeps the grid cells on
        tissue (`get_valid_patches`) and optionally applies tile selection and
        colour filtering (`select_patches`).

        Parameters
        ----------
        select_tiles : bool, optional
            Whether to apply `tile_selection`. Default is None, meaning True if
            `tile_selection` is set.

        color_filter : bool, optional
            Whether to apply colour filtering. Default is None, meaning True if
            `color_thresh` is set.

        store_idx : bool, optional
            Whether to store the selected indices in `idx_selected`. Default is
            True.

        Returns
        -------
        nb_valid_patches : int
            Number of valid patches.

        valid_patches : list[list[int]]
            Valid patch coordinates, `(N, 4)` `[x, y, w, h]` rows in pixels at
            `level`.
        """

        # Parsing default args
        if select_tiles is None:
            select_tiles = True if self.tile_selection is not None else False
        if color_filter is None:
            color_filter = True if self.color_thresh is not None else False

        # Tissue segmentation task and retrieving useful coordinates and sizes at the
        # original resolution for patching

        mask, margin = self.get_seg_mask(self.margin)

        min_row, min_col, max_row, max_col = (
            margin[0],
            margin[1],
            mask.shape[0] - margin[0],
            mask.shape[1] - margin[1],
        )

        point_start_mask = min_row, min_col
        point_end_mask = max_row, max_col

        # Converting the coordinates and size to the target level resolution
        shape_at_level = (
            self.slide.level_dimensions[self.level][1],
            self.slide.level_dimensions[self.level][0],
        )
        point_start = get_x_y_to(point_start_mask, mask.shape, shape_at_level)
        point_end = get_x_y_to(point_end_mask, mask.shape, shape_at_level)

        # Creating the grid of the patches, removing the overlapping margin
        patch_shape_no_margin = (
            self.patch_size_level - self.overlap_level,
            self.patch_size_level - self.overlap_level,
        )
        grid_coord = grid_blob(point_start, point_end, patch_shape_no_margin)

        # Obtain the valid patches from tissue segmentation and grid coordinates
        nb_valid_patches, valid_patches = self.get_valid_patches(mask, grid_coord)

        # Perform tile selection and color filtering of the tiles
        if select_tiles or color_filter:
            nb_valid_patches, valid_patches, idx_selected = self.select_patches(
                patch_list=valid_patches,
                return_idx=True,
                color_filter=color_filter,
                select_tiles=select_tiles,
            )
            if store_idx:
                self.idx_selected = idx_selected

        # Returning the valid patches (intersection of the mask on the grid and additional filtering)
        return nb_valid_patches, valid_patches

    def get_invalid_patches(
        self,
        from_tile_method: bool = False,
        from_color_filter: bool = True,
        verbose: bool = True,
    ) -> tuple[int, list[list[int]]]:
        """Return the tissue patches rejected by tile selection.

        Parameters
        ----------
        from_tile_method : bool, optional
            Whether to disable tile selection in the re-run, i.e. count patches
            rejected by `tile_selection`. Default is False.

        from_color_filter : bool, optional
            Whether to disable colour filtering in the re-run, i.e. count patches
            rejected by colour filtering. Default is True.

        verbose : bool, optional
            Whether to print the number of invalid patches. Default is True.

        Returns
        -------
        nb_invalid_patches : int
            Number of rejected patches.

        invalid_patches : list[list[int]]
            Rejected patch coordinates, `[x, y, w, h]` rows in pixels at `level`.

        Notes
        -----
        The re-run uses the default `store_idx=True`, so `idx_selected` is
        overwritten.
        """

        # Parsing select tiles and color filter
        color_filter, select_tiles = None, None
        if from_color_filter:
            color_filter = False
        if from_tile_method:
            select_tiles = False

        # Retrieving all patches from tissue segmentation
        _, all_patches = self.patch_sampling(
            select_tiles=select_tiles, color_filter=color_filter
        )

        # Retrieving invalid patches coordinates
        invalid_patches = np.array(all_patches)[
            ~np.isin(
                np.array(list(map(lambda x: str(x), np.array(all_patches)))),
                np.array(list(map(lambda x: str(x), np.array(self.valid_patches)))),
            )
        ].tolist()

        # Printing information
        if verbose:
            print(
                "Number of invalid patches: {}/{}".format(
                    len(invalid_patches), len(all_patches)
                )
            )

        # Returning number of invalid patches and corresponding coordinates
        return len(invalid_patches), invalid_patches

    def get_valid_patches(
        self, mask: np.ndarray, grid_coord: np.ndarray
    ) -> tuple[int, list[list[int]]]:
        """Keep the grid cells that lie on tissue.

        Each grid point is mapped to mask coordinates; the tissue fraction of a
        square of half-side `patch_size_mask // 2` (at least 1) around the patch
        centre is checked with `mask_percentage` against `mask_tolerance`.
        Patches whose square would cross the mask border are dropped.

        Parameters
        ----------
        mask : np.ndarray
            Binary tissue mask of shape `(height, width)` at `level_mask`.

        grid_coord : np.ndarray
            Grid top-left corners as `(row, col)` pairs, in pixels at `level`.

        Returns
        -------
        nb_valid_patches : int
            Number of valid patches.

        valid_patches : list[list[int]]
            Valid patch coordinates `[x, y, w, h]` (`x = col`, `y = row`,
            `w = h = patch_size_level`), in pixels at `level`.
        """

        # Retrieving the source and target shapes
        shape_at_level = (
            self.slide.level_dimensions[self.level][1],
            self.slide.level_dimensions[self.level][0],
        )
        shape_mask = mask.shape

        # Initialization of a list to store the valid patches at the desired resolution
        patches_at_level = []

        # Computing the size of the patch in the original resolution
        patch_size_mask = get_size_to(
            size=(self.patch_size_level, 0),
            downsample_from=self.downsample_level,
            downsample_to=self.mask_downsample,
        )[0]

        # Computing radius of patch in original resolution
        radius = np.array([max(patch_size_mask // 2, 1), max(patch_size_mask // 2, 1)])

        # For each patch in the grid, check if valid patch and obtain coordinates and image at target resolution
        for coord in grid_coord:

            # Get coordinates in original resolution
            coord_mask = get_x_y_to(coord, shape_at_level, shape_mask)

            # Compute patch's centroids at original resolution
            point_cent_mask = np.array(
                [coord_mask + radius, shape_mask - np.array([1, 1]) - radius]
            ).min(axis=0)

            # Check if valid patch by computing tissue percentage
            if mask_percentage(
                mask=mask,
                point=point_cent_mask,
                radius=radius,
                mask_tolerance=self.mask_tolerance,
            ):

                # Check if entire patch is in image
                still_add = True
                if np.array((coord_mask + radius) != point_cent_mask).any():
                    still_add = False

                # If valid patch, add its coordinates to the list
                if still_add:
                    valid_patch = [
                        coord[1],
                        coord[0],
                        self.patch_size_level,
                        self.patch_size_level,
                    ]
                    patches_at_level.append(valid_patch)  # x, y, w, h

        # Return number of valid patches and the list of valid patches
        return len(patches_at_level), patches_at_level

    def filter_patch_on_color(
        self,
        img_list: list[np.ndarray] | np.ndarray,
        color_thresh: tuple[int, int, int] = None,
        stat_fct: str = None,
        q: float = None,
        remove_background: bool = None,
        background_thresh: int | tuple[int, int, int] = None,
        background_average: bool = None,
        local_average: bool = None,
        filter_size: int | tuple[int, int] = None,
        filter_all: bool = None,
        aggregate: str = None,
    ) -> list[int]:
        """Filter tiles on a per-channel (R, G, B) colour threshold.

        Thin wrapper around `color_filter_selection_torch`; every argument left
        to None falls back to the instance attribute of the same name.

        Parameters
        ----------
        img_list : list[np.ndarray] | np.ndarray
            RGB tiles to filter.

        color_thresh : tuple[int, int, int], optional
            Colour threshold per channel. Default is None (`self.color_thresh`).

        stat_fct : str, optional
            Statistic compared to `color_thresh`. Default is None
            (`self.stat_fct`).

        q : float, optional
            Quantile for quantile-based statistics. Default is None (`self.q`).

        remove_background : bool, optional
            Whether to remove background pixels before computing the statistic.
            Default is None (`self.remove_background`).

        background_thresh : int | tuple[int, int, int], optional
            Background colour threshold. Default is None
            (`self.background_thresh`).

        background_average : bool, optional
            Whether background is detected on channel-averaged pixels. Default is
            None (`self.background_average`).

        local_average : bool, optional
            Whether to apply local averaging first. Default is None
            (`self.local_average`).

        filter_size : int | tuple[int, int], optional
            Local averaging window, in pixels. Default is None
            (`self.filter_size`).

        filter_all : bool, optional
            Whether the criterion must hold on all channels. Default is None
            (`self.filter_all`).

        aggregate : str, optional
            Aggregation of the per-pixel statistic. Default is None
            (`self.aggregate`).

        Returns
        -------
        idx_selected : list[int]
            Indices (in `img_list`) of the tiles kept, as returned by
            `color_filter_selection_torch`.
        """

        # Parsing default args
        if color_thresh is None:
            color_thresh = self.color_thresh
        if stat_fct is None:
            stat_fct = self.stat_fct
        if q is None:
            q = self.q
        if remove_background is None:
            remove_background = self.remove_background
        if background_thresh is None:
            background_thresh = self.background_thresh
        if background_average is None:
            background_average = self.background_average
        if local_average is None:
            local_average = self.local_average
        if filter_size is None:
            filter_size = self.filter_size
        if filter_all is None:
            filter_all = self.filter_all
        if aggregate is None:
            aggregate = self.aggregate

        # Filtering patches based on mean color value for each channel
        return color_filter_selection_torch(
            img_list,
            color_thresh=color_thresh,
            stat_fct=stat_fct,
            q=q,
            remove_background=remove_background,
            background_thresh=background_thresh,
            background_average=background_average,
            local_average=local_average,
            filter_size=filter_size,
            filter_all=filter_all,
            aggregate=aggregate,
        )

    def select_patches(
        self,
        patch_list: list[list] | np.ndarray,
        return_idx: bool = False,
        color_filter: bool = None,
        select_tiles: bool = None,
        *args,
        **kwargs,
    ) -> tuple[int, list[list]] | tuple[int, list[list], list]:
        """Filter tissue patches with tile selection and/or colour filtering.

        Reads the tiles (`get_all_tiles`), applies `tile_selection`, then
        `filter_patch_on_color` on the remaining tiles. Prints a message when no
        patch is kept.

        Parameters
        ----------
        patch_list : list[list] | np.ndarray
            Candidate coordinates, `(N, 4)` `[x, y, w, h]` rows in pixels at
            `level`.

        return_idx : bool, optional
            Whether to also return the selected indices. Default is False.

        color_filter : bool, optional
            Whether to apply colour filtering. Default is None, meaning True if
            `color_thresh` is set.

        select_tiles : bool, optional
            Whether to apply `tile_selection`. Default is None, meaning True if
            `tile_selection` is set.

        *args
            Extra positional arguments passed to `tile_selection`.

        **kwargs
            Extra keyword arguments passed to `tile_selection`.

        Returns
        -------
        nb_valid_patches : int
            Number of patches kept.

        valid_patches : list[list]
            Coordinates of the patches kept.

        idx_selected : np.ndarray
            Indices in `patch_list` of the patches kept, only returned when
            `return_idx` is True.
        """
        # Parsing default args
        if color_filter is None:
            color_filter = True if self.color_thresh is not None else False
        if select_tiles is None:
            select_tiles = True if self.tile_selection is not None else False

        # Retrieving all valid tiles
        tiles = self.get_all_tiles(patch_list)

        # Performing tile selection using the selection strategy
        if select_tiles:
            idx_selected = self.tile_selection(img_list=tiles, *args, **kwargs)
        else:
            idx_selected = np.array(list(range(len(tiles))))

        # Performing tile selection based on color filtering
        if color_filter:
            idx_kept = self.filter_patch_on_color(
                img_list=np.array(tiles)[idx_selected]
            )
            idx_selected = idx_selected[idx_kept]

        # Updating valid patches
        nb_valid_patches = len(idx_selected)
        valid_patches = (np.array(patch_list)[idx_selected]).tolist()

        # Check if any patch selected
        if nb_valid_patches == 0:
            print(
                f"Tile selection failed, no valid patches found. Check the selection strategy parameters."
            )

        # Returning valid patches
        res = nb_valid_patches, valid_patches
        if return_idx:
            res = nb_valid_patches, valid_patches, idx_selected
        return res

    def get_tile(self, x: int, y: int, w: int, h: int) -> PIL.Image.Image | np.ndarray:
        """Read one tile at `level` and resize it to the target patch size.

        Parameters
        ----------
        x : int
            Left coordinate, in pixels at `level`.

        y : int
            Top coordinate, in pixels at `level`.

        w : int
            Width of the region read, in pixels at `level`.

        h : int
            Height of the region read, in pixels at `level`.

        Returns
        -------
        tile : PIL.Image.Image | np.ndarray
            RGB tile of size `patch_size_target` x `patch_size_target`; a
            `PIL.Image.Image` if `pil` is True, else an array of shape
            `(patch_size_target, patch_size_target, 3)`.
        """

        # Get the tile from the slide's PIL image
        if self.pil:
            tile = self.slide.read_region(
                location=(x, y), level=self.level, size=(w, h), numpy=False
            ).convert("RGB")
            tile = tile.resize((self.patch_size_target, self.patch_size_target))

        # Get the tile from the slide's array
        else:
            tile = self.slide.read_region(
                location=(x, y), level=self.level, size=(w, h), numpy=True
            )
            tile = cv2.resize(tile, (self.patch_size_target, self.patch_size_target))[
                :, :, :3
            ]

        # Returning the tile
        return tile

    def get_all_tiles(
        self, patch_list: list[list[int]] | None = None
    ) -> list[PIL.Image.Image] | list[np.ndarray]:
        """Read all tiles at once by cropping the whole image at `level`.

        Loads the full `level` image in memory with `slide.read_whole`, crops
        each patch and resizes it to `patch_size_target`.

        Parameters
        ----------
        patch_list : list[list[int]] | None, optional
            Coordinates `[x, y, w, h]` in pixels at `level`. Default is None,
            meaning `valid_patches`.

        Returns
        -------
        tiles : list[PIL.Image.Image] | list[np.ndarray]
            RGB tiles of size `patch_size_target`; `PIL.Image.Image` if `pil` is
            True, else arrays of shape `(patch_size_target, patch_size_target, 3)`.

        Notes
        -----
        The crops currently iterate over `self.valid_patches`, not over
        `patch_list`.
        """
        if patch_list is None:
            patch_list = self.valid_patches

        # Get the tile from the slide's PIL image
        if self.pil:
            whole = self.slide.read_whole(self.level, numpy=False).convert("RGB")
            tiles = [
                whole.crop((x, y, x + w, y + h)).resize(
                    (self.patch_size_target, self.patch_size_target)
                )
                for (x, y, w, h) in self.valid_patches
            ]

        # Get the tile from the slide's array
        else:
            whole = self.slide.read_whole(self.level, numpy=True)
            tiles = [
                cv2.resize(
                    whole[y : y + h, x : x + w],
                    (self.patch_size_target, self.patch_size_target),
                )[:, :, :3]
                for (x, y, w, h) in self.valid_patches
            ]

        # Returning the tile
        return tiles

    def get_thumbnail(
        self, size: tuple[int, int], numpy: bool = False
    ) -> np.ndarray | PIL.Image.Image:
        """Return a thumbnail of the slide.

        Parameters
        ----------
        size : tuple[int, int]
            Maximum `(width, height)` of the thumbnail, in pixels.

        numpy : bool, optional
            Whether to return an RGB `np.ndarray` instead of a `PIL.Image.Image`.
            Default is False.

        Returns
        -------
        thumbnail : np.ndarray | PIL.Image.Image
            Slide thumbnail; an array of shape `(height, width, 3)` if `numpy`.
        """
        thumbnail = self.slide.get_thumbnail(size)
        if numpy:
            thumbnail = np.array(thumbnail)[:, :, :3]
        return thumbnail

    def visualize_tissue_seg(
        self,
        size: tuple[int, int],
        save_seg: str = None,
        show: bool = False,
        *args,
        **kwargs,
    ) -> str:
        """Plot the tissue segmentation mask with slide annotations.

        Parameters
        ----------
        size : tuple[int, int]
            Size of the rendered mask image, in pixels.

        save_seg : str, optional
            Directory where the figure is saved. Default is None, meaning not
            saved.

        show : bool, optional
            Whether to display the figure. Default is False.

        *args
            Extra positional arguments passed to `visualize_tissue_seg`.

        **kwargs
            Extra keyword arguments passed to `visualize_tissue_seg`.

        Returns
        -------
        seg_path : str
            Path of the saved figure, or None if it was not saved.
        """

        # Computing segmentation mask
        mask, _ = self.get_seg_mask(self.margin)
        mask = mask.astype(np.uint8) * 255
        # Creating annotations
        slide_attrs = {
            "size": (self.width, self.height),
            "mpp": self.pixel_size_0,
            "mag_0": self.mag_0,
            "downsample": self.mask_downsample,
            "strategy": self.mask_strategy,
        }

        # Plotting the tissue segmentation mask
        seg_path = visualize_tissue_seg(
            mask=mask,
            size=size,
            slide_name=self.slide.name,
            slide_attrs=slide_attrs,
            save_seg=save_seg,
            show=show,
            *args,
            **kwargs,
        )

        # Returning path to the saved image
        return seg_path

    def visualize_cut(
        self,
        size: tuple[int, int],
        save_cut: str = None,
        show: bool = False,
        *args,
        **kwargs,
    ) -> str:
        """Plot the valid patches over the slide.

        Parameters
        ----------
        size : tuple[int, int]
            Size of the slide image on which patches are drawn, in pixels.

        save_cut : str, optional
            Directory where the figure is saved. Default is None, meaning not
            saved.

        show : bool, optional
            Whether to display the figure. Default is False.

        *args
            Extra positional arguments passed to `visualize_cut`.

        **kwargs
            Extra keyword arguments passed to `visualize_cut`.

        Returns
        -------
        cut_path : str
            Path of the saved figure, as returned by `visualize_cut`.
        """

        # Creating slide additional annotations
        slide_attrs = {
            "overlap_target": self.overlap_target,
            "overlap_level": self.overlap_level,
            "level": self.level,
            "tissue_tolerance": self.mask_tolerance,
        }

        # Plotting the patches
        cut_path = visualize_cut(
            wsi_slide=self.slide,
            coords=self.valid_patches,
            size=size,
            patch_size_target=self.patch_size_target,
            mag_target=self.mag_target,
            slide_attrs=slide_attrs,
            save_cut=save_cut,
            show_plot=show,
            *args,
            **kwargs,
        )

        # Returning path to saved image
        return cut_path

    def sample_plot_patches(
        self,
        patches: list[list[int]] | np.ndarray = None,
        n_patches: int = None,
        save_path: str = None,
        figsize=None,
        show=True,
    ):
        """Randomly sample patches and plot them as a mosaic.

        Parameters
        ----------
        patches : list[list[int]] | np.ndarray, optional
            Coordinates `[x, y, w, h]` (pixels at `level`) to sample from.
            Default is None, but a value is required (`len` is taken on it).

        n_patches : int, optional
            Number of patches sampled without replacement. Default is None,
            meaning all `patches`.

        save_path : str, optional
            Directory where the figure is saved (created if needed). Default is
            None, meaning not saved.

        figsize : tuple[int, int], optional
            Figure size in inches. Default is None, meaning (5, 5), (10, 10) or
            (15, 15) for up to 16, up to 100 or more tiles.

        show : bool, optional
            Whether to display the figure; otherwise it is closed. Default is
            True.
        """

        # Getting unselected patches
        total_patches = len(patches)
        if not n_patches:
            n_patches = total_patches
        # Sampling in the invalid patches
        if n_patches < total_patches:
            i_patches = np.random.choice(
                np.arange(total_patches), n_patches, replace=False
            ).tolist()
            patches = np.array(patches)[i_patches, :].tolist()
        else:
            patches = np.array(patches).tolist()

        # Retrieving corresponding tiles image
        invalid_img = []
        for patch in patches:
            invalid_img.append(self.get_tile(*patch))

        # Plotting the images
        if figsize is None:
            if len(invalid_img) <= 16:
                figsize = (5, 5)
            elif len(invalid_img) <= 100:
                figsize = (10, 10)
            else:
                figsize = (15, 15)
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
        ax = mosaic(
            invalid_img,
            title=f"Tiles not selected in: {self.slide.name}",
            ax=ax,
        )
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, "{}.jpeg".format(self.slide.name)))
        if show:
            plt.show()
        else:
            plt.close(fig)

    def sample_plot_invalid_patches(
        self,
        n_patches: int = 36,
        save_path: str = None,
        invalid_on: str = "all",
        figsize=None,
        verbose: bool = True,
        show: bool = True,
    ):
        """Compute the rejected patches, then sample and plot some of them.

        Parameters
        ----------
        n_patches : int, optional
            Number of patches sampled. Default is 36.

        save_path : str, optional
            Directory where the figure is saved. Default is None, meaning not
            saved.

        invalid_on : str, optional
            Which rejection to show: ``"all"`` (tile selection and colour
            filtering), ``"color"`` (colour filtering only) or ``"quality"``
            (tile selection only). Any other value behaves as ``"all"``. Default
            is ``"all"``.

        figsize : tuple[int, int], optional
            Figure size in inches. Default is None (see `sample_plot_patches`).

        verbose : bool, optional
            Whether to print the number of invalid patches. Default is True.

        show : bool, optional
            Whether to display the figure. Default is True.
        """

        # Parsing which selection process we want to plot
        from_tile_method = True
        from_color_filter = True
        if invalid_on == "color":
            from_tile_method = False
        elif invalid_on == "quality":
            from_color_filter = False

        # Getting unselected patches
        nb_invalid_patches, invalid_patches = self.get_invalid_patches(
            from_tile_method=from_tile_method,
            from_color_filter=from_color_filter,
            verbose=verbose,
        )

        # Sampling and plotting the invalid patches
        if nb_invalid_patches > 0:
            self.sample_plot_patches(
                patches=invalid_patches,
                n_patches=n_patches,
                save_path=save_path,
                figsize=figsize,
                show=show,
            )

    def save_patch(self, dst: str = None, save_as: str = "h5") -> str:
        """Save the valid patch coordinates and patching metadata.

        Writes ``<dst>/patches/<slide.name>_patches.<save_as>`` (overwritten if
        it exists). 
        
        For ``"h5"``, the `coords` dataset holds `valid_patches` as
        an `(N, 4)` `xywh` array (pixels at `level`) and its attributes store
        `magnification`, `mpp`, `target_magnification`, `target_patch_size`,
        `target_mpp`, `target_overlap`, `level`, `level_size`,
        `level_patch_size`, `level_overlap`, `tissue_thr`, `mask`, `name` and
        `savetodir`.

        Parameters
        ----------
        dst : str, optional
            Output directory. Default is None, but a path is required.
            
        save_as : str, optional
            Output format; only ``"h5"`` is supported. Default is ``"h5"``.

        Returns
        -------
        patch_file : str
            Path of the written file.

        Raises
        ------
        ValueError
            If `save_as` is not ``"h5"``.
        """

        # Saving the patches as h5 file
        if save_as == "h5":

            # Retrieving the coordinates and storing useful patches attributes
            coords = {"coords": np.array(self.valid_patches)}
            attributes = {
                "magnification": self.mag_0,
                "mpp": self.pixel_size_0,
                "target_magnification": self.mag_target,
                "target_patch_size": self.patch_size_target,
                "target_mpp": self.pixel_size_target,
                "target_overlap": self.overlap_target,
                "level": self.level,
                "level_size": self.slide.level_dimensions[self.level],
                "level_patch_size": self.patch_size_level,
                "level_overlap": self.overlap_level,
                "tissue_thr": self.mask_tolerance,
                "mask": self.mask_strategy,
                "name": self.slide.name,
                "savetodir": dst,
            }

            # Creating path to the patch file
            os.makedirs(os.path.join(dst, "patches"), exist_ok=True)
            patch_file = os.path.join(
                dst, "patches", f"{self.slide.name}_patches.{save_as}"
            )

            # Save the assets and attributes to a h5 file
            save_h5(
                patch_file, assets=coords, attributes={"coords": attributes}, mode="w"
            )

        # Otherwise, raise an error
        else:
            raise ValueError(
                f"Invalid save_as argument: {save_as}. Only h5 files are supported."
            )

        # Returning the path to the patch file
        return patch_file



