"""Read pyramidal WSIs with OpenSlide.

Provides a wrapper around `openslide.OpenSlide` plus functional helpers
to open a slide, read a whole pyramid level or a region, and inspect the
pyramid metadata.
"""

# General libraries
from typing import Optional
import os
from pathlib import Path

# Data libraries
import numpy as np

# Image reader
import openslide
import PIL
from openslide import OpenSlide

OPENSLIDE_READABLE_FORMATS = [
    ".svs",
    ".tif",
    ".vms",
    ".vmu",
    ".ndpi",
    ".scn",
    ".mrxs",
    ".tiff",
    ".svslide",
    ".bif",
]

MPP_KEYS = [
    openslide.PROPERTY_NAME_MPP_X,
    "openslide.mirax.MPP",
    "aperio.MPP",
    "hamamatsu.XResolution",
    "openslide.comment",
]


def parse_slide_path(slide: str | Path) -> tuple[str, str]:
    """Split a slide path into name, stem and parent directory.

    Parameters
    ----------
    slide : str | Path
        Path to the slide file.

    Returns
    -------
    name : str
        File name with extension (`Path.name`).

    stem : str
        File name without extension (`Path.stem`).

    parent : Path
        Parent directory of the slide.

    Raises
    ------
    FileNotFoundError
        If `slide` is not an existing file.

    Notes
    -----
    `slide.is_file()` is called before the conversion to `Path`, so a `str`
    input currently raises `AttributeError`.
    """

    if not slide.is_file():
        raise FileNotFoundError(f"Configuration file not found: {slide}")

    slide = Path(slide)
    return slide.name, slide.stem, slide.parent


def check_openslide(slide: str | Path):
    """Check whether a slide can be opened with `openslide.OpenSlide`.

    Parameters
    ----------
    slide : str | Path
        Path to the slide file.

    Returns
    -------
    is_readable : bool
        True if the second element returned by `parse_slide_path`, lowercased,
        is in `OPENSLIDE_READABLE_FORMATS`.

    Raises
    ------
    FileNotFoundError
        If `slide` is not an existing file (from `parse_slide_path`).
    """
    _, ext, _ = parse_slide_path(slide)
    is_readable = ext.lower() in OPENSLIDE_READABLE_FORMATS
    return is_readable


def get_slide_reader(slide):
    """Return the reader class suited to open a slide.

    Parameters
    ----------
    slide : str | Path
        Path to the slide file.

    Returns
    -------
    reader : type
        `OpenWSI` if the slide format is readable by OpenSlide.

    Raises
    ------
    TypeError
        If no reader supports the slide extension.

    FileNotFoundError
        If `slide` is not an existing file (from `parse_slide_path`).
    """

    if check_openslide(slide):
        return OpenWSI
    # possibility to add slide readers
    else:
        _, ext, _ = parse_slide_path(slide)
        raise TypeError(
            f"Slide: {slide} as invalid extension: {ext}. Accepts: {OPENSLIDE_READABLE_FORMATS}"
        )


def openslide_metadata_to_xml(slide: str | Path):
    """Convert OpenSlide metadata to XML (not implemented).

    Parameters
    ----------
    slide : str | Path
        Path to the slide file (unused).

    Returns
    -------
    message : str
        The placeholder string `"Not implemented yet"`.
    """
    return "Not implemented yet"


# get metadata of slide (uses openslide to handle the image, must be compatible)
def get_openslide_pyramid_info(slide, verbose=False):
    """Collect pyramid metadata of an OpenSlide-compatible slide.

    Parameters
    ----------
    slide : str | openslide.OpenSlide
        Slide path or already opened `OpenSlide` object.

    verbose : bool, optional
        If True, print the collected metadata. Default is False.

    Returns
    -------
    infos : dict[str, Any]
        Dictionary with keys:

        - `"level_count"`: number of pyramid levels.
        - `"size_0"`: `(height, width)` of level 0, in pixels.
        - `"res_0"`: `(mpp_y, mpp_x)` at level 0, in microns per pixel.
        - `"objective"`: objective power (magnification), as `int`.
        - `"downsampling"`: list of per-level downsample factors, as `int`.

    Raises
    ------
    KeyError
        If one of the required `openslide.*` properties is missing.

    ValueError
        If a property cannot be converted with `int` (e.g. a non-integer
        downsample string such as `"4.0001"`).
    """
    # assert slide format
    if isinstance(slide, str):
        slide = OpenSlide(slide)
    slide_prop = dict(slide.properties)
    infos = {
        "level_count": slide.level_count,
        "size_0": (slide.level_dimensions[0][1], slide.level_dimensions[0][0]),
        "res_0": (
            float(slide_prop["openslide.mpp-y"]),
            float(slide_prop["openslide.mpp-x"]),
        ),  # YX = HW
        "objective": int(slide_prop["openslide.objective-power"]),
        "downsampling": [
            int(slide_prop[f"openslide.level[{l}].downsample"])
            for l in range(slide.level_count)
        ],
    }
    if verbose:
        print(infos)
    return infos


class OpenWSI:
    """Wrap `openslide.OpenSlide` with cached metadata, mpp and magnification.

    Parameters
    ----------
    img_path : str
        Path to the slide file.

    name : str, optional
        Name used instead of the file name; its extension (if any) is split
        into `ext`. Default is None (use the basename of `img_path`).

    mpp : float, optional
        Pixel size at level 0, in microns per pixel. Default is None (read it
        from the slide metadata with `_fetch_mpp`).

    Attributes
    ----------
    img_path : str
        Path to the slide file.

    name : str
        Slide name without extension.

    ext : str
        Slide extension (e.g. `".svs"`), possibly empty.

    img : openslide.OpenSlide
        Underlying OpenSlide reader.

    dimensions : tuple[int, int]
        `(width, height)` of level 0, in pixels.

    width, height : int
        Width and height of level 0, in pixels.

    level_count : int
        Number of pyramid levels.

    level_dimensions : tuple[tuple[int, int], ...]
        `(width, height)` of each level, in pixels.

    level_downsamples : tuple[float, ...]
        Downsample factor of each level relative to level 0.

    properties : Mapping[str, str]
        Raw OpenSlide properties.

    mpp : float | None
        Pixel size at level 0, in microns per pixel.

    magnification : int
        Estimated objective magnification at level 0.

    Raises
    ------
    openslide.OpenSlideError
        If OpenSlide cannot open `img_path`.

    ValueError
        If `mpp` is outside the range handled by `_fetch_magnification`.
    """

    def __init__(
        self, img_path: str, name: Optional[str] = None, mpp: Optional[float] = None
    ):
        """Open the slide and fetch its metadata, mpp and magnification."""
        self.img_path = img_path
        if name is None:
            self.name, self.ext = os.path.splitext(os.path.basename(img_path))
        else:
            self.name, self.ext = os.path.splitext(name)
        self.img = openslide.OpenSlide(self.img_path)
        self._fetch_meta()

        self.mpp = mpp
        if self.mpp is None:
            self.mpp = self._fetch_mpp()

        self.magnification = self._fetch_magnification()

    def _fetch_meta(self):
        """Fetch and store slide metadata as attributes.

        Sets `dimensions`, `width`, `height`, `level_count`,
        `level_dimensions`, `level_downsamples` and `properties` from the
        underlying `OpenSlide` object.
        """
        self.dimensions = self.img.dimensions
        self.width, self.height = self.dimensions
        self.level_count = self.img.level_count
        self.level_dimensions = self.img.level_dimensions
        self.level_downsamples = self.img.level_downsamples
        self.properties = self.img.properties

    def _fetch_mpp(self) -> float | None:
        """Retrieve the level-0 pixel size (mpp) from slide metadata.

        The keys of `MPP_KEYS` are tried in order; the first value convertible
        to `float` is used. Otherwise, the mpp is derived from
        `tiff.XResolution` and `tiff.ResolutionUnit` (centimeter or inch).

        Returns
        -------
        mpp_x : float | None
            Pixel size along x in microns per pixel, rounded to 4 decimals, or
            None if the TIFF resolution is missing or its unit is unsupported.

        Notes
        -----
        `mpp_x` is not initialised before the loop, so if no key of
        `MPP_KEYS` yields a value, the `mpp_x is None` check raises
        `UnboundLocalError` instead of falling back to the TIFF tags.
        """

        # Search for mpp_x
        for key in MPP_KEYS:
            if key in self.img.properties:
                try:
                    mpp_x = float(self.img.properties[key])
                    break
                except ValueError:
                    continue
        # Convert pixel resolution to mpp
        if mpp_x is None:
            x_resolution = self.img.properties.get("tiff.XResolution", None)
            unit = self.img.properties.get("tiff.ResolutionUnit", None)
            if not x_resolution or not unit:
                return None
            if unit == "CENTIMETER" or unit == "centimeter":
                mpp_x = 10000 / float(x_resolution)  # 1 cm = 10,000 microns
            elif unit == "INCH":
                mpp_x = 25400 / float(x_resolution)  # 1 inch = 25,400 microns
            else:
                return None  # Unsupported unit -- add more conditions is needed.
        mpp_x = round(mpp_x, 4)
        return mpp_x

    def _fetch_magnification(self) -> int:
        """Estimate the objective magnification from the level-0 mpp.

        Thresholds on `mpp` (microns per pixel): < 0.16 -> 80, < 0.2 -> 60,
        < 0.3 -> 40, < 0.6 -> 20, < 1.2 -> 10, < 2.4 -> 5. Without mpp, the
        OpenSlide objective-power property is used.

        Returns
        -------
        magnification : int
            Estimated objective magnification.

        Raises
        ------
        ValueError
            If `mpp` is greater than or equal to 2.4.

        Notes
        -----
        When `self.mpp` is None, `mpp_x` is never assigned and the
        `mpp_x is not None` check raises `UnboundLocalError`, so the
        objective-power fallback is unreachable.
        """
        if self.mpp is not None:
            mpp_x = self.mpp
        if mpp_x is not None:
            if mpp_x < 0.16:
                return 80
            elif mpp_x < 0.2:
                return 60
            elif mpp_x < 0.3:
                return 40
            elif mpp_x < 0.6:
                return 20
            elif mpp_x < 1.2:
                return 10
            elif mpp_x < 2.4:
                return 5
            else:
                raise ValueError(f"mpp as unexpected value: mpp={mpp_x}")
        else:
            mag = self.img.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
            return int(mag)

    def get_thumbnail(self, size: tuple = (1024, 1024)) -> PIL.Image.Image:
        """Generate a thumbnail image of the whole slide.

        Parameters
        ----------
        size : tuple[int, int], optional
            Maximum `(width, height)` of the thumbnail, in pixels; the aspect
            ratio is preserved. Default is `(1024, 1024)`.

        Returns
        -------
        thumbnail : PIL.Image.Image
            RGB thumbnail of the slide.
        """
        return self.img.get_thumbnail(size)

    def read_region(
        self, location: tuple, level: int, size: tuple, numpy: bool = True
    ) -> PIL.Image.Image | np.ndarray:
        """Read a rectangular region of the slide at a given pyramid level.

        Parameters
        ----------
        location : tuple[int, int]
            `(x, y)` top-left corner expressed in the `level` reference frame;
            it is converted to level 0 with `get_xy_0` before reading.

        level : int
            Pyramid level to read from.

        size : tuple[int, int]
            `(width, height)` of the region at `level`, in pixels.

        numpy : bool, optional
            If True, return an RGB array (alpha channel dropped) instead of a
            PIL image. Default is True.

        Returns
        -------
        crop : np.ndarray | PIL.Image.Image
            `(height, width, 3)` uint8 RGB array if `numpy`, otherwise the RGBA
            PIL image returned by OpenSlide.
        """
        x0, y0 = self.get_xy_0(location, level, integer=True)
        # location should be a tuple giving the top left pixel in the level 0 reference frame
        crop = self.img.read_region((x0, y0), level, size)
        if numpy:
            crop = np.array(crop)[:, :, :3]
        return crop

    def read_whole(
        self, level: int, numpy: bool = True
    ) -> PIL.Image.Image | np.ndarray:
        """Read the entire slide image at a given pyramid level.

        Parameters
        ----------
        level : int
            Pyramid level to read. If it is not a valid level index, the last
            (lowest-resolution) level is used.

        numpy : bool, optional
            If True, return an RGB array (alpha channel dropped) instead of a
            PIL image. Default is True.

        Returns
        -------
        whole : np.ndarray | PIL.Image.Image
            `(height, width, 3)` uint8 RGB array if `numpy`, otherwise the RGBA
            PIL image returned by OpenSlide.
        """
        if level >= len(self.level_dimensions):
            level = -1
        whole = self.img.read_region(
            location=(0, 0), level=level, size=self.level_dimensions[level]
        )
        if numpy:
            whole = np.array(whole)[:, :, :3]
        return whole

    def get_best_level_for_downsample(
        self, ask_downsample: float, precision: float = 0.01
    ) -> tuple[int, int, float]:
        """Find the pyramid level best suited to a requested downsample factor.

        If a level matches `ask_downsample` within `precision`, it is returned
        with a resize factor of 1. Otherwise, for downsampling, the deepest
        level with a downsample lower than or equal to `ask_downsample` is
        chosen; for upsampling (`ask_downsample` below the level-0
        downsample), the first level with a downsample greater than or equal
        to it is chosen.

        Parameters
        ----------
        ask_downsample : float
            Requested downsample factor relative to level 0.

        precision : float, optional
            Absolute tolerance for an exact match. Default is 0.01.

        Returns
        -------
        level : int
            Selected pyramid level.

        level_downsample : float
            Downsample factor of the selected level.

        resize_factor : float
            Factor to apply to the image read at `level` to reach
            `ask_downsample` (1 for an exact match).

        Raises
        ------
        ValueError
            If no suitable level is found.
        """
        level_downsamples = self.level_downsamples
        # First, check for a close match
        for level_best, level_downsample in enumerate(level_downsamples):
            if abs(level_downsample - ask_downsample) <= precision:
                return (
                    level_best,
                    level_downsample,
                    1,
                )  # Exact match, no custom downsampling needed
        # If not,
        if ask_downsample >= level_downsamples[0]:
            # Downsampling: find the highest level_downsample less than or equal to the desired downsample
            level_best = None
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample <= ask_downsample:
                    level_best = level
                    resize_factor = level_downsample / ask_downsample
                else:
                    break  # level_downsamples are sorted, no need to check further
            if level_best is not None:
                return level_best, level_downsamples[level_best], resize_factor
        else:
            # Upsampling: find the smallest level_downsample greater than or equal to the desired downsample
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample >= ask_downsample:
                    resize_factor = ask_downsample / level_downsample
                    return level, level_downsamples[level], resize_factor

        # If no suitable level is found, raise an error
        raise ValueError(f"No level found for downsample {ask_downsample}.")

    def get_xy_0(self, point, level, integer=True) -> tuple[int, int]:
        """Convert coordinates from a given pyramid level to level 0.

        Parameters
        ----------
        point : tuple[float, float]
            `(x, y)` coordinates in the `level` reference frame, in pixels.

        level : int
            Pyramid level `point` is expressed at.

        integer : bool, optional
            If True, truncate the result with `int`. Default is True.

        Returns
        -------
        point_0 : tuple[int, int] | tuple[float, float]
            `(x, y)` coordinates at level 0, in pixels.
        """
        x, y = point
        x_0 = x * self.dimensions[0] / self.level_dimensions[level][0]
        y_0 = y * self.dimensions[1] / self.level_dimensions[level][1]
        if integer:
            point_0 = (int(x_0), int(y_0))
        else:
            point_0 = (x_0, y_0)
        return point_0


def get_slide_whole(slide, level=None, numpy=True):
    """Read the whole slide image at a given pyramid level.

    Parameters
    ----------
    slide : str | openslide.OpenSlide
        Slide path or already opened `OpenSlide` object.

    level : int, optional
        Pyramid level to read. If None or greater than the last level, the
        last (lowest-resolution) level is used. Default is None.

    numpy : bool, optional
        If True, return an RGB array (alpha channel dropped) instead of a PIL
        image. Default is True.

    Returns
    -------
    sample : np.ndarray | PIL.Image.Image
        `(height, width, 3)` uint8 RGB array if `numpy`, otherwise the RGBA PIL
        image returned by OpenSlide.
    """
    if isinstance(slide, str):
        slide = OpenSlide(slide)
    if level is None:
        level = slide.level_count - 1
    elif level > slide.level_count - 1:
        print(" level ask is too low... It was setted accordingly")
        level = slide.level_count - 1
    sample = slide.read_region((0, 0), level, slide.level_dimensions[level])
    if numpy:
        sample = np.array(sample)[:, :, 0:3]
    return sample


def get_slide_image(slide, para, numpy=True):
    """Read a crop of a slide described by a set of parameters.

    Parameters
    ----------
    slide : str | openslide.OpenSlide
        Slide path or already opened `OpenSlide` object.

    para : dict[str, int] | list[int]
        Crop parameters, either a dict with keys `"x"`, `"y"`, `"xsize"`,
        `"ysize"`, `"level"`, or a sequence of 5 integers
        `[x, y, size_x, size_y, level]`. `x`, `y` are the top-left corner at
        level 0 and `size_x`, `size_y` the crop size at `level`, in pixels.

    numpy : bool, optional
        If True, return an RGB array (alpha channel dropped) instead of a PIL
        image. Default is True.

    Returns
    -------
    crop : np.ndarray | PIL.Image.Image
        `(size_y, size_x, 3)` uint8 RGB array if `numpy`, otherwise the RGBA
        PIL image returned by OpenSlide.

    Raises
    ------
    NameError
        If `para` is a sequence whose length is not 5.
    """
    if isinstance(para, dict):
        slide = OpenSlide(slide) if isinstance(slide, str) else slide
        slide = slide.read_region(
            (para["x"], para["y"]), para["level"], (para["xsize"], para["ysize"])
        )
        if numpy:
            slide = np.array(slide)[:, :, 0:3]
    else:
        if len(para) != 5:
            raise NameError("Not enough parameters...")
        slide = OpenSlide(slide) if isinstance(slide, str) else slide
        slide = slide.read_region((para[0], para[1]), para[4], (para[2], para[3]))
        if numpy:
            slide = np.array(slide)[:, :, 0:3]
    return slide
