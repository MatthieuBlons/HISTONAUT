"""
Test WSI patchification with `SlidePatcher`.

For each slide found in `--src`, compute a tissue mask, cut the slide into
patches and save the patch coordinates to `<dst>/patches/<slide>_patches.h5`.

Optional outputs (written under the same run directory):

- `--save_mask`: tissue segmentation thumbnails (`segmentation/`).
- `--save_grid`: patch grid drawn over the slide (`visualization/`).
- `--save_tiles`: every patch as a jpeg, resized to `--tile_mpp`
  (`tiles_<mpp>mpp/`), plus a csv summary per run.

Arguments can recive a YAML file (`--config`). Values from the file
replace the parser defaults; arguments typed on the command line still win.

Example
-------
python test/1_test_patchify.py \\
    --src HISTONAUT/data/HE/TCGA-5M-AAT6-01Z-00-DX1.8834C952-14E3-4491-8156-52FC917BB014.svs \\
    --dst . \\
    --verbose \\
    --clock \\
    --tqdm \\
    --config ./configs/1_test_patchify.yaml \\
"""

from __future__ import annotations

# General libraries
import logging
import warnings
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
)
from pathlib import Path
from typing import Any

# Data libraries
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Project modules
from histonaut.patcher.cut import SlidePatcher
from histonaut.patcher.io import read_h5_coords
from histonaut.slide.reader import get_slide_reader

# Test helpers
from logger import (
    configure_logging,
    load_yaml_config,
    log_configuration,
    timetracker,
)

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


# Coords h5 attribute -> SlidePatcher keyword, used with `--custom_coords`.
CUSTOM_COORDS_ATTRS = {
    "target_magnification": "mag_target",
    "target_patch_size": "patch_size",
    "target_overlap": "overlap",
    "tissue_thr": "mask_tolerance",
    "mask": "mask_strategy",
}

VISUALIZATION_SIZE = (1024, 1024)


# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------


def build_test_parser() -> ArgumentParser:
    """
    Build the patchification test argument parser.

    Patcher arguments are named after the `SlidePatcher` keywords so that a
    YAML config can use the same names.

    Returns
    -------
    argparse.ArgumentParser
        Configured command-line parser.
    """
    parser = ArgumentParser(
        description="Patchify WSIs with SlidePatcher.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        required=True,
        help="Directory searched (recursively) for slides. Required.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        required=True,
        help="Directory to store outputs. Required.",
    )
    parser.add_argument(
        "--list",
        type=str,
        nargs="*",
        default=None,
        help="Slide file names to process. When omitted, process all slides.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="svs",
        help="Slide file extension, without the dot.",
    )
    parser.add_argument(
        "--custom_coords",
        type=str,
        default=None,
        help="Directory holding `<slide>_patches.h5` coords to reuse.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Reprocess slides whose coords file already exists.",
    )
    parser.add_argument(
        "--save_mask",
        action="store_true",
        default=False,
        help="Save the tissue segmentation visualization as jpeg.",
    )
    parser.add_argument(
        "--save_grid",
        action="store_true",
        default=False,
        help="Save the patch grid visualization as jpeg.",
    )
    parser.add_argument(
        "--save_tiles",
        action="store_true",
        default=False,
        help="Save every patch as jpeg, resized to --tile_mpp.",
    )
    parser.add_argument(
        "--tile_mpp",
        type=float,
        default=None,
        help="Pixel size (mpp) of saved tiles. Defaults to the patcher mpp.",
    )

    # Runtime
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Display detailed logging.",
    )
    parser.add_argument(
        "--clock",
        action="store_true",
        default=False,
        help="Display total execution timing.",
    )
    parser.add_argument(
        "--tqdm",
        action="store_true",
        default=False,
        help="Display a progress bar.",
    )

    # Patcher
    parser.add_argument(
        "--mag_target",
        type=int,
        default=20,
        help="Magnification at which patches are extracted.",
    )
    parser.add_argument(
        "--pixel_size_0",
        type=float,
        default=None,
        help="Level 0 pixel size (mpp). Defaults to the slide metadata.",
    )
    parser.add_argument(
        "--pixel_size_target",
        type=float,
        default=None,
        help="Pixel size (mpp) at which patches are extracted.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Patch size in pixels, at the target magnification.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Patch overlap in pixels, at the target magnification.",
    )
    parser.add_argument(
        "--mask_strategy",
        type=str,
        default="otsu",
        help="Tissue segmentation strategy.",
    )
    parser.add_argument(
        "--mask_downsample",
        type=int,
        default=32,
        help="Downsampling factor used for tissue segmentation.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=None,
        help="Margin (pixels) around the tissue mask.",
    )
    parser.add_argument(
        "--mask_tolerance",
        type=float,
        default=0.5,
        help="Minimum tissue proportion in a valid patch.",
    )
    parser.add_argument(
        "--selection_strategy",
        type=str,
        default=None,
        help="Optional patch selection strategy (e.g. hog).",
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML file whose values replace the parser defaults.",
    )

    return parser


def parse_test_arguments(
    known_args: list[str] | None = None,
) -> Namespace:
    """
    Parse and validate the test arguments.

    Parameters
    ----------
    known_args : list of str, optional
        Explicit argument sequence. When None, use command-line arguments.

    Returns
    -------
    argparse.Namespace
        Validated configuration.

    Raises
    ------
    KeyError
        If the YAML config holds a key that is not a parser argument.

    ValueError
        If `src` or `dst` is missing.
    """
    parser = build_test_parser()
    args = parser.parse_args(known_args)

    # Config values become defaults, so explicit command-line args still win
    if args.config is not None:
        config = load_yaml_config(args.config)
        unknown_keys = set(config) - set(vars(args))
        if unknown_keys:
            raise KeyError(
                f"Unknown keys in `config` ({args.config}): {sorted(unknown_keys)}. "
                f"Expected a subset of: {sorted(vars(args))}."
            )
        parser.set_defaults(**config)
        args = parser.parse_args(known_args)

    for required in ("src", "dst"):
        if getattr(args, required) is None:
            raise ValueError(f"`{required}` is required (command line or config).")

    return args


def build_test_directory(
    args: Namespace,
    auto_name: bool = True,
) -> dict[str, Path]:
    """
    Create the output directories.

    Parameters
    ----------
    args : argparse.Namespace
        Test configuration.

    auto_name : bool, default=True
        Create a run subdirectory named after the patching parameters.

    Returns
    -------
    dict of str to pathlib.Path
        Output directories: always `root` and `patches`, plus `segmentation`,
        `visualization` and `tiles` when the matching `--save_*` flag is set.
    """
    root = Path(args.dst)
    if auto_name:
        root = root / (
            f"features_{args.mag_target}x_{args.patch_size}px_{args.overlap}px_overlap"
        )

    # SlidePatcher writes its coords under `<root>/patches`
    output_directories = {
        "root": root,
        "patches": root / "patches",
    }
    if args.save_mask:
        output_directories["segmentation"] = root / "segmentation"
    if args.save_grid:
        output_directories["visualization"] = root / "visualization"
    if args.save_tiles:
        tile_mpp = "auto" if args.tile_mpp is None else args.tile_mpp
        output_directories["tiles"] = root / f"tiles_{tile_mpp}mpp"

    for directory in output_directories.values():
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return output_directories


def resolve_input_slides(
    args: Namespace,
    patches_dir: Path,
) -> list[Path]:
    """
    List the slides to process.

    Parameters
    ----------
    args : argparse.Namespace
        Test configuration.

    patches_dir : pathlib.Path
        Directory holding the `<slide>_patches.h5` files already computed.

    Returns
    -------
    list of pathlib.Path
        Slides to process, sorted.

    Raises
    ------
    FileNotFoundError
        If no slide with extension `ext` is found in `src`.
    """
    src = Path(args.src)
    slides = sorted(src.glob(f"**/*.{args.ext}"))
    LOGGER.info(
        "%s slide(s) with extension .%s found in %s",
        len(slides),
        args.ext,
        src,
    )

    if args.list is not None:
        requested_names = {Path(slide).name for slide in args.list}
        slides = [slide for slide in slides if slide.name in requested_names]
        LOGGER.info("%s slide(s) kept from `list`", len(slides))

    if not slides:
        raise FileNotFoundError(
            f"No slide with extension `ext`={args.ext!r} found in `src`={src}."
        )

    # Skip slides already processed, unless asked to overwrite
    if not args.overwrite:
        processed_names = {
            path.stem
            for path in patches_dir.glob(f"*.h5")
        }
        slides = [slide for slide in slides if slide.stem not in processed_names]
        LOGGER.info("%s slide(s) already processed are skipped", len(processed_names))

    LOGGER.info("%s slide(s) will be processed", len(slides))
    return slides


def resolve_patcher_kwargs(
    args: Namespace,
    slide_name: str,
) -> dict[str, Any]:
    """
    Gather the `SlidePatcher` keywords for one slide.

    With `--custom_coords`, coordinates and patching parameters are read from
    `<custom_coords>/<slide>_patches.h5` and take precedence over `args`.

    Parameters
    ----------
    args : argparse.Namespace
        Test configuration.

    slide_name : str
        Slide name, without extension.

    Returns
    -------
    dict of str to Any
        Keyword arguments for `SlidePatcher` (except the slide itself).
    """
    patcher_kwargs = {
        "pixel_size_0": args.pixel_size_0,
        "pixel_size_target": args.pixel_size_target,
        "mag_target": args.mag_target,
        "patch_size": args.patch_size,
        "overlap": args.overlap,
        "mask_downsample": args.mask_downsample,
        "margin": args.margin,
        "mask_strategy": args.mask_strategy,
        "mask_tolerance": args.mask_tolerance,
        "selection_strategy": args.selection_strategy,
        "custom_xywh": None,
    }

    if args.custom_coords is not None:
        coords_path = Path(args.custom_coords) / f"{slide_name}.h5"
        attrs, coords = read_h5_coords(coords_path)
        patcher_kwargs["custom_xywh"] = coords
        for attr_name, kwarg_name in CUSTOM_COORDS_ATTRS.items():
            if attr_name in attrs:
                patcher_kwargs[kwarg_name] = attrs[attr_name]

    return patcher_kwargs


def export_tiles_as_jpeg(
    patcher: SlidePatcher,
    tiles_dir: Path,
    tile_mpp: float | None = None,
) -> list[dict[str, str]]:
    """
    Save every patch of a slide as a jpeg.

    Parameters
    ----------
    patcher : SlidePatcher
        Patcher built with `xywh_only=False`.

    tiles_dir : pathlib.Path
        Output directory.

    tile_mpp : float, optional
        Pixel size of the saved tiles. When None, keep the patcher pixel size.

    Returns
    -------
    list of dict of str to str
        One summary row (`slide_name`, `image_path`) per saved tile.
    """
    if tile_mpp is None:
        tile_mpp = patcher.pixel_size_target
    resize_factor = patcher.pixel_size_target / tile_mpp

    summary_rows = []
    for tile, (x, y, w, h) in patcher:
        image_name = f"{patcher.slide.name}_{patcher.level}_{x}_{y}_{w}_{h}.jpeg"
        image_path = tiles_dir / image_name
        tile_width, tile_height = (np.array((w, h)) * resize_factor).astype(int)
        tile = cv2.resize(
            tile,
            (tile_width, tile_height),
        )
        cv2.imwrite(
            str(image_path),
            cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
        )
        summary_rows.append(
            {
                "slide_name": patcher.slide.name,
                "image_path": str(image_path),
            }
        )

    return summary_rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main(
    known_args: list[str] | None = None,
) -> Path:
    """
    Test WSI patchification.

    Parameters
    ----------
    known_args : list of str, optional
        Explicit arguments. When None, use command-line arguments.

    Returns
    -------
    pathlib.Path
        Run output directory.
    """
    args = parse_test_arguments(known_args=known_args)
    configure_logging(verbose=args.verbose)
    log_configuration(vars(args))

    output_dirs = build_test_directory(args)
    slides = resolve_input_slides(args, output_dirs["patches"])

    timer = timetracker(verbose=args.clock)
    progress = tqdm(
        slides,
        desc=f"Tiling {args.patch_size}x{args.patch_size}px at {args.mag_target}x",
        unit="wsi",
        disable=not args.tqdm,
    )

    timer.tic()
    for slide_path in progress:
        try:
            slide = get_slide_reader(slide_path)(slide_path)
        except Exception as error:
            LOGGER.warning("Cannot open %s, skipped: %s", slide_path, error)
            continue
        progress.set_postfix_str(f"wsi: {slide.name}", refresh=True)

        patcher = SlidePatcher(
            slide,
            mag_0=slide.magnification,
            xywh_only=not args.save_tiles,
            pil=False,
            overwrite=True,
            dst=output_dirs["root"],
            **resolve_patcher_kwargs(args, slide.name),
        )

        if args.save_tiles:
            summary_rows = export_tiles_as_jpeg(
                patcher,
                tiles_dir=output_dirs["tiles"],
                tile_mpp=args.tile_mpp,
            )
            summary_path = output_dirs["tiles"] / "tiles_summary.csv"
            pd.DataFrame(summary_rows).to_csv(
                summary_path,
                mode="a",
                header=not summary_path.exists(),
                index=False,
            )

        if args.save_mask:
            patcher.visualize_tissue_seg(
                size=VISUALIZATION_SIZE,
                save_seg=output_dirs["segmentation"],
                show=False,
            )

        if args.save_grid:
            patcher.visualize_cut(
                size=VISUALIZATION_SIZE,
                save_cut=output_dirs["visualization"],
                show=False,
            )

    progress.close()
    timer.toc()
    LOGGER.info("Tiling done, results saved to %s", output_dirs["root"])

    return output_dirs["root"]


if __name__ == "__main__":
    main()
