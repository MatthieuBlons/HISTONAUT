"""
Test patch encoding with `TileEncoder`.

For each slide found in `--src`, cut the slide into patches with
`SlidePatcher` (or reuse coordinates from `--custom_xywh`), encode every
patch with `--tile_encoder` and save the features to
`<dst>/features_<tile_encoder>/<slide>.h5`.

Optional outputs (written under the same run directory):

- `--save_mask`: tissue segmentation thumbnails (`segmentation/`). Not
  available with `--custom_xywh` (no tissue mask is computed).
- `--save_grid`: patch grid drawn over the slide (`visualization/`).
- `--save_tiles`: every patch as a jpeg, resized to `--tile_mpp`
  (`tiles_<mpp>mpp/`), plus a csv summary per run.
- `--save_rep`: patch embeddings drawn over the slide, as PCA colours
  (`rgb`) or Leiden clusters (`leiden`) (`visualization_<encoder>/`).

Arguments can recive a YAML file (`--config`). Values from the file
replace the parser defaults; arguments typed on the command line still win.

Example
-------
python test/3_test_patch_encoding.py \\
    --src HISTONAUT/data/HE/ \\
    --dst . \\
    --gpu \\
    --verbose \\
    --clock \\
    --tqdm \\
    --config ./configs/3_test_patch_encoding.yaml \\
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

# Deep learning libraries
import torch

# Project modules
from histonaut.encoder.encode import TileEncoder
from histonaut.encoder.factory import patch_encoder_factory
from histonaut.patcher.cut import SlidePatcher
from histonaut.patcher.io import read_h5_coords
from histonaut.slide.reader import get_slide_reader

# Test helpers
from logger import (
    configure_logging,
    load_yaml_config,
    log_configuration,
    resolve_device,
    timetracker,
)

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


VISUALIZATION_SIZE = (1024, 1024)


# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------


def build_test_parser() -> ArgumentParser:
    """
    Build the encoding test argument parser.

    Patcher and encoder arguments are named after the `SlidePatcher` and
    `TileEncoder` keywords so that a YAML config can use the same names.

    Returns
    -------
    argparse.ArgumentParser
        Configured command-line parser.
    """
    parser = ArgumentParser(
        description="Encode WSI patches with TileEncoder.",
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
        "--custom_xywh",
        type=str,
        default=None,
        help="Directory holding `<slide>.h5` coords to reuse as `custom_xywh`.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Reprocess slides whose features file already exists.",
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
    parser.add_argument(
        "--save_rep",
        type=str,
        choices=["rgb", "leiden"],
        default=None,
        help="Save the patch embeddings visualization as jpeg.",
    )

    # Runtime
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "PyTorch device, for example cpu, cuda, cuda:0, or mps. "
            "When omitted, the device is inferred."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help=(
            "Request a CUDA device when --device is not explicitly set. "
            "Falls back to CPU if CUDA is unavailable."
        ),
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="CUDA device index used with --gpu.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers used for encoding.",
    )
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

    # Encoder
    parser.add_argument(
        "--tile_encoder",
        type=str,
        default="hoptimus1",
        help="Patch encoder name, as registered in `patch_encoder_factory`.",
    )
    parser.add_argument(
        "--batch_max",
        type=int,
        default=32,
        help="Batch size used for encoding.",
    )
    parser.add_argument(
        "--model_summary",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Verbosity of the encoder `torchinfo` summary (0: none).",
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
        If `src` or `dst` is missing, or if both `device` and `gpu` are set.
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

    if args.device is not None and args.gpu:
        raise ValueError("Use either `device` or `gpu`, not both.")

    return args


def build_test_directory(
    args: Namespace,
    encoder_name: str,
    auto_name: bool = True,
) -> dict[str, Path]:
    """
    Create the output directories.

    Parameters
    ----------
    args : argparse.Namespace
        Test configuration.

    encoder_name : str
        Encoder name (`enc_name`), used by `TileEncoder` to name its
        `features_<encoder_name>` directory.

    auto_name : bool, default=True
        Create a run subdirectory named after the patching parameters.

    Returns
    -------
    dict of str to pathlib.Path
        Output directories: always `root`, `patches` and `features`, plus
        `segmentation`, `visualization`, `tiles` and `representation` when
        the matching `--save_*` option is set. `segmentation` is never
        created with `--custom_xywh`.
    """
    root = Path(args.dst)

    if auto_name:
        if args.custom_xywh is None: 
            root = root / (
                f"features_{args.mag_target}x_{args.patch_size}px_{args.overlap}px_overlap"
            )
        else:
            root = Path(args.custom_xywh).parent

    # SlidePatcher writes its coords under `<root>/patches`, TileEncoder its
    # features under `<root>/features_<encoder_name>`
    output_directories = {
        "root": root,
        "patches": root / "patches",
        "features": root / f"features_{encoder_name}",
    }
    if args.save_mask:
        if args.custom_xywh is None:
            output_directories["segmentation"] = root / "segmentation"
        else:
            LOGGER.warning("`save_mask` is ignored with `custom_xywh` (no mask).")
    if args.save_grid:
        output_directories["visualization"] = root / "visualization"
    if args.save_tiles:
        tile_mpp = "auto" if args.tile_mpp is None else args.tile_mpp
        output_directories["tiles"] = root / f"tiles_{tile_mpp}mpp"
    if args.save_rep is not None:
        output_directories["representation"] = root / f"visualization_{encoder_name}"

    for directory in output_directories.values():
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return output_directories


def resolve_input_slides(
    args: Namespace,
    features_dir: Path,
) -> list[Path]:
    """
    List the slides to process.

    Parameters
    ----------
    args : argparse.Namespace
        Test configuration.

    features_dir : pathlib.Path
        Directory holding the `<slide>.h5` feature files already computed.

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

    # Skip slides already encoded, unless asked to overwrite
    if not args.overwrite:
        processed_names = {path.stem for path in features_dir.glob("*.h5")}
        slides = [slide for slide in slides if slide.stem not in processed_names]
        LOGGER.info("%s slide(s) already processed are skipped", len(processed_names))

    LOGGER.info("%s slide(s) will be processed", len(slides))
    return slides


# ----------------------------------------------------------------------------
# Patching
# ----------------------------------------------------------------------------


def resolve_custom_coords(
    custom_coords_dir: str | Path,
    slide_name: str,
) -> dict[str, Any]:
    """
    Read reusable coordinates for one slide as `SlidePatcher` keywords.

    Parameters
    ----------
    custom_coords_dir : str or pathlib.Path
        Directory holding `<slide>.h5` coords, as written by `SlidePatcher`.

    slide_name : str
        Slide name, without extension.

    Returns
    -------
    dict of str to Any
        `custom_xywh` (shape `(N, 4)` xywh at level 0) and the keywords that
        still matter without tissue masking: `mag_target`,
        `pixel_size_target`, `patch_size` and `overlap`.

    Raises
    ------
    FileNotFoundError
        If `<custom_coords_dir>/<slide_name>.h5` does not exist.
    """
    coords_path = Path(custom_coords_dir) / f"{slide_name}.h5"
    if not coords_path.is_file():
        raise FileNotFoundError(
            f"No coords for slide {slide_name!r} in `custom_xywh`: {coords_path}."
        )

    attrs, coords = read_h5_coords(coords_path)
    LOGGER.debug(
        "%s custom coords read from %s",
        len(coords),
        coords_path,
    )

    return {
        "custom_xywh": coords,
        "mag_target": attrs["target_magnification"],
        "pixel_size_target": attrs["target_mpp"],
        "patch_size": attrs["target_patch_size"],
        "overlap": attrs["target_overlap"],
    }


def resolve_patcher_kwargs(
    args: Namespace,
    slide_name: str,
) -> dict[str, Any]:
    """
    Gather the `SlidePatcher` keywords for one slide.

    With `--custom_xywh`, the keywords returned by `resolve_custom_coords`
    take precedence over `args`.

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

    if args.custom_xywh is not None:
        patcher_kwargs.update(
            resolve_custom_coords(
                args.custom_xywh,
                slide_name,
            )
        )

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
# Encoding
# ----------------------------------------------------------------------------


def resolve_encoder(
    args: Namespace,
    device: str,
) -> torch.nn.Module:
    """
    Build the patch encoder in eval mode on `device`.

    Parameters
    ----------
    args : argparse.Namespace
        Test configuration.

    device : str
        PyTorch device.

    Returns
    -------
    torch.nn.Module
        Patch encoder from `patch_encoder_factory`.
    """
    model = patch_encoder_factory(args.tile_encoder)
    model.to(device)
    model.eval()
    if args.model_summary:
        model.print_summary(verbose=args.model_summary)

    return model


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main(
    known_args: list[str] | None = None,
) -> Path:
    """
    Test WSI patch encoding.

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

    device = resolve_device(args)
    model = resolve_encoder(args, device)
    output_dirs = build_test_directory(args, encoder_name=model.enc_name)
    slides = resolve_input_slides(args, output_dirs["features"])

    timer = timetracker(verbose=args.clock)
    progress = tqdm(
        slides,
        desc=f"Encoding {args.patch_size}x{args.patch_size}px at {args.mag_target}x",
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
        progress.set_postfix_str(f"slide: {slide.name}", refresh=True)

        patcher = SlidePatcher(
            slide,
            mag_0=slide.magnification,
            xywh_only=not args.save_tiles,
            pil=False,
            overwrite=True,
            dst=output_dirs["root"],
            **resolve_patcher_kwargs(args, slide.name),
        )

        if "tiles" in output_dirs:
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

        if "segmentation" in output_dirs:
            patcher.visualize_tissue_seg(
                size=VISUALIZATION_SIZE,
                save_seg=output_dirs["segmentation"],
                show=False,
            )

        if "visualization" in output_dirs:
            patcher.visualize_cut(
                size=VISUALIZATION_SIZE,
                save_cut=output_dirs["visualization"],
                show=False,
            )

        # Encodes at construction (lazy=True) and writes
        # `<root>/features_<enc_name>/<slide>.h5`
        encoder = TileEncoder(
            slide,
            tile_encoder=model,
            coords_path=patcher.patch_path,
            device=device,
            num_workers=args.num_workers,
            batch_max=args.batch_max,
            feat_only=False,
            dst=output_dirs["root"],
            verbose=args.tqdm,
        )

        if "representation" in output_dirs:
            encoder.visualize_embeddings(
                method=args.save_rep,
                save_vis=output_dirs["representation"],
                show=False,
            )

    progress.close()
    timer.toc()
    LOGGER.info("Encoding done, results saved to %s", output_dirs["root"])

    return output_dirs["root"]


if __name__ == "__main__":
    main()
