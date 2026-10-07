"""
Test patch filtering with `SlidePatcher`.

Reload the coords written by `1_test_patchify.py`, filter them with tile
selection (`hog`) and colour filtering, then save the kept coords and, for
each encoder in `--list_encoder`, the matching rows of its features file.

Example
-------
python test/2_test_patch_filter.py \\
    --src_patch ./test/features_20x_512px_0px_overlap/patches \\
    --src_wsi /cluster/CBIO/home/mblons/py-packages/HISTONAUT/data/HE/TCGA-COAD \\
    --plot_invalid_patch 36 \\
    --update_viz \\
    --verbose \\
    --clock \\
    --tqdm \\
    --config .configs/2_test_patch_filter.yaml \\
"""

from __future__ import annotations

# General libraries
import logging
import warnings
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
    Namespace,
)
from pathlib import Path

# Data libraries
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Project modules
from histonaut.patcher.cut import SlidePatcher
from histonaut.patcher.io import read_h5_coords, read_h5_features, save_h5
from histonaut.slide.reader import get_slide_reader

# Test helpers
from logger import (
    configure_logging,
    load_yaml_config,
    log_configuration,
    timetracker,
    resolve_device,
)

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


def build_test_parser() -> ArgumentParser:
    """
    Build the filter test argument parser.

    Filter arguments are named after the `SlidePatcher` keywords so that a
    YAML config can use the same names.

    Returns
    -------
    argparse.ArgumentParser
        Configured command-line parser.
    """
    parser = ArgumentParser(
        description="Filter Patches with SlidePatcher.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # I/O

    parser.add_argument(
        "--src_patch",
        type=str,
        default=None,
        help="Directory containing the coords h5 files to filter. Required.",
    )
    parser.add_argument(
        "--src_wsi",
        type=str,
        default=None,
        help="Directory containing the original slides. Required.",
    )
    parser.add_argument(
        "--list_patch",
        type=str,
        nargs="*",
        default=None,
        help="List of patch files to process. When omitted, process all patches found in 'src_patch'.",
    )
    parser.add_argument(
        "--ext_wsi",
        type=str,
        default="svs",
        help="Slide extension, without the dot.",
    )
    # could provide a list of encoders -> no need to encode again after updating the patches
    parser.add_argument(
        "--list_encoder",
        nargs="*",
        type=str,
        default=None,
        help=f"Name of the encoder used to generate the features hdf5 files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Reprocess slides whose coords file already exists.",
    )
    parser.add_argument(
        "--update_viz",
        action="store_true",
        default=False,
        help="Redraw the patch visualization after tile selection.",
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
        "--dst",
        type=str,
        default=None,
        help="Directory where the `<run>_filtered` output directory is created. "
        "When omitted, use the parent of `src_patch`.",
    )

    # Runtime

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "PyTorch device passed to each run, for example cpu, cuda, "
            "cuda:0, or mps. When omitted, the device is inferred."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
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

    # Filtering

    parser.add_argument(
        "--selection_strategy",
        type=str,
        default="hog",
        choices=list(SlidePatcher.tile_selection_mapping.keys()),
        help="Tile selection strategy.",
    )
    parser.add_argument(
        "--hog_thresh",
        type=float,
        default=0.5,
        help="HOG threshold used by the `hog` selection strategy.",
    )
    parser.add_argument(
        "--stat_fct",
        type=str,
        default="channel_std",
        help="Statistic computed for colour filtering.",
    )
    parser.add_argument(
        "--color_thresh",
        type=float,
        default=3.5,
        help="Colour threshold compared to the `q` quantile of the statistic.",
    )
    parser.add_argument(
        "-q",
        type=float,
        default=0.2,
        help="Quantile of the statistic compared to the colour threshold.",
    )
    parser.add_argument(
        "--local_average",
        action=BooleanOptionalAction,
        default=True,
        help="Compute a local average before the statistic.",
    )
    parser.add_argument(
        "--filter_size",
        type=int,
        default=16,
        help="Filter size of the local average.",
    )
    parser.add_argument(
        "--remove_background",
        action=BooleanOptionalAction,
        default=True,
        help="Remove the background before computing the statistic.",
    )
    parser.add_argument(
        "--background_thresh",
        type=int,
        default=240,
        help="Colour threshold above which a pixel is background.",
    )
    parser.add_argument(
        "--background_average",
        action=BooleanOptionalAction,
        default=False,
        help="Detect the background on the average patch instead of the pixels.",
    )
    parser.add_argument(
        "--filter_all",
        action=BooleanOptionalAction,
        default=False,
        help="Require the colour criterion on all channels instead of at least one.",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        help="Function aggregating the channel std over the pixels.",
    )
    parser.add_argument(
        "--plot_invalid_patch",
        type=int,
        default=None,
        help="Number of invalid patches to sample and plot. 0 plots all of them; "
        "omit to skip the plot.",
    )
    parser.add_argument(
        "--invalid_on",
        type=str,
        default="all",
        choices=["all", "color", "quality"],
        help="Rejection to plot: tile selection and colour filtering (`all`), "
        "colour filtering only (`color`) or tile selection only (`quality`).",
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
        If `src_patch` or `src_wsi` is missing.
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

    for required in ("src_patch", "src_wsi"):
        if getattr(args, required) is None:
            raise ValueError(f"`{required}` is required (command line or config).")
    args.src_patch = Path(args.src_patch)
    args.src_wsi = Path(args.src_wsi)

    if args.device is not None and args.gpu:
        parser.error("Use either --device or --gpu, not both.")

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
        Create a `<run>_filtered` subdirectory, where `<run>` is the patching
        run directory holding `src_patch`.

    Returns
    -------
    dict of str to pathlib.Path
        Output directories: always `root` and `patches`, plus `visualization`
        with `--update_viz` and `tiles` with `--save_tiles`.
    """
    src = Path(args.src_patch)
    run_dir = src.parent
    if args.dst is None:
        root = run_dir.parent
    else:
        root = Path(args.dst)
    if auto_name:
        root = root / f"{run_dir.name}_filtered"

    # SlidePatcher writes its coords under `<root>/patches`
    output_directories = {
        "root": root,
        "patches": root / "patches",
    }

    if args.update_viz:
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


def resolve_input_files(
    args: Namespace,
    filtered_patches_dir: Path,
) -> list[Path]:
    """
    List the files to process.

    Parameters
    ----------
    args : argparse.Namespace
        Test configuration.

    filtered_patches_dir : pathlib.Path
        Directory holding the `<slide>.h5` files already filtered.

    Returns
    -------
    list of pathlib.Path
        Coords h5 files to process, sorted.

    Raises
    ------
    FileNotFoundError
        If no h5 file is found in `src_patch`, or none matches `list_patch`.
    """
    src = Path(args.src_patch)
    files = sorted(src.glob(f"*.h5"))
    LOGGER.info(
        "%s patch file(s) with extension .h5 found in %s",
        len(files),
        src,
    )

    if args.list_patch is not None:
        requested_names = {Path(patch).name for patch in args.list_patch}
        files = [file for file in files if file.name in requested_names]
        LOGGER.info("%s patch file(s) kept from `list_patch`", len(files))

    if not files:
        raise FileNotFoundError(
            f"No h5 file found in `src_patch`={src} "
            f"(or none matching `list_patch`={args.list_patch})."
        )

    # Skip files already filtered, unless asked to overwrite
    if not args.overwrite:
        processed_names = {path.stem for path in filtered_patches_dir.glob("*.h5")}
        nb_files = len(files)
        files = [file for file in files if file.stem not in processed_names]
        LOGGER.info(
            "%s patch file(s) already filtered are skipped",
            nb_files - len(files),
        )

    LOGGER.info("%s patch file(s) will be filtered", len(files))
    return files


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


def tile_selection_from_wsi_name(
    slide_path: str,
    patch_path: str,
    output_dirs: str,
    encoder: list = [],
    selection_strategy: str = "hog",
    hog_thresh: float = 0.5,
    color_thresh: int | tuple[int, int, int] = 3.5,
    stat_fct: str = "channel_std",
    local_average: bool = True,
    filter_size: int | tuple[int, int] = 16,
    remove_background: bool = True,
    background_thresh: int = 240,
    background_average: bool = False,
    q: float = 0.2,
    filter_all: bool = False,
    aggregate: str = "mean",
    verbose: bool = True,
    filter_patch: bool = True,
    update_viz: bool = False,
    save_tiles: bool = False,
    tile_mpp: float | None = None,
    plot_invalid_patch: int | None = None,
    invalid_on: str = "all",
    device="cpu",
):
    """
    Run the tile selection pipeline for a given WSI, specified by its name.

    Args:
        patch_path:             str, path to the unfiltered generated patches hdf5 files.
        slide_path:             str, path to the WSI files.
        output_dirs:            dict, path to the directory where to save the filtered patches.
        features_path:          str, path to the directory containing the patches features hdf5 files, default=None.
        selection_strategy:     str, name of the strategy to use to select patches. Default='hog'.
        hog_thresh:             float, threshold for the HOG tile selection method. Default=0.5.
        color_thresh:           int or tuple of int, threshold for the color filtering, default=3.5.
        stat_fct:               str, statistic to compute for tile color filtering. Default='channel_std'.
        local_average:          bool, whether to compute the local average before computing the statistic. Default=True.
        filter_size:            int or tuple of int, size of the filter for computing the local average, default=16.
        remove_background:      bool, whether to remove the background before computing the statistic. Default=True.
        background_thresh:      int or tuple of int, color threshold for the detecting background, default=240.
        background_average:     bool, whether to compute detect the background on the average pixels or the pixels,
                                default=False.
        q:                      float, quantile to compute to compare to the color threshold, default=0.2.
        filter_all:             bool, whether to filter to compute the criterion for all channel or at least one,
                                default=False.
        aggregate:              str, aggregation function for aggregating the channel std over the pixels, default='mean'.
        verbose:                bool, whether to print information about the tile selection process. Default=True.
        filter_patch:           bool, whether to filter the patches during tile selection. Default=True.
        update_viz:             bool, whether to update the visualization of the patches after tile selection. Default=False.
        plot_invalid_patch:     bool, whether to plot some invalid patches. Default=True.
        invalid_on:             str, whether plot invalid patches from the whole selection process or the tile selection
                                or the color filtering, default='all'.
        n_invalid_patch:        int, number of invalid patches to sample and plot. Default=36.

    """
    # Creating slide reader and slide
    reader = get_slide_reader(slide_path)
    slide = reader(slide_path)
    slide_name = slide.name
    if slide_name != patch_path.stem:
        raise ValueError(
            f"slide at: {slide_path} does not match patch file at: {patch_path}"
        )

    # Loading h5 features file
    attrs, coords = read_h5_coords(patch_path)

    # Creating slide patcher loading the already computed coordinates
    # if not in attrs -> get SlidePatcher default values
    patcher = SlidePatcher(
        slide=slide,
        mag_0=attrs.get("magnification", None),
        mag_target=attrs.get("target_magnification", 20),
        patch_size=attrs.get("target_patch_size", 256),
        overlap=attrs.get("target_overlap", 0),
        mask_strategy=attrs.get("mask", "otsu"),
        mask_tolerance=attrs.get("tissue_thr", 0.5),
        custom_xywh=coords,
        selection_strategy=selection_strategy,
        hog_thresh=hog_thresh,
        stat_fct=stat_fct,
        color_thresh=color_thresh,
        q=q,
        filter_size=filter_size,
        filter_all=filter_all,
        remove_background=remove_background,
        background_thresh=background_thresh,
        background_average=background_average,
        local_average=local_average,
        aggregate=aggregate,
        xywh_only=not save_tiles,
        lazy=True,
        dst=None,
        device=device,
    )

    nb_src_patches = patcher.nb_valid_patches
    # Filtering the patches using the selection strategy
    if verbose:
        print("------\t\t Slide {} \t\t------".format(slide_name))
        print("Filtering tiles...")

    nb_selected_patches, selected_patches, idx_selected = patcher.select_patches(
        patcher.valid_patches, return_idx=True
    )

    if verbose:
        print(
            "Number of patches selected: {}/{}.".format(
                nb_selected_patches, nb_src_patches
            )
        )

    # From here on, the patcher only holds the selected patches
    patcher.nb_valid_patches = nb_selected_patches
    patcher.valid_patches = selected_patches

    # Sample and plot some invalid patches (None: no plot, 0: all of them)
    if plot_invalid_patch is not None:
        LOGGER.debug("Plotting invalid patches of slide %s", slide_name)
        invalid_patches = np.delete(coords, idx_selected, axis=0)
        n_invalid_patch = plot_invalid_patch or len(invalid_patches)
        if invalid_on == "all":
            if len(invalid_patches) > 0:
                patcher.sample_plot_patches(
                    patches=invalid_patches,
                    n_patches=n_invalid_patch,
                    save_path=output_dirs["root"] / "invalid_patches",
                    show=False,
                )
        else:
            patcher.sample_plot_invalid_patches(
                n_patches=n_invalid_patch,
                verbose=verbose,
                save_path=output_dirs["root"] / "invalid_patches",
                invalid_on=invalid_on,
                show=False,
            )

    # Updating patches
    if filter_patch:
        if verbose:
            print("Saving filtered patches...")
        save_h5(
            save_path=output_dirs["patches"] / f"{slide_name}.h5",
            assets={"coords": np.array(selected_patches)},
            attributes={"coords": attrs},
            mode="w",
        )

    # Visualizing patches on tissue
    if update_viz:
        visu_dir = output_dirs["visualization"]
        _ = patcher.visualize_cut(size=(1024, 1024), save_cut=visu_dir, show=False)

    # Filtering the features computed on the unfiltered coords, one file per encoder
    for model in encoder or []:
        source_features_path = (
            patch_path.parent.parent / f"features_{model}" / f"{slide_name}.h5"
        )
        if not source_features_path.exists():
            LOGGER.warning(
                "No feature file for slide %s with encoder %s: %s",
                slide_name,
                model,
                source_features_path,
            )
            continue

        feats_attrs, feats = read_h5_features(source_features_path)
        if len(feats) != nb_src_patches:
            raise ValueError(
                f"`{source_features_path}` holds {len(feats)} features, expected "
                f"{nb_src_patches} (one per patch in `{patch_path}`)."
            )

        output_features_dir = output_dirs["root"] / f"features_{model}"
        output_features_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        LOGGER.debug("Saving filtered %s features of slide %s", model, slide_name)
        save_h5(
            save_path=output_features_dir / f"{slide_name}.h5",
            assets={
                "features": feats[idx_selected],
                "coords": np.array(selected_patches),
            },
            attributes={
                "features": feats_attrs,
                "coords": attrs,
            },
            mode="w",
        )

    # Save tiles
    if save_tiles:
        summary_rows = export_tiles_as_jpeg(
            patcher,
            tiles_dir=output_dirs["tiles"],
            tile_mpp=tile_mpp,
        )
        summary_path = output_dirs["tiles"] / "tiles_summary.csv"
        pd.DataFrame(summary_rows).to_csv(
            summary_path,
            mode="a",
            header=not summary_path.exists(),
            index=False,
        )

    return selected_patches, idx_selected, nb_selected_patches, nb_src_patches


def main(
    known_args: list[str] | None = None,
) -> Path:
    """
    Test patch filtering.

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

    output_dirs = build_test_directory(args)
    patches = resolve_input_files(args, output_dirs["patches"])

    timer = timetracker(verbose=args.clock)
    progress = tqdm(
        patches,
        desc=f"Filtering",
        unit="wsi",
        disable=not args.tqdm,
    )

    for patch_path in progress:
        name = patch_path.stem  # no suffix
        slide_path = args.src_wsi / f"{name}.{args.ext_wsi}"
        # Running the tile selection pipeline
        timer.tic()
        _, _, nselected, ninit = tile_selection_from_wsi_name(
            slide_path=slide_path,
            patch_path=patch_path,
            encoder=args.list_encoder,
            output_dirs=output_dirs,
            selection_strategy=args.selection_strategy,
            hog_thresh=args.hog_thresh,
            color_thresh=args.color_thresh,
            filter_all=args.filter_all,
            stat_fct=args.stat_fct,
            remove_background=args.remove_background,
            background_thresh=args.background_thresh,
            background_average=args.background_average,
            local_average=args.local_average,
            filter_size=args.filter_size,
            q=args.q,
            aggregate=args.aggregate,
            verbose=args.verbose,
            filter_patch=True,
            update_viz=args.update_viz,
            save_tiles=args.save_tiles,
            tile_mpp=args.tile_mpp,
            plot_invalid_patch=args.plot_invalid_patch,
            invalid_on=args.invalid_on,
            device=device,
        )
        timer.toc()
        progress.set_postfix_str(
            f"{name}, patches selected: {nselected}/{ninit}", refresh=True
        )
    progress.close()
    print(f"Done! Results saved in: ", output_dirs["root"])

    return output_dirs["root"]


if __name__ == "__main__":
    main()
