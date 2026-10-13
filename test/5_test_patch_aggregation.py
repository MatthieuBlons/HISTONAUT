"""
Test slide-level feature aggregation with `aggragate_tiles_features`.

Reload the features written by `3_test_patch_encoding.py`, aggregate the
patch features of every `<slide>.h5` file of `--src_feat` into a single
slide embedding with `--slide_encoder` and save the result to
`<dst>/features_<slide_encoder>/<slide>.h5`.

Each output file holds three datasets:

- `features`: slide embedding `(D_slide,)`, attributes `encoder`, `name`
  and `dst`.
- `tile_features`: input patch features `(N, D)`, copied with their
  attributes.
- `coords`: patch coordinates `(N, 4)` as `xywh` at level 0, copied with
  their attributes.

Supported slide encoders (`--slide_encoder`):

- Pretrained aggregators: `titan`, `threads`, `prism`, `chief`, `gigapath`,
  `madeleine`. The patch encoder of `--src_feat` must match
  `SLIDE_TO_TILE_ENCODER_MATCH` (for example `titan` needs `conch_v15`).
- Mean pooling: `mean-<patch encoder>`, where `<patch encoder>` must be
  the patch encoder of `--src_feat` (for example `mean-hoptimus1`).

The patch encoder is read from the `--src_feat` directory name
(`features_<patch encoder>`, as written by `TileEncoder`).

Arguments can recive a YAML file (`--config`). Values from the file
replace the parser defaults; arguments typed on the command line still win.

Example
-------
python test/5_test_patch_aggregation.py \\
    --src_feat ./test/features_20x_224px_0px_overlap/features_hoptimus1 \\
    --slide_encoder mean-hoptimus1 \\
    --gpu \\
    --verbose \\
    --clock \\
    --tqdm \\
    --config ./configs/5_test_patch_aggregation.yaml \\
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

# Data libraries
from tqdm import tqdm

# ML libraries
import torch

# Figure libraries

# Project modules
from histonaut.encoder.factory import SLIDE_TO_TILE_ENCODER_MATCH, slide_encoder_factory
from histonaut.encoder.aggregate import aggragate_tiles_features

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


# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------


def build_test_parser() -> ArgumentParser:
    """
    Build the slide-level aggregation test argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured command-line parser.
    """
    parser = ArgumentParser(
        description="Aggregate patch encodings at the slide-level.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # I/O

    parser.add_argument(
        "--src_feat",
        type=str,
        default=None,
        help="Directory holding the `<slide>.h5` features files. Required.",
    )

    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="Directory to store outputs. Defaults to the parent of `src_feat`.",
    )

    # Slide-Encoder

    parser.add_argument(
        "--slide_encoder",
        type=str,
        required=True,
        help="Slide encoder to use for tile feature aggregation",
    )

    parser.add_argument(
        "--model_summary",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Verbosity of the encoder `torchinfo` summary (0: none).",
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
        Validated configuration, with `src_feat` as a `pathlib.Path`.

    Raises
    ------
    KeyError
        If the YAML config holds a key that is not a parser argument.

    ValueError
        If `src_feat` is missing.
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

    if args.src_feat is None:
        raise ValueError("`src_feat` is required (command line or config).")
    args.src_feat = Path(args.src_feat)

    return args


def build_test_directory(
    args: Namespace,
    encoder_name: str,
) -> dict[str, Path]:
    """
    Create the output directories.

    Parameters
    ----------
    args : argparse.Namespace
        Test configuration.

    encoder_name : str
        Slide encoder name (`enc_name`), used to name the
        `features_<encoder_name>` directory.

    Returns
    -------
    dict of str to pathlib.Path
        Directories: `root` (`--dst`, or the parent of `--src_feat`),
        `features` (input patch features, `--src_feat`) and
        `features_aggregated` (output slide embeddings).
    """
    # Default to the encoding run directory holding `src_feat`
    src = Path(args.src_feat)
    root = src.parent if args.dst is None else Path(args.dst)

    output_directories = {
        "root": root,
        "features": src,
        "features_aggregated": root / f"features_{encoder_name}",
    }

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
    List the features files to process.

    Parameters
    ----------
    args : argparse.Namespace
        Test configuration.

    features_dir : pathlib.Path
        Output directory holding the `<slide>.h5` slide embeddings already
        computed. Their slides are skipped unless `--overwrite` is set.

    Returns
    -------
    list of pathlib.Path
        Patch features files to process, sorted.

    Raises
    ------
    FileNotFoundError
        If no file with extension `h5` is found in `src`.
    """
    src = Path(args.src)
    slides = sorted(src.glob(f"**/*.h5"))
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
            f"No slide with extension `ext`=h5 found in `src`={src}."
        )

    # Skip slides already encoded, unless asked to overwrite
    if not args.overwrite:
        processed_names = {path.stem for path in features_dir.glob("*.h5")}
        slides = [slide for slide in slides if slide.stem not in processed_names]
        LOGGER.info("%s slide(s) already processed are skipped", len(processed_names))

    LOGGER.info("%s slide(s) will be processed", len(slides))
    return slides


# ----------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------


def resolve_encoder(
    args: Namespace,
    slide_encoder: str,
    device: str,
) -> torch.nn.Module:
    """
    Check the slide / patch encoder match and build the slide encoder.

    The patch encoder name is read from the `--src_feat` directory name
    (`features_<patch encoder>`, as written by `TileEncoder`).

    Parameters
    ----------
    args : argparse.Namespace
        Test configuration (`src_feat`, `model_summary`).

    slide_encoder : str
        Slide encoder name, see `slide_encoder_factory`.

    device : str
        PyTorch device the model is moved to.

    Returns
    -------
    torch.nn.Module
        Slide encoder (`BaseSlideEncoder`) in eval mode, on `device`.

    Raises
    ------
    ValueError
        If `slide_encoder` is `mean-<name>` with `<name>` different from the
        patch encoder, or if it is listed in `SLIDE_TO_TILE_ENCODER_MATCH`
        with another patch encoder.
    """
    encoders = {}
    encoders["slide"] = slide_encoder

    src = Path(args.src_feat)
    tile_encoder = src.name.removeprefix("features_")
    encoders["tile"] = tile_encoder
    if slide_encoder.startswith("mean-"):
        if slide_encoder.removeprefix("mean-") != tile_encoder:
            raise ValueError(
                "slide aggregator: %s does not match tile encoder: %s",
                slide_encoder,
                tile_encoder,
            )

    if slide_encoder in SLIDE_TO_TILE_ENCODER_MATCH:
        if SLIDE_TO_TILE_ENCODER_MATCH[slide_encoder] != tile_encoder:
            raise ValueError(
                "slide aggregator: %s does not match tile encoder: %s",
                slide_encoder,
                tile_encoder,
            )

    model = slide_encoder_factory(slide_encoder)
    model.to(device)
    model.eval()
    if args.model_summary:
        model.print_summary(verbose=args.model_summary)

    return model


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


# could use a dataloader
def main(
    known_args: list[str] | None = None,
) -> Path:
    """
    Run the slide-level aggregation test.

    Build the slide encoder, then aggregate the patch features of every
    `<slide>.h5` file of `--src_feat` into
    `<root>/features_<slide_encoder>/<slide>.h5`.

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

    slides = resolve_input_slides(args, output_dirs["features_aggregated"])

    timer = timetracker(verbose=args.clock)
    progress = tqdm(
        slides,
        desc=f"Encoding {args.patch_size}x{args.patch_size}px at {args.mag_target}x",
        unit="wsi",
        disable=not args.tqdm,
    )

    timer.tic()
    for slide_path in progress:
        slide_name = slide_path.stem
        progress.set_postfix_str(f"slide: {slide_name}", refresh=True)
        _ = aggragate_tiles_features(
            features_path=output_dirs["features"],
            slide_encoder=model,
            device=device,
            dst=output_dirs["features_aggregated"],
        )

    progress.close()
    timer.toc()
    LOGGER.info("Encoding done, results saved to %s", output_dirs["root"])

    return output_dirs["root"]


if __name__ == "__main__":
    main()
