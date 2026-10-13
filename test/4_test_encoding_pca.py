"""
Test PCA on patch encodings with `fit_pca`.

Reload the features written by `3_test_patch_encoding.py`, fit an
`IncrementalPCA` over every `<slide>.h5` file of `--src_feat`, project each
slide on the principal components and save the result to
`<dst>/features_<encoder>_pca/<slide>.h5` (features and coords).

Optional outputs (written under the same run directory):

- `--save_variance`: explained variance ratio per component, as a csv
  (`pca_explained_variance.csv`).
- `--save_rep`: the first three components drawn over the slide as RGB
  colours (`rgb`) (`visualization_<encoder>_pca/`). Needs `--src_wsi`.

Arguments can recive a YAML file (`--config`). Values from the file
replace the parser defaults; arguments typed on the command line still win.

Example
-------
python test/4_test_encoding_pca.py \\
    --src_feat ./test/features_20x_224px_0px_overlap/features_hoptimus1 \\
    --src_wsi /cluster/CBIO/home/mblons/py-packages/HISTONAUT/data/HE/TCGA-COAD \\
    --verbose \\
    --clock \\
    --tqdm \\
    --config ./configs/4_test_encoding_pca.yaml \\
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
import numpy as np
import pandas as pd
from tqdm import tqdm

# ML libraries
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler

# Figure libraries
import matplotlib.pyplot as plt

# Project modules
from histonaut.encoder.pca import fit_pca
from histonaut.patcher.io import read_h5_coords, read_h5_features, save_h5
from histonaut.slide.draw import visualise_tile_rgb
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


VISUALIZATION_FIGSIZE = (10, 10)


# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------


def build_test_parser() -> ArgumentParser:
    """
    Build the PCA test argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured command-line parser.
    """
    parser = ArgumentParser(
        description="Fit and apply a PCA on patch encodings.",
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
        "--src_wsi",
        type=str,
        default=None,
        help="Directory containing the original slides. Required with --save_rep.",
    )
    parser.add_argument(
        "--ext_wsi",
        type=str,
        default="svs",
        help="Slide extension, without the dot.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="Directory to store outputs. Defaults to the parent of `src_feat`.",
    )
    parser.add_argument(
        "--save_variance",
        action="store_true",
        default=False,
        help="Save the explained variance ratio per component as csv.",
    )
    parser.add_argument(
        "--save_rep",
        type=str,
        choices=["rgb"],
        default=None,
        help="Save the PCA embeddings visualization as jpeg.",
    )

    # Visualization
    parser.add_argument(
        "--level_to_view",
        type=int,
        default=4,
        help="Pyramid level of the slide drawn under the embeddings.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Opacity of the embeddings overlay.",
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
        If `src_feat` is missing, or if `save_rep` is set without `src_wsi`.
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

    if args.save_rep is not None and args.src_wsi is None:
        raise ValueError("`src_wsi` is required with `save_rep` (slides to draw on).")
    if args.src_wsi is not None:
        args.src_wsi = Path(args.src_wsi)

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
        Encoder name, used to name the `features_<encoder_name>_pca`
        directory.

    Returns
    -------
    dict of str to pathlib.Path
        Output directories: always `root` and `features_pca`, plus
        `representation` with `--save_rep`.
    """
    # Default to the encoding run directory holding `src_feat`
    root = args.src_feat.parent if args.dst is None else Path(args.dst)

    output_directories = {
        "root": root,
        "features_pca": root / f"features_{encoder_name}_pca",
    }
    if args.save_rep is not None:
        output_directories["representation"] = (
            root / f"visualization_{encoder_name}_pca"
        )

    for directory in output_directories.values():
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return output_directories


def resolve_encoder_name(
    features_dir: Path,
) -> str:
    """
    Infer the encoder name from a features directory.

    Parameters
    ----------
    features_dir : pathlib.Path
        Directory named `features_<encoder_name>`, as written by `TileEncoder`.

    Returns
    -------
    str
        Encoder name (the directory name without the `features_` prefix).
    """
    return features_dir.name.removeprefix("features_")


# ----------------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------------


def save_explained_variance(
    ipca: IncrementalPCA,
    variance_path: Path,
) -> pd.DataFrame:
    """
    Save the explained variance ratio of each component as csv.

    Parameters
    ----------
    ipca : sklearn.decomposition.IncrementalPCA
        Fitted PCA.

    variance_path : pathlib.Path
        Output csv file.

    Returns
    -------
    pandas.DataFrame
        One row per component: `component` (1-based),
        `explained_variance_ratio` and `cumulative_explained_variance_ratio`.
    """
    variance_df = pd.DataFrame(
        {
            "component": np.arange(1, ipca.n_components_ + 1),
            "explained_variance_ratio": ipca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                ipca.explained_variance_ratio_
            ),
        }
    )
    variance_df.to_csv(
        variance_path,
        index=False,
    )
    LOGGER.info("Explained variance saved to %s", variance_path)

    return variance_df


def apply_pca(
    ipca: IncrementalPCA,
    features_path: Path,
    features_pca_dir: Path,
    encoder_name: str,
) -> tuple[str, Path]:
    """
    Project the patch features of one slide and save them as h5.

    Parameters
    ----------
    ipca : sklearn.decomposition.IncrementalPCA
        Fitted PCA.

    features_path : pathlib.Path
        `<slide>.h5` file holding the `features` and `coords` datasets.

    features_pca_dir : pathlib.Path
        Output directory.

    encoder_name : str
        Encoder name, stored in the `features` attributes.

    Returns
    -------
    tuple of (str, pathlib.Path)
        Slide name and path of the saved `<slide>.h5` PCA features file.
    """
    features_attrs, features = read_h5_features(features_path)
    coords_attrs, coords = read_h5_coords(features_path)
    slide_name = features_attrs["name"]

    features_pca_path = features_pca_dir / f"{slide_name}.h5"
    save_h5(
        features_pca_path,
        assets={
            "features": ipca.transform(features),
            "coords": coords,
        },
        attributes={
            "features": {
                "encoder": encoder_name,
                "name": slide_name,
                "dst": str(features_pca_path),
            },
            "coords": coords_attrs,
        },
        mode="w",
    )

    return slide_name, features_pca_path


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------


def draw_pca_rgb(
    slide_path: Path,
    features_pca_path: Path,
    representation_dir: Path,
    encoder_name: str,
    level_to_view: int = 4,
    alpha: float = 1.0,
) -> Path:
    """
    Draw the first three PCA components of a slide as RGB colours.

    Parameters
    ----------
    slide_path : pathlib.Path
        Slide file.

    features_pca_path : pathlib.Path
        `<slide>.h5` PCA features file, as written by `apply_pca`.

    representation_dir : pathlib.Path
        Output directory.

    encoder_name : str
        Encoder name, used in the figure title.

    level_to_view : int, default=4
        Pyramid level of the slide drawn under the embeddings.

    alpha : float, default=1.0
        Opacity of the embeddings overlay.

    Returns
    -------
    pathlib.Path
        Path of the saved `<slide>.jpg` figure.
    """
    _, features_pca = read_h5_features(features_pca_path)
    coords_attrs, coords = read_h5_coords(features_pca_path)

    # Min-max scale the first three components to [0, 1] RGB values
    tiles_df = pd.DataFrame(
        coords,
        columns=["x", "y", "w", "h"],
    )
    tiles_df[["r", "g", "b"]] = MinMaxScaler().fit_transform(features_pca[:, :3])

    slide = get_slide_reader(slide_path)(slide_path)
    fig, ax = plt.subplots(
        1,
        1,
        figsize=VISUALIZATION_FIGSIZE,
        layout="constrained",
    )
    visualise_tile_rgb(
        slide,
        data=tiles_df,
        analyse_level=coords_attrs["level"],
        level_to_view=level_to_view,
        alpha=alpha,
        ax=ax,
        title=f"{encoder_name} PCA rgb",
        show=False,
    )

    representation_path = representation_dir / f"{slide.name}.jpg"
    fig.savefig(representation_path)
    plt.close(fig)

    return representation_path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main(
    known_args: list[str] | None = None,
) -> Path:
    """
    Test PCA on WSI patch encodings.

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

    encoder_name = resolve_encoder_name(args.src_feat)
    output_dirs = build_test_directory(args, encoder_name=encoder_name)

    timer = timetracker(verbose=args.clock)
    timer.tic()

    LOGGER.info("Fitting PCA on %s", args.src_feat)
    files, ipca = fit_pca(args.src_feat)
    files = sorted(Path(file) for file in files)
    LOGGER.info(
        "PCA fitted on %s file(s), %s component(s)",
        len(files),
        ipca.n_components_,
    )

    if args.save_variance:
        save_explained_variance(
            ipca,
            variance_path=output_dirs["root"] / f"pca_{encoder_name}_variance.csv",
        )

    progress = tqdm(
        files,
        desc=f"Applying {encoder_name} PCA",
        unit="wsi",
        disable=not args.tqdm,
    )
    for features_path in progress:
        slide_name, features_pca_path = apply_pca(
            ipca,
            features_path=features_path,
            features_pca_dir=output_dirs["features_pca"],
            encoder_name=encoder_name,
        )
        progress.set_postfix_str(f"wsi: {slide_name}", refresh=True)

        if "representation" in output_dirs:
            slide_path = args.src_wsi / f"{slide_name}.{args.ext_wsi}"
            if not slide_path.is_file():
                LOGGER.warning("Slide not found, visualization skipped: %s", slide_path)
                continue
            draw_pca_rgb(
                slide_path,
                features_pca_path=features_pca_path,
                representation_dir=output_dirs["representation"],
                encoder_name=encoder_name,
                level_to_view=args.level_to_view,
                alpha=args.alpha,
            )

    progress.close()
    timer.toc()
    LOGGER.info("PCA done, results saved to %s", output_dirs["root"])

    return output_dirs["root"]


if __name__ == "__main__":
    main()
