"""Aggregate patch-level encodings into a slide-level embedding.

`aggragate_tiles_features` reads a features h5 file, checks that its patch
encoder matches the slide encoder (`SLIDE_TO_TILE_ENCODER_MATCH`) and runs
the slide encoder.

Supported output formats (`save_as`): `"h5"`.
"""

# General libraries
import os
from typing import Optional

# Tensor op
import torch

# Project modules
from histonaut.patcher.io import (
    save_h5,
    read_h5_coords,
    read_h5_features,
)
from histonaut.encoder.factory import SLIDE_TO_TILE_ENCODER_MATCH


@torch.inference_mode()
def aggragate_tiles_features(
    features_path: str,
    slide_encoder: torch.nn.Module,
    device: Optional[str] = "cuda",
    dst: Optional[str] = None,
    save_as: Optional[str] = "h5",
) -> str:
    """Encode a slide from its patch features and save the embedding.

    Reads the `"coords"` and `"features"` datasets of `features_path`, checks
    the patch encoder against `SLIDE_TO_TILE_ENCODER_MATCH` (skipped for
    `mean-*` encoders), runs `slide_encoder` under autocast and writes the
    result to `dst/<name>.<save_as>`.

    Parameters
    ----------
    features_path : str
        Path to a patch features h5 file with datasets `"coords"` (`(N, 4)`
        xywh at level 0) and `"features"` (`(N, D)`, attributes `"name"` and
        `"encoder"`).

    slide_encoder : torch.nn.Module
        Slide encoder (`BaseSlideEncoder`) exposing `enc_name` and `precision`
        and called as `slide_encoder(batch, device)` with
        `batch = {"features", "coords", "attributes"}`.

    device : str, optional
        Torch device, also used as autocast `device_type`. Default is `"cuda"`.

    dst : str, optional
        Output directory, created if needed. Default is None.

    save_as : str, optional
        Output file extension; only `"h5"` is supported. Default is `"h5"`.

    Returns
    -------
    save_path : str
        Path to the saved h5 file, holding datasets `"features"` (slide
        embedding `(D_slide,)`, attributes `"encoder"`, `"name"`, `"dst"`),
        `"tile_features"` and `"coords"` (copied with their attributes).

    Raises
    ------
    KeyError
        If the `"features"` attributes lack `"name"` or `"encoder"`.
    ValueError
        If `dst` is None.
    """

    # Set the slide encoder model to device and eval
    slide_encoder.to(device)
    slide_encoder.eval()

    xywh_attrs, xywh = read_h5_coords(features_path)
    tile_attrs, tile_feats = read_h5_features(features_path)
    slide_name = tile_attrs["name"]
    tile_encoder = tile_attrs["encoder"]
    # tile encoder sanity check:
    if not slide_encoder.enc_name.startswith("mean-"):
        try:
            SLIDE_TO_TILE_ENCODER_MATCH[slide_encoder.enc_name] == tile_encoder
        except ValueError as e:
            print(
                f"Tile features were extracted with a tile encoder which does not match the slide encoder provided"
            )

    # Convert slide_features to tensor
    tile_features = torch.from_numpy(tile_feats).float().to(device)
    tile_features = tile_features.unsqueeze(0)  # Add batch dimension

    coords = torch.from_numpy(xywh[:, :2]).to(device)
    coords = coords.unsqueeze(0)  # Add batch dimension

    # Prepare input batch dictionary
    batch = {"features": tile_features, "coords": coords, "attributes": xywh_attrs}

    # Generate slide-level features
    with torch.autocast(
        device_type=device,
        enabled=(slide_encoder.precision != torch.float32),
    ):
        slide_feats = slide_encoder(batch, device)
    slide_feats = slide_feats.float().cpu().numpy().squeeze()

    # Save slide-level features if save path is provided
    if dst:
        os.makedirs(dst, exist_ok=True)
        save_path = os.path.join(dst, f"{slide_name}.{save_as}")
        assets = {
            "features": slide_feats,
            "tile_features": tile_feats,
            "coords": xywh,
        }
        attributes = {
            "features": {
                "encoder": slide_encoder.enc_name,
                "name": slide_name,
                "dst": dst,
            },
            "tile_features": tile_attrs,
            "coords": xywh_attrs,
        }
        save_h5(
            save_path,
            assets=assets,
            attributes=attributes,
            mode="w",
        )
    else:
        raise ValueError(f"Invalid save_as: {save_as}. Only h5 is supported.")

    return save_path
