"""Torch (GPU) versions of the patch filters in `histonaut.slide.processing`."""

# General libraries
from functools import partial
import numpy as np

# Torch for tensor op
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Image processing modules
from torchvision.transforms.functional import rgb_to_grayscale

# installed from https://github.com/Simon-Bertrand/HOG-PyTorch.git
from torch_hog import hog as torch_hog

# Project modules
# Defining valid statistic function
from histonaut.slide.processing import VALID_STAT_FUNC, VALID_AGG_FUNC


# down sample images? wth transform
class TensorImg(Dataset):
    """Torch `Dataset` over a list of NumPy patches.

    Parameters
    ----------
    img_list : list[np.ndarray] | np.ndarray
        Patches as `(H, W, C)` arrays, or a stacked `(B, H, W, C)` array.

    Attributes
    ----------
    img_list : list[np.ndarray] | np.ndarray
        The patches given at construction.
    """

    def __init__(self, img_list):
        """Store the list of patches."""
        self.img_list = img_list

    def __len__(self):
        """Return the number of patches.

        Returns
        -------
        length : int
            Number of patches in `img_list`.
        """
        return len(self.img_list)

    def __getitem__(self, idx):
        """Return a patch as a float tensor with its index.

        Parameters
        ----------
        idx : int
            Index of the patch in `img_list`.

        Returns
        -------
        img : torch.Tensor
            `(H, W, C)` float32 tensor (channels last, values not rescaled).

        idx : int
            The input index, used to map batch results back to `img_list`.
        """
        img = self.img_list[idx]
        img = torch.from_numpy(img).float()
        return img, idx  # H×WxC


def safe_loader(img_list, batch_size=64, num_workers=0):
    """Build an ordered `DataLoader` over a list of patches.

    Parameters
    ----------
    img_list : list[np.ndarray] | np.ndarray
        Patches as `(H, W, C)` arrays, or a stacked `(B, H, W, C)` array.

    batch_size : int, optional
        Number of patches per batch. Default is 64.

    num_workers : int, optional
        Number of `DataLoader` worker processes. Default is 0.

    Returns
    -------
    loader : torch.utils.data.DataLoader
        Non-shuffled loader over `TensorImg(img_list)` with pinned memory,
        yielding `(batch, idxs)` with `batch` of shape `(B, H, W, C)`.
    """
    return DataLoader(
        TensorImg(img_list),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def hog_selection_torch(
    img_list: list[np.ndarray] | np.ndarray,
    hog_thresh: float = 0.5,
    hog_std_thresh: float | None = None,
    orientations: int = 9,
    pixels_per_cell: int = 16,
    cells_per_block: int = 2,
    use_grayscale: bool = False,
    device: str = "cpu",
    batch_size: int = 1024,
    num_workers: int = 0,
):
    """Select patches by the mean (and optionally std) of their HOG image.

    Torch version of `histonaut.slide.processing.hog_selection`.

    Parameters
    ----------
    img_list : list[np.ndarray] | np.ndarray
        Patches as `(H, W, 3)` RGB arrays, or a stacked `(B, H, W, 3)` array.

    hog_thresh : float, optional
        Minimum mean of the HOG image to keep a patch. Default is 0.5.

    hog_std_thresh : float | None, optional
        Minimum std of the HOG image to keep a patch. Default is None (no std
        criterion).

    orientations : int, optional
        Number of orientation bins (`nPhaseBins`). Default is 9.

    pixels_per_cell : int, optional
        Cell size in pixels (`cellSize`). Default is 16.

    cells_per_block : int, optional
        Block size in cells (`blockSize`). Default is 2.

    use_grayscale : bool, optional
        If True, convert patches to grayscale before computing the HOG.
        Default is False.

    device : str, optional
        Torch device used for the computation. Default is `"cpu"`.

    batch_size : int, optional
        Number of patches per batch. Default is 1024.

    num_workers : int, optional
        Number of `DataLoader` worker processes. Default is 0.

    Returns
    -------
    idx_selected : np.ndarray
        1D integer array of the indices (in `img_list`) of the kept patches.

    Notes
    -----
    The threshold is applied to `hog.visualize(...)` (the HOG rendering),
    not to the HOG feature vector.
    """
    loader = safe_loader(
        img_list,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    hog = torch_hog.HOG(
        cellSize=pixels_per_cell,
        blockSize=cells_per_block,
        nPhaseBins=orientations,
        kernel="finite",
        normalization="L2",
        accumulate="simple",
        channelWise=True,
    ).to(device)

    selected_indices = []
    with torch.no_grad():
        for batch, idxs in loader:
            # Move batch only
            # torch_hog looks GPU-ready, but is not fully CUDA-safe
            batch = batch.to(device)
            batch = batch.permute(0, 3, 1, 2)  # (B, C, H, W)
            if use_grayscale:
                batch = rgb_to_grayscale(batch)

            hog_imgs = hog.visualize(
                batch
            )  # (B, C, H, W) ask lily why we use the image and not the hog feats?

            hog_flat = hog_imgs.reshape(hog_imgs.shape[0], -1)
            hog_means = hog_flat.mean(dim=1)
            hog_filter = hog_means >= hog_thresh

            if hog_std_thresh is not None:
                hog_stds = hog_flat.std(dim=1)
                hog_filter &= hog_stds >= hog_std_thresh

            selected_indices += idxs[hog_filter].cpu().tolist()
    return np.array(selected_indices)


# torch implementation OK
def patch_mean_pooling_torch(
    img: torch.Tensor, patch_size: int | tuple[int, int]
) -> torch.Tensor:
    """Average-pool a batch of images over non-overlapping patches, ignoring NaN.

    NaN pixels (e.g. masked background) are excluded from each patch mean;
    patches containing only NaN are set to NaN.

    Parameters
    ----------
    img : torch.Tensor
        `(B, H, W, C)` float tensor of images (channels last).

    patch_size : int | tuple[int, int]
        Pooling kernel and stride, in pixels; an `int` gives square patches.

    Returns
    -------
    mean_img : torch.Tensor
        `(B, H // ph, W // pw, C)` tensor of per-patch means (channels last).
        Trailing pixels that do not fill a full patch are dropped.
    """

    # Parsing patch size
    if isinstance(patch_size, int):
        pw, ph = patch_size, patch_size
    else:
        pw, ph = patch_size

    # convert to NCHW
    img = img.permute(0, 3, 1, 2)  # B, C, W, H

    # Crop to valid size
    # Or use padding

    # Returning the averaged image per patch
    valid = ~torch.isnan(img)
    img_safe = torch.where(valid, img, torch.zeros_like(img))

    pooled_sum = F.avg_pool2d(
        img_safe,
        kernel_size=(pw, ph),
        stride=(pw, ph),
        divisor_override=1,
    )

    pooled_count = F.avg_pool2d(
        valid.float(),
        kernel_size=(pw, ph),
        stride=(pw, ph),
        divisor_override=1,
    )

    mean_img = pooled_sum / pooled_count
    mean_img[pooled_count == 0] = torch.nan

    return mean_img.permute(0, 2, 3, 1)  # B, W, H, C


# torch implementation OK
def compute_channel_std_torch(
    img_list: list[torch.Tensor] | torch.Tensor, aggregate: str = "mean", q: float = 0.5
):
    """Compute the per-pixel std across channels, aggregated per image.

    For each pixel, the standard deviation across the 3 channels is
    computed; these values are then aggregated over the pixels of each image
    with `aggregate`. NaN pixels are ignored.

    Parameters
    ----------
    img_list : list[torch.Tensor] | torch.Tensor
        Flattened images, either a list of `(N, 3)` tensors (stacked) or a
        `(B, N, 3)` tensor, with `N` the number of pixels.

    aggregate : str, optional
        Aggregation over pixels: `"quantile"` or the name of a `torch`
        reduction accepting `dim` (e.g. `"mean"`, `"median"`, see
        `VALID_AGG_FUNC`). Default is `"mean"`.

    q : float, optional
        Quantile used when `aggregate="quantile"`. Default is 0.5.

    Returns
    -------
    stat_values : torch.Tensor
        `(B, 1)` tensor of aggregated channel std per image.

    Raises
    ------
    AssertionError
        If the input is not of shape `(B, N, 3)`, or `aggregate` is neither
        `"quantile"` nor an attribute of `torch`.
    """
    # Convert input to tensor
    if isinstance(img_list, (list, tuple)):
        imgs = torch.stack(img_list, dim=0)
    else:
        imgs = img_list

    # imgs: (B, N, 3)
    assert imgs.ndim == 3 and imgs.shape[-1] == 3, "Expected input of shape (B, WxH, 3)"

    # Retrieving the aggregate function
    if aggregate == "quantile":
        agg_func = lambda t: torch.quantile(t, q=q, dim=-1)
    else:
        assert hasattr(torch, aggregate), (
            "Aggregate function {} not implemented.".format(aggregate)
            + f"Please choose a valid argument between: {', '.join(VALID_AGG_FUNC)}."
        )
        agg_func = lambda t: getattr(torch, aggregate)(t, dim=-1)

    # Computing channel std
    channel_std = imgs.std(dim=-1)

    # Aggregated the results across images in the list
    if torch.isnan(channel_std).any():
        # fallback per-image (matches np.apply_along_axis logic)
        out = []
        for i in range(channel_std.shape[0]):
            vals = channel_std[i]
            vals = vals[~torch.isnan(vals)]
            out.append(agg_func(vals))
        stat_values = torch.stack(out)
    else:
        stat_values = agg_func(channel_std)

    return stat_values.reshape(-1, 1)


# torch implementation OK
def color_filter_selection_torch(
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
    device: str = "cpu",
    batch_size: int = 1024,
    num_workers: int = 0,
) -> list[int]:
    """Select patches whose per-channel colour statistic exceeds a threshold.

    For each patch, optionally mask background pixels (set to NaN), optionally
    average over local `filter_size` windows, compute `stat_fct` over the
    pixels for each channel (R, G, B) and keep the patch if the statistic is
    at least `color_thresh` for any channel (or all channels if
    `filter_all`).

    Parameters
    ----------
    img_list : list[np.ndarray] | np.ndarray
        Patches as `(H, W, 3)` RGB arrays, or a stacked `(B, H, W, 3)` array.

    color_thresh : tuple[int, int, int] | int
        Minimum statistic value per channel (R, G, B); an `int` is used for
        all channels.

    stat_fct : str, optional
        Statistic computed over pixels: `"quantile"`, `"channel_std"` (see
        `compute_channel_std_torch`) or the name of a `torch` reduction
        accepting `dim`. Must be in `VALID_STAT_FUNC`. Default is
        `"quantile"`.

    q : float, optional
        Quantile used when `stat_fct="quantile"`. Default is 0.05.

    remove_background : bool, optional
        Whether to mask background pixels before computing the statistic.
        Default is True.

    background_thresh : tuple[int, int, int] | int, optional
        Per-channel intensity threshold defining background. Default is 245.

    local_average : bool, optional
        Whether to average over `filter_size` windows with
        `patch_mean_pooling_torch` before computing the statistic. Default is
        True.

    filter_size : tuple[int, int] | int, optional
        Window size for local averaging, in pixels. Default is 16.

    filter_all : bool, optional
        If True, the threshold must be met by all channels; otherwise by at
        least one. Default is False.

    background_average : bool, optional
        If True, mask background after local averaging; otherwise before.
        Default is True.

    aggregate : str, optional
        Aggregation over pixels used by `compute_channel_std_torch` when
        `stat_fct="channel_std"`. Default is `"mean"`.

    device : str, optional
        Torch device used for the computation. Default is `"cpu"`.

    batch_size : int, optional
        Number of patches per batch. Default is 1024.

    num_workers : int, optional
        Number of `DataLoader` worker processes. Default is 0.

    Returns
    -------
    idx_selected : list[int]
        Indices (in `img_list`) of the kept patches.

    Raises
    ------
    AssertionError
        If `color_thresh` is None or `stat_fct` is not in `VALID_STAT_FUNC`.

    Notes
    -----
    Masking rules differ between the two branches: before averaging, a
    pixel is masked if any channel is `>= background_thresh`; after
    averaging (`background_average=True`), a window is masked if any channel
    is `< background_thresh`.
    """
    # cast to tensor
    loader = safe_loader(
        img_list,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Parsing color threshold argument
    if isinstance(color_thresh, int):
        color_thresh = (color_thresh, color_thresh, color_thresh)
    color_thresh = torch.tensor(color_thresh, device=device)

    if remove_background and isinstance(background_thresh, int):
        background_thresh = (background_thresh, background_thresh, background_thresh)
        background_thresh = torch.tensor(background_thresh, device=device)

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
        stat_fn = lambda x: torch.quantile(x, q=q, dim=-2)  # is it the right dim?
    elif stat_fct == "channel_std":
        stat_fn = partial(compute_channel_std_torch, aggregate=aggregate)
    else:
        stat_fn = lambda x: getattr(torch, stat_fct)(x, dim=-2)  # is it the right dim?

    idx_selected = []
    with torch.no_grad():
        for batch, idxs in loader:
            # Move batch only
            batch = batch.to(device)
            B = batch.shape[0]
            # Removing background on image before computing the local average
            if remove_background and not background_average:
                mask = (batch >= background_thresh).any(dim=-1, keepdim=True)
                batch = batch.masked_fill(mask, torch.nan)

            # Local averaging of the input
            if local_average:
                batch = patch_mean_pooling_torch(batch, filter_size)

            if remove_background and background_average:
                mask = (batch < background_thresh).any(dim=-1, keepdim=True)
                batch = batch.masked_fill(mask, torch.nan)
                flat = batch.reshape(B, -1, 3)
                stat_values = stat_fn(flat)

            else:
                flat = batch.reshape(B, -1, 3)
                stat_values = stat_fn(flat)

            # Filter for the images based on the channels' minimum color threshold
            mask = stat_values >= color_thresh
            img_filter = mask.all(dim=-1) if filter_all else mask.any(dim=-1)
            # Retrieving the indices selected
            idx_selected += idxs[img_filter].tolist()
    # Returning selected indices
    return idx_selected


# torch implementation OK
def filter_gray_img_torch(
    img_list: list[np.ndarray] | np.ndarray,
    std_thresh: int,
    remove_background: bool = True,
    background_thresh: int | tuple[int, int, int] = 245,
    local_average: bool = True,
    filter_size: int | tuple[int, int] = 16,
    aggregate: str = "mean",
):
    """Filter out gray patches based on their channel standard deviation.

    Calls `color_filter_selection_torch` with `stat_fct="channel_std"`: a
    patch is kept if its aggregated across-channel std is at least
    `std_thresh`, after optional background removal and local averaging.

    Parameters
    ----------
    img_list : list[np.ndarray] | np.ndarray
        Patches as `(H, W, 3)` RGB arrays, or a stacked `(B, H, W, 3)` array.

    std_thresh : int
        Minimum aggregated channel std to keep a patch.

    remove_background : bool, optional
        Whether to mask background before computing the channel std. Default
        is True.

    background_thresh : int | tuple[int, int, int], optional
        Per-channel intensity threshold defining background. Default is 245.

    local_average : bool, optional
        Whether to average over `filter_size` windows before computing the
        channel std. Default is True.

    filter_size : int | tuple[int, int], optional
        Window size for local averaging, in pixels. Default is 16.

    aggregate : str, optional
        Aggregation of the channel std over pixels (see
        `compute_channel_std_torch`). Default is `"mean"`.

    Returns
    -------
    idx_selected : list[int]
        Indices (in `img_list`) of the kept (non-gray) patches.

    Notes
    -----
    `device`, `batch_size` and `num_workers` are not forwarded, so the
    computation always uses the defaults of `color_filter_selection_torch`
    (CPU).
    """

    # Color filter based on channel std
    idx_selected = color_filter_selection_torch(
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
