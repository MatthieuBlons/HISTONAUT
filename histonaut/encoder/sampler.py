"""Torch Datasets for encoding.

Datasets feeding images to a patch encoder, or sampling the patch encodings of
a slide from a features h5 file.
"""

# General libraries
import os
from functools import partial

# Data libraries
import numpy as np
import PIL

# Tensor op
from torchvision.transforms import CenterCrop
import torch
from torch.utils.data import Dataset

# Project modules
from histonaut.patcher.io import read_h5_coords, read_h5_features


class ImageSampler(Dataset):
    """Dataset over a list of images, applying an optional crop and transforms.

    Parameters
    ----------
    imgs : list[str | Path | np.ndarray]
        Images given as file paths (opened with `PIL.Image.open`) or as
        `(H, W)` / `(H, W, C)` arrays

    transform : callable | None
        Transforms applied after the crop.

    crop : int, optional
        Size of a `CenterCrop` applied before `transform`. Default is None.
    """

    def __init__(self, imgs, transform, crop=None):
        """Store the images and build the optional center crop."""
        self.imgs = imgs
        self.transform = transform
        if crop is not None:
            self.crop = CenterCrop(crop)

    def __len__(self):
        """Return the number of images.

        Returns
        -------
        length : int
            Number of images in `imgs`.
        """
        return len(self.imgs)

    def __getitem__(self, index):
        """Load, crop and transform the image at `index`.

        Parameters
        ----------
        index : int
            Index of the image in `imgs`.

        Returns
        -------
        img : torch.Tensor | PIL.Image.Image
            Transformed image (typically a `(3, H, W)` tensor after the encoder
            transforms).
        """
        img = self.imgs[index]
        if not isinstance(img, np.ndarray):
            img = PIL.Image.open(img)
        else:
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            if img.dtype not in [np.uint8, np.float32]:
                img = np.float32(img)
            img = torch.tensor(img)
        if self.crop:
            img = self.crop(img)
        if self.transform:
            img = self.transform(img)
        return img


# add possibility to generate random biopsies
# add quantile selection
# add cluster selection
class EncodingSampler(Dataset):
    """Sample the patch encodings of a slide according to a sampling strategy.

    Parameters
    ----------
    sampler_name : str
        Sampling strategy, one of `valid_samplers`:
            - ``"all"``: every patch;
            - ``"random"``: `n_samples` indices drawn uniformly with replacement;
            - ``"random_strict"``: ``"random"`` if enough patches, else ``"all"``;
            - ``"niche"``: square neighbourhoods of `hop` patches around random
            seeds;
            - ``"exact_coords"``: patches whose ``(x, y)`` match `query_coords`.

    feat_path : str
        Path of the features h5 file (datasets ``features`` and ``coords``).

    n_samples : int, optional
        Number of patches to sample. Default is None.

    q_samples : float, optional
        Fraction of patches to sample, in ``(0, 1]``; overrides `n_samples`
        when features are read. Default is None.

    min_samples : int, optional
        Minimum number of sampled patches. Default is 3.

    hop : int, optional
        Neighbourhood radius (in patches) of the ``"niche"`` sampler. Default
        is 2.

    query_coords : np.ndarray, optional
        `(M, 2)` coordinates for the ``"exact_coords"`` sampler (only ``x``,
        ``y`` are used). Default is None.

    read_feat : bool, optional
        If True, read the features (and `total_tiles`) from `feat_path` at
        construction. Default is True.

    read_coords : bool, optional
        If True, read the coordinates and attributes from `feat_path`, and set
        `patch_dim` from the ``level_patch_size`` attribute. Default is True.

    total_tiles : int, optional
        Total number of patches, used when `read_feat` is False. Default is
        None.

    coords : np.ndarray, optional
        `(N, 4)` xywh coordinates, used when `read_coords` is False. Default is
        None.

    patch_dim : tuple[int], optional
        Patch ``(width, height)`` in pixels, used when `read_coords` is False.
        Default is None.

    Attributes
    ----------
    valid_samplers : list[str]
        Supported sampler names.
    no_n_samples : list[str]
        Samplers that do not need `n_samples` / `q_samples`.
    no_total_tiles : list[str]
        Samplers that do not need `total_tiles`.
    with_tile_coords : list[str]
        Samplers that need patch coordinates.
    name : str
        Slide name (basename of `feat_path` without extension).
    features : np.ndarray
        `(N, D)` features (set when `read_feat` is True).
    attributes : dict
        Attributes of the ``coords`` dataset (set when `read_coords` is True).
    sampler : functools.partial
        Bound ``{sampler_name}_sampler`` method.
    sample_ids : torch.Tensor
        Indices of the sampled patches.
    sample_feats : np.ndarray
        Features of the sampled patches.

    Raises
    ------
    AssertionError
        If neither a valid `n_samples` nor `q_samples` is given for a sampler
        that needs it, or if `query_coords` is missing for ``"exact_coords"``.
    ValueError
        If `sampler_name` is not in `valid_samplers`.
    """

    # Defining valid options for samplers
    valid_samplers = ["all", "random", "random_strict", "niche", "exact_coords"]
    no_n_samples = ["all", "exact_coords"]
    no_total_tiles = ["niche"]
    with_tile_coords = ["niche", "exact_coords"]

    def __init__(
        self,
        sampler_name: str,
        feat_path: str,
        n_samples: int = None,
        q_samples: float = None,
        min_samples: int = 3,
        hop: int = 2,
        query_coords: np.ndarray = None,
        read_feat: bool = True,
        read_coords: bool = True,
        total_tiles: int = None,
        coords: np.ndarray = None,
        patch_dim: tuple[int] = None,
    ):
        """Read the h5 file, check the arguments and draw the samples."""

        # Storing useful variables
        self.sampler_name = sampler_name
        self.feat_path = feat_path
        self.name, _ = os.path.splitext(os.path.basename(feat_path))
        self.n_samples = n_samples
        self.q_samples = q_samples
        self.hop = hop
        self.query_coords = query_coords
        self.read_feat = read_feat
        self.read_coords = read_coords
        self.total_tiles = total_tiles
        self.coords = coords
        self.patch_dim = patch_dim
        self.min_samples = min_samples

        # Check that a number of samples was given
        if sampler_name not in EncodingSampler.no_n_samples:
            assert (n_samples is not None and n_samples > 0) or (
                q_samples is not None and 0 < q_samples <= 1
            ), (
                f"Please provide a valid value for n_samples (int > 0) or q_samples (0 < float <= 1) "
                + f"when sampler name is not in: {', '.join(EncodingSampler.no_n_samples)}."
            )

        if sampler_name == "exact_coords":
            assert query_coords is not None and isinstance(
                query_coords, np.ndarray
            ), "exact_coords sampler requires query_coords to be provided."

        # Reading file information
        if self.read_feat:
            # Retrieve features and attributes and coordinates
            _, self.features = self.read_h5_feat(feat_path)
            self.total_tiles = self.features.shape[0]
            if self.q_samples is not None:
                self.n_samples = max(
                    int(self.q_samples * self.total_tiles), self.min_samples
                )

        if self.read_coords:
            self.attributes, self.coords = self.read_h5_coord(feat_path)
            # Retrieve patch's dimensions
            self.patch_dim = (
                self.attributes["level_patch_size"],
                self.attributes["level_patch_size"],
            )

        # Retrieve sampler function
        if sampler_name.lower() in self.valid_samplers:
            self.sampler = partial(
                getattr(self, self.sampler_name + "_sampler"),
                **{
                    "n_samples": self.n_samples,
                    "total_tiles": self.total_tiles,
                    "hop": self.hop,
                    "query_coords": self.query_coords,
                },
            )
        else:
            raise ValueError(
                f"Invalid sampler: {sampler_name}."
                + f"\nPlease choose a valid sampler from : {', '.join(self.valid_samplers)}"
            )

        self.sample_ids, self.sample_feats = self.get_feat()

    def __len__(self):
        """Return the total number of patches of the slide.

        Returns
        -------
        total_tiles : int
            Total number of patches (not the number of sampled patches).
        """
        return self.total_tiles

    def __getitem__(self, idx):
        """Return the sampled patch index and features at position `idx`.

        Parameters
        ----------
        idx : int
            Position in the sampled subset.

        Returns
        -------
        sample_id : torch.Tensor
            Index of the patch in the slide.

        sample_feat : np.ndarray
            `(D,)` features of the patch.
        """
        return self.sample_ids[idx], self.sample_feats[idx]

    def get_feat(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw the patch indices with `sampler` and gather their features.

        Returns
        -------
        samples_id : torch.Tensor
            Indices of the sampled patches.

        feat : np.ndarray
            `(n, D)` features of the sampled patches.
        """
        if (
            self.sampler_name not in EncodingSampler.no_n_samples
            and self.n_samples is None
            and self.q_samples is not None
        ):
            self.n_samples = max(
                int(self.total_tiles * self.q_samples), self.min_samples
            )
        samples_id = self.sampler(n_samples=self.n_samples)
        if self.read_feat:
            feat = self.features[samples_id]
        else:
            feat = self.read_h5_feat(self.feat_path)[samples_id]
        return samples_id, feat

    def random_sampler(
        self, n_samples: int, total_tiles: int = None, *args, **kwargs
    ) -> torch.Tensor:
        """Draw patch indices uniformly at random, with replacement.

        Parameters
        ----------
        n_samples : int
            Number of indices to draw, raised to at least `min_samples`.

        total_tiles : int, optional
            Total number of patches. Default is None (use `self.total_tiles`).

        *args, **kwargs
            Ignored (allow a common sampler signature).

        Returns
        -------
        indices : torch.Tensor
            `(n_samples,)` int64 indices in ``[0, total_tiles)``.

        Raises
        ------
        AssertionError
            If `total_tiles` is None or not positive.
        """

        # Default args parsing
        if total_tiles is None:
            total_tiles = self.total_tiles
        assert total_tiles is not None and total_tiles > 0, (
            "Invalid total_tiles value: {}".format(total_tiles)
            + "\nPlease provide a valid value (int > 0) for total_tiles."
        )

        # Check min samples
        n_samples = max(n_samples, self.min_samples)

        # Returning selected indices
        return torch.randint(0, total_tiles, (n_samples,))

    def all_sampler(self, total_tiles: int = None, *args, **kwargs) -> torch.Tensor:
        """Return the indices of all patches.

        Parameters
        ----------
        total_tiles : int, optional
            Total number of patches. Default is None (use `self.total_tiles`).

        *args, **kwargs
            Ignored (allow a common sampler signature).

        Returns
        -------
        indices : torch.Tensor
            `(total_tiles,)` int32 indices ``0, ..., total_tiles - 1``.

        Raises
        ------
        AssertionError
            If `total_tiles` is None or not positive.
        """

        # Default args parsing
        if total_tiles is None:
            total_tiles = self.total_tiles
        assert total_tiles is not None and total_tiles > 0, (
            "Invalid total_tiles value: {}".format(total_tiles)
            + "\nPlease provide a valid value (int > 0) for total_tiles."
        )

        # Returning the selected indices
        return torch.IntTensor(list(range(total_tiles)))

    def random_strict_sampler(
        self, n_samples: int, total_tiles: int = None, *args, **kwargs
    ) -> torch.Tensor:
        """Draw random indices if enough patches exist, otherwise return all.

        Parameters
        ----------
        n_samples : int
            Number of indices to draw.

        total_tiles : int, optional
            Total number of patches. Default is None (use `self.total_tiles`).

        *args, **kwargs
            Ignored (allow a common sampler signature).

        Returns
        -------
        indices : torch.Tensor
            Output of `random_sampler` if ``total_tiles >= n_samples``, else of
            `all_sampler`.

        Raises
        ------
        AssertionError
            If `total_tiles` is None or not positive.
        """
        # Default args parsing
        if total_tiles is None:
            total_tiles = self.total_tiles
        assert total_tiles is not None and total_tiles > 0, (
            "Invalid total_tiles value: {}".format(total_tiles)
            + "\nPlease provide a valid value (int > 0) for total_tiles."
        )

        # Returning the indices selected with the corresponding method
        if total_tiles >= n_samples:
            return self.random_sampler(n_samples, total_tiles=total_tiles)
        else:
            return self.all_sampler()

    def niche_sampler(
        self,
        n_samples: int,
        hop: int = None,
        coords: np.ndarray = None,
        patch_dim: tuple[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Sample square neighbourhoods ("niches") around random seed patches.

        Parameters
        ----------
        n_samples : int
            Target number of patches, raised to at least `min_samples`.

        hop : int, optional
            Neighbourhood radius in patches. Default is None (use `self.hop`).

        coords : np.ndarray, optional
            `(N, 2+)` patch coordinates. Default is None (use `self.coords`).

        patch_dim : tuple[int], optional
            Patch ``(width, height)`` in pixels, used as grid step. Default is
            None (use `self.patch_dim`).

        *args, **kwargs
            Ignored (allow a common sampler signature).

        Returns
        -------
        indices : torch.Tensor
            Unique int32 indices of the patches in the niches.

        Raises
        ------
        AssertionError
            If `patch_dim` is None, or `coords` is None or not an `np.ndarray`.
        """

        # Parse default args
        if hop is None:
            hop = self.hop
        if coords is None:
            coords = self.coords
        if patch_dim is None:
            patch_dim = self.patch_dim

        # Checking minimum samples
        n_samples = max(n_samples, self.min_samples)

        # Check that all the information are given
        assert patch_dim is not None, (
            "Invalid patch_dim value: {}.".format(patch_dim)
            + "\n Please provide a tuple of ints (width, height) in number of pixels for patch_dim."
        )
        assert coords is not None and isinstance(coords, np.ndarray), (
            "Invalid coords value: {}.".format(coords)
            + "\nPlease provide an array of size (total_tiles, 2)."
        )

        # Initialize indices, number of niches and niche centers
        indices = []
        n_niches = int(n_samples / (2 * hop + 1) ** 2)
        seeds = self.random_sampler(n_niches)

        # For each niche center, retrieve the tils in the niche
        for i, s in enumerate(seeds):
            seed_x, seed_y = coords[s, :2]
            neigh_x = np.arange(
                seed_x - hop * patch_dim[0],
                seed_x + hop * patch_dim[0] + 1,
                patch_dim[0],
            )
            neigh_y = np.arange(
                seed_y - hop * patch_dim[1],
                seed_y + hop * patch_dim[1] + 1,
                patch_dim[1],
            )
            inx = np.nonzero(np.isin(coords[:, 0], neigh_x))[0]
            iny = np.nonzero(np.isin(coords[:, 1], neigh_y))[0]
            inter = np.intersect1d(inx, iny)
            indices += inter.tolist()

        # Returning the list of selected tiles' indices
        return torch.IntTensor(list(set(indices)))

    def exact_coords_sampler(self, query_coords=None, *args, **kwargs) -> torch.Tensor:
        """Return the indices of the patches whose ``(x, y)`` match `query_coords`.

        Parameters
        ----------
        query_coords : np.ndarray, optional
            `(M, 2+)` query coordinates; only ``x``, ``y`` are matched, unknown
            coordinates are skipped. Default is None (use `self.query_coords`).

        *args,
        **kwargs
            Ignored (allow a common sampler signature).

        Returns
        -------
        indices : torch.Tensor
            Unique int32 indices of the matching patches in `coords`.
        """

        if query_coords is None:
            query_coords = self.query_coords

        coord_to_idx = {tuple(c): i for i, c in enumerate(self.coords[:, :2])}

        indices = [
            coord_to_idx[tuple(c)]
            for c in query_coords[:, :2]
            if tuple(c) in coord_to_idx
        ]

        return torch.IntTensor(list(set(indices)))

    def read_h5_feat(self, path: str) -> tuple[dict, np.ndarray]:
        """Read the ``features`` dataset of a features h5 file.

        Parameters
        ----------
        path : str
            Path of the features h5 file.

        Returns
        -------
        attributes : dict
            Attributes of the ``features`` dataset.

        features : np.ndarray
            `(N, D)` features.
        """

        # Retrieving the h5 features and attributes
        return read_h5_features(embs_path=path)

    def read_h5_coord(self, path: str) -> tuple[dict, np.ndarray]:
        """Read the ``coords`` dataset of a features h5 file.

        Parameters
        ----------
        path : str
            Path of the features h5 file.

        Returns
        -------
        attributes : dict
            Attributes of the ``coords`` dataset (patching attributes).

        coords : np.ndarray
            `(N, 4)` xywh coordinates at level 0.
        """
        return read_h5_coords(coords_path=path)
