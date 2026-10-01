"""Patch-level encoding with pathology foundation models.

Run a patch encoder (built with `patch_encoder_factory`) over a set of images
and store the resulting features.

- `ImageEncoder`: encode a list of images (paths or arrays).
- `TileEncoder`: encode the patches of a slide from a coordinates h5 file.
"""

# General libraries
import os
from tqdm import tqdm
import warnings
from typing import Optional
from functools import partial
import h5py

# Data libraries
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import PIL

# Tensor op
import torch
from torch.utils.data import DataLoader

# ML libraries
from sklearn.preprocessing import MinMaxScaler

# Plotting libraries
import matplotlib.pyplot as plt

# Project modules
from histonaut.patcher.io import (
    save_h5,
    read_h5_coords,
)
from histonaut.slide.draw import (
    visualise_tile_feat,
    visualise_tile_rgb,
)
from histonaut.patcher.cut import SlidePatcher
from histonaut.patcher.sampler import PatchSampler
from histonaut.encoder.sampler import ImageSampler


class ImageEncoder:
    """Encode a list of images with a patch encoder.

    Images are loaded through an `ImageSampler`, optionally center-cropped,
    transformed with the encoder `eval_transforms` and encoded batch-wise under
    `torch.inference_mode`, with autocast when the encoder precision is not
    `torch.float32`.

    Features are kept in memory and optionally saved to disk.

    Parameters
    ----------
    imgs : list[str | Path | np.ndarray]
        Images to encode, given as file paths (opened with `PIL.Image.open`) or
        as `(H, W)` / `(H, W, C)` arrays.

    tile_encoder : torch.nn.Module
        Patch encoder, typically built with `patch_encoder_factory`.

    crop : int, optional
        Size of the center crop applied before the transforms. Default is None
        (no crop).

    ext : str, optional
        Image file extension. Stored but currently unused. Default is None.

    device : str, optional
        Device the batches are moved to; also used as autocast `device_type`.
        Default is ``"cuda"``.

    num_workers : int, optional
        Number of `DataLoader` workers. Default is 0.

    batch_max : int, optional
        Batch size of the `DataLoader`. Default is 512.

    dst : str, optional
        Output directory. If given, features are saved there after encoding.
        Default is None (no saving).

    save_as : str, optional
        Output format, only ``"h5"`` is supported. Default is ``"h5"``.

    lazy : bool, optional
        If True, encode (and save) immediately at construction time.
        Default is True.

    feat_only : bool, optional
        If True, indexing returns the features only; otherwise it also returns
        the image. Default is True.

    pil : bool, optional
        If True, images returned by indexing are `PIL.Image.Image`; otherwise
        `np.ndarray`. Default is True.

    verbose : bool, optional
        If True, display a progress bar. Default is False.

    Attributes
    ----------
    encoder : torch.nn.Module
        The patch encoder (`tile_encoder`).
    precision : torch.dtype
        Encoder precision, used to cast batches and to enable autocast.
    transforms : callable
        Encoder evaluation transforms.
    nb_features : int
        Number of encoded images (set by `lazy_encoder`).
    features : np.ndarray
        `(N, D)` float32 features (set by `lazy_encoder`).
    feat_path : str
        Path of the saved features file (set only when `dst` is given).

    Notes
    -----
    Output file ``{dst}/features_{enc_name}.h5`` contains:

    - dataset ``features``: `(N, D)` float32, attribute ``encoder`` (encoder
      name);
    - dataset ``images``: the stacked images as arrays, `(N, H, W, C)`,
      attribute ``paths`` (image paths, UTF-8 strings).

    `nb_features` and `features` only exist after `lazy_encoder` was called
    (automatically when `lazy=True`).

    Examples
    --------
    >>> from histonaut.encoder.factory import patch_encoder_factory
    >>> from histonaut.encoder.encode import ImageEncoder
    >>> encoder = patch_encoder_factory("uni_v1").to("cuda").eval()
    >>> image_encoder = ImageEncoder(
    ...     imgs=["patch_0.png", "patch_1.png"],
    ...     tile_encoder=encoder,
    ...     dst="features/",
    ... )
    >>> image_encoder.features.shape  # (N, D)
    """

    def __init__(
        self,
        imgs: list,
        tile_encoder: torch.nn.Module,
        crop: Optional[int] = None,
        ext: Optional[str] = None,
        device: Optional[str] = "cuda",
        num_workers: Optional[int] = 0,
        batch_max: Optional[int] = 512,
        dst: Optional[str] = None,
        save_as: Optional[str] = "h5",
        lazy: Optional[bool] = True,
        feat_only: Optional[bool] = True,
        pil: Optional[bool] = True,
        verbose: Optional[bool] = False,
    ):
        """Store the configuration and encode immediately if `lazy` is True."""
        self.imgs = imgs
        self.ext = ext
        self.crop = crop
        self.encoder = tile_encoder
        self.precision = tile_encoder.precision
        self.transforms = tile_encoder.eval_transforms
        self.device = device
        self.num_workers = num_workers
        self.batch_max = batch_max
        self.dst = dst
        self.save_as = save_as
        self.feat_only = feat_only
        self.pil = pil
        self.verbose = verbose
        self.i = 0
        if lazy:
            self.lazy_encoder()

    def __len__(self):
        """Return the number of encoded images.

        Returns
        -------
        nb_features : int
            Number of encoded images.
        """
        return self.nb_features

    def __iter__(self):
        """Reset the iteration counter and return the encoder itself.

        Returns
        -------
        self : ImageEncoder
            The encoder, iterable over its items.
        """
        self.i = 0
        return self

    def __next__(self):
        """Return the next item (see `__getitem__`).

        Returns
        -------
        item : np.ndarray | tuple
            Next item, as returned by `__getitem__`.

        Raises
        ------
        StopIteration
            When all encoded images have been returned.
        """
        if self.i >= self.nb_features:
            raise StopIteration
        x = self.__getitem__(self.i)
        self.i += 1
        return x

    def __getitem__(self, index):
        """Return the features (and optionally the image) at `index`.

        Parameters
        ----------
        index : int
            Index of the image, in ``[0, len(self))``.

        Returns
        -------
        feat : np.ndarray
            `(D,)` features, returned alone when `feat_only` is True.
        img : tuple
            Only when `feat_only` is False: output of `get_image`, i.e. the
            ``(image, path)`` tuple, returned as ``(img, feat)``.

        Raises
        ------
        IndexError
            If `index` is out of range.
        """
        if 0 <= index < len(self):
            feat = self.features[index]
            if self.feat_only:
                return feat
            else:
                img = self.get_image(index, numpy=not self.pil)
                return img, feat
        else:
            raise IndexError("Index out of range")

    def get_image(self, index, numpy=True):
        """Load the image at `index`.

        Arrays are returned as is. Paths are opened with `PIL.Image.open` and,
        if `numpy` is True, converted to an array with a channel axis
        (`(H, W, C)`); dtypes other than uint8/float32 are cast to float32.

        Parameters
        ----------
        index : int
            Index of the image in `imgs`.
        numpy : bool, optional
            If True, return path-based images as `np.ndarray`, otherwise as
            `PIL.Image.Image`. Default is True.

        Returns
        -------
        img : np.ndarray | PIL.Image.Image
            The image.
        path : str | Path | None
            Image path, or None when the input was an array.
        """
        img = self.imgs[index]
        if isinstance(img, np.ndarray):
            return img, None
        else:
            path = img
            img = PIL.Image.open(path)
            if not numpy:
                return img, path
            img = np.asarray(img)
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            if img.dtype not in [np.uint8, np.float32]:
                img = np.float32(img)
            return img, path

    def progress_bar(self, lenght, verbose):
        """Create a `tqdm` progress bar over batches.

        Parameters
        ----------
        lenght : int
            Total number of batches.
        verbose : bool
            If False, the progress bar is disabled.

        Returns
        -------
        progress : tqdm.tqdm
            The progress bar.
        """
        progress = tqdm(
            desc=f"Images enc with {self.encoder.enc_name}",
            total=lenght,
            unit="batch",
            initial=0,
            leave=False,
            disable=not verbose,
        )
        return progress

    def lazy_encoder(self):
        """Encode all images and save the features if `dst` is set.

        Sets `nb_features`, `features` and, when `dst` is given, `feat_path`.
        """
        self.nb_features, self.features = self.extract_patch_features()
        if self.dst:
            self.feat_path = self.save_features(self.dst, self.save_as)

    @torch.inference_mode()
    def extract_patch_features(self):
        """Encode all images batch-wise.

        Returns
        -------
        nb_features : int
            Number of encoded images.
        features : np.ndarray
            `(N, D)` float32 features, in the order of `imgs`.
        """
        dataset = ImageSampler(
            imgs=self.imgs, transform=self.transforms, crop=self.crop
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_max,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        progress = self.progress_bar(dataloader.__len__(), verbose=self.verbose)
        features = []
        for batch_imgs in dataloader:
            batch_imgs = batch_imgs.to(self.device, self.precision)
            with torch.autocast(
                device_type=self.device,
                dtype=self.precision,
                enabled=(self.precision != torch.float32),
            ):
                batch_features = self.encoder(batch_imgs)
                progress.update()
            features.append(batch_features.to("cpu", dtype=torch.float32).numpy())
        progress.clear()
        # Concatenate features
        features = np.concatenate(features, axis=0)
        print(features.shape)
        return features.shape[0], features

    def save_features(self, dst, save_as="h5"):
        """Save the features and the images to ``{dst}/features_{enc_name}.h5``.

        Parameters
        ----------
        dst : str
            Output directory (created if needed).
        save_as : str, optional
            Output format, only ``"h5"`` is supported. Default is ``"h5"``.

        Returns
        -------
        features_path : str
            Path of the written file.

        Raises
        ------
        ValueError
            If `save_as` is not ``"h5"``.

        Notes
        -----
        See the class docstring for the h5 layout. All images are loaded as
        arrays and stacked, so they must share the same shape.
        """
        images, paths = [], []
        for i in range(self.nb_features):
            img, path = self.get_image(i, numpy=True)
            images.append(img)
            paths.append(path)
        images = np.stack(images, axis=0)
        paths = np.array(paths, dtype=h5py.string_dtype(encoding="utf-8"))
        # Save the features to disk
        os.makedirs(self.dst, exist_ok=True)
        features_path = os.path.join(dst, f"features_{self.encoder.enc_name}.{save_as}")
        if save_as == "h5":
            assets = {"features": self.features, "images": images}
            attributes = {
                "features": {
                    "encoder": self.encoder.enc_name,
                },
                "images": {"paths": paths},
            }
            save_h5(
                features_path,
                assets=assets,
                attributes=attributes,
                mode="w",
            )
        else:
            raise ValueError(f"Invalid save_as: {save_as}. Only h5 is supported.")
        return features_path


class TileEncoder:
    """Encode the patches of a slide listed in a coordinates h5 file or.

    Coordinates and patching attributes are read from `coords_path` (as written
    by `SlidePatcher`). A `SlidePatcher` is rebuilt with these coordinates
    (`custom_xywh`), wrapped in a `PatchSampler` and encoded batch-wise under
    `torch.inference_mode`, with autocast when the encoder precision is not
    `torch.float32`. Features are kept in memory and optionally saved to disk.

    Parameters
    ----------
    slide : OpenWSI
        Slide the patches are read from. Its `name` must match the ``name``
        attribute of the coordinates file.

    tile_encoder : torch.nn.Module
        Patch encoder, typically built with `patch_encoder_factory`.

    coords_path : str
        Path of the coordinates h5 file (dataset ``coords``, `(N, 4)` xywh at
        level 0, with patching attributes).

    device : str, optional
        Device the batches are moved to; also used as autocast `device_type`.
        Default is ``"cuda"``.

    num_workers : int, optional
        Number of `DataLoader` workers. Default is 0.

    batch_max : int, optional
        Batch size of the `DataLoader`. Default is 512.

    dst : str, optional
        Output root directory. If given, features are saved there after
        encoding. Default is None (no saving).

    save_as : str, optional
        Output format, only ``"h5"`` is supported. Default is ``"h5"``.

    feat_only : bool, optional
        If True, indexing returns the features only; otherwise it also returns
        the tile and its coordinates. Default is True.

    lazy : bool, optional
        If True, encode (and save) immediately at construction time. Default is
        True.

    verbose : bool, optional
        If True, display a progress bar. Default is False.

    Attributes
    ----------
    encoder : torch.nn.Module
        The patch encoder (`tile_encoder`).
    precision : torch.dtype
        Encoder precision, used to cast batches and to enable autocast.
    transforms : callable
        Encoder evaluation transforms.
    tile_attr : dict
        Attributes of the ``coords`` dataset of `coords_path`.
    tile_coords : np.ndarray
        `(N, 4)` xywh patch coordinates at level 0.
    name : str | None
        Slide name read from the coordinates file.
    mag_0, pixel_size_0, mag_target, pixel_size_target : float | None
        Level-0 and target magnification / mpp read from the coordinates file.
    patch_size_target, overlap_target : int | None
        Patch size and overlap at the target magnification.
    level, level_size, patch_size_level, overlap_level : int | None
        Pyramid level used for patching and the corresponding size, patch size
        and overlap.
    tissu_thr : float | None
        Tissue threshold used for patching.
    nb_features : int
        Number of encoded patches (set by `lazy_encoder`).
    features : np.ndarray
        `(N, D)` float32 features, aligned with `tile_coords` (set by
        `lazy_encoder`).
    feat_path : str
        Path of the saved features file (set only when `dst` is given).

    Notes
    -----
    Output file ``{dst}/features_{enc_name}/{name}.h5`` contains:

    - dataset ``features``: `(N, D)` float32, attributes ``encoder`` (encoder
      name), ``name`` (slide name) and ``dst`` (features directory);
    - dataset ``coords``: `(N, 4)` xywh at level 0, with the attributes of the
      input coordinates file (`tile_attr`) copied over.

    `nb_features` and `features` only exist after `lazy_encoder` was called
    (automatically when `lazy=True`).

    Examples
    --------
    >>> from histonaut.slide.reader import OpenWSI
    >>> from histonaut.encoder.factory import patch_encoder_factory
    >>> from histonaut.encoder.encode import TileEncoder
    >>> slide = OpenWSI("slide.svs")
    >>> encoder = patch_encoder_factory("uni_v1").to("cuda").eval()
    >>> tile_encoder = TileEncoder(
    ...     slide=slide,
    ...     tile_encoder=encoder,
    ...     coords_path="coords/slide.h5",
    ...     dst="features/",
    ... )
    >>> tile_encoder.feat_path  # 'features/features_uni_v1/slide.h5'
    >>> tile_encoder.features.shape  # (N, D)
    """

    def __init__(
        self,
        slide,
        tile_encoder,
        coords_path: str | None,
        device: Optional[str] = "cuda",
        num_workers: Optional[int] = 0,
        batch_max: Optional[int] = 512,
        dst: Optional[str] = None,
        save_as: Optional[str] = "h5",
        feat_only: Optional[bool] = True,
        lazy: Optional[bool] = True,
        verbose: Optional[bool] = False,
    ):
        """Read the coordinates file and encode immediately if `lazy` is True."""
        self.slide = slide
        self.encoder = tile_encoder
        self.precision = tile_encoder.precision
        self.transforms = tile_encoder.eval_transforms
        self.tile_attr, self.tile_coords = (None, None)
        if coords_path is not None:
            self.tile_attr, self.tile_coords = read_h5_coords(coords_path)

        self.get_patch_attributes()
        self.device = device
        self.num_workers = num_workers
        self.batch_max = batch_max
        self.dst = dst
        self.save_as = save_as
        self.feat_only = feat_only
        self.verbose = verbose
        self.i = 0
        if lazy:
            self.lazy_encoder()

    def __len__(self):
        """Return the number of encoded patches.

        Returns
        -------
        nb_features : int
            Number of encoded patches.
        """
        return self.nb_features

    def __iter__(self):
        """Reset the iteration counter and return the encoder itself.

        Returns
        -------
        self : TileEncoder
            The encoder, iterable over its items.
        """
        self.i = 0
        return self

    def __next__(self):
        """Return the next item (see `__getitem__`).

        Returns
        -------
        item : np.ndarray | tuple
            Next item, as returned by `__getitem__`.

        Raises
        ------
        StopIteration
            When all encoded patches have been returned.
        """
        if self.i >= self.nb_features:
            raise StopIteration
        x = self.__getitem__(self.i)
        self.i += 1
        return x

    def __getitem__(self, index):
        """Return the features (and optionally the tile) at `index`.

        Parameters
        ----------
        index : int
            Index of the patch, in ``[0, len(self))``.

        Returns
        -------
        feat : np.ndarray
            `(D,)` features, returned alone when `feat_only` is True.

        tile : PIL.Image.Image | np.ndarray
            Only when `feat_only` is False: the patch read with
            `slide.get_tile`, returned as ``(tile, feat, (x, y, w, h))``.

        xywh : tuple[int, int, int, int]
            Only when `feat_only` is False: patch coordinates at level 0.

        Raises
        ------
        IndexError
            If `index` is out of range.
        """
        if 0 <= index < len(self):
            feat = self.features[index]
            if self.feat_only:
                return feat
            else:
                x, y, w, h = self.tile_coords[index]
                tile = self.slide.get_tile(x, y, w, h)
                return tile, feat, (x, y, w, h)
        else:
            raise IndexError("Index out of range")

    def progress_bar(self, lenght, verbose):
        """Create a `tqdm` progress bar over batches.

        Parameters
        ----------
        lenght : int
            Total number of batches.

        verbose : bool
            If False, the progress bar is disabled.

        Returns
        -------
        progress : tqdm.tqdm
            The progress bar.
        """
        progress = tqdm(
            desc=f"{self.slide.name} enc with {self.encoder.enc_name}",
            total=lenght,
            unit="batch",
            initial=0,
            leave=False,
            disable=not verbose,
        )
        return progress

    def lazy_encoder(self):
        """Encode all patches and save the features if `dst` is set.

        Sets `nb_features`, `features` and, when `dst` is given, `feat_path`.

        Raises
        ------
        AssertionError
            If the coordinates file `name` differs from `slide.name`.
        """
        if self.name:
            assert self.name == self.slide.name, "tiles are from another slide"
        self.nb_features, self.features = self.extract_patch_features()
        if self.dst:
            self.feat_path = self.save_features(self.dst, self.save_as)

    # move to slidepatcher
    def get_patch_attributes(self):
        """Read the patching attributes from `tile_attr` into instance attributes.

        Sets `mag_0`, `pixel_size_0`, `mag_target`, `pixel_size_target`,
        `patch_size_target`, `overlap_target`, `level`, `level_size`,
        `patch_size_level`, `overlap_level`, `tissu_thr` and `name` (None when
        missing).

        Warns
        -----
        UserWarning
            If `patch_size_target`, `mag_0`, `mag_target` or `overlap_target` is
            missing (the internal `KeyError` is caught and turned into a
            warning).
        """
        try:
            self.mag_0 = self.tile_attr.get("magnification", None)
            self.pixel_size_0 = self.tile_attr.get("mpp", None)
            self.mag_target = self.tile_attr.get("target_magnification", None)
            self.pixel_size_target = self.tile_attr.get("target_mpp", None)
            self.patch_size_target = self.tile_attr.get("target_patch_size", None)
            self.overlap_target = self.tile_attr.get("target_overlap", None)
            self.level = self.tile_attr.get("level", None)
            self.level_size = self.tile_attr.get("level_size", None)
            self.patch_size_level = self.tile_attr.get("level_patch_size", None)
            self.overlap_level = self.tile_attr.get("level_overlap", None)
            self.tissu_thr = self.tile_attr.get("tissu_thr", None)
            self.name = self.tile_attr.get("name", None)
            if None in (
                self.patch_size_target,
                self.mag_0,
                self.mag_target,
                self.overlap_target,
            ):
                raise KeyError("Missing attributes in patch file.")
        except (KeyError, FileNotFoundError, ValueError) as e:
            warnings.warn(f"Cannot read patch file attributes ({str(e)}).")
            # todo work around to get patch info

    @torch.inference_mode()
    def extract_patch_features(self):
        """Encode all patches of `tile_coords` batch-wise.

        Returns
        -------
        nb_features : int
            Number of encoded patches.

        features : np.ndarray
            `(N, D)` float32 features, aligned with `tile_coords`.
        """
        patcher = SlidePatcher(
            slide=self.slide,
            mag_0=self.mag_0,
            mag_target=self.mag_target,
            patch_size=self.patch_size_target,
            overlap=self.overlap_target,
            custom_xywh=self.tile_coords,
            xywh_only=False,
            pil=True,
        )

        dataset = PatchSampler(patcher=patcher, transform=self.transforms)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_max,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        progress = self.progress_bar(dataloader.__len__(), verbose=self.verbose)
        features = []
        for batch_tiles, _ in dataloader:
            batch_tiles = batch_tiles.to(self.device, self.precision)
            with torch.autocast(
                device_type=self.device,
                dtype=self.precision,
                enabled=(self.precision != torch.float32),
            ):
                batch_features = self.encoder(batch_tiles)
                progress.update()
            features.append(batch_features.to("cpu", dtype=torch.float32).numpy())
        progress.clear()

        # Concatenate features
        features = np.concatenate(features, axis=0)
        return features.shape[0], features

    def leiden_patch_features(
        self,
        pcs: int | None = None,
        neighbors: int = 50,
        resolution: float = 0.3,
        solver: str = "arpack",
    ):
        """Cluster the patch features with Leiden.

        Builds an `anndata.AnnData` from `features` (with `tile_coords` as
        ``x``, ``y``, ``w``, ``h`` obs columns), runs PCA (when `pcs` > 0), a
        neighbour graph, UMAP and Leiden clustering with scanpy.

        Parameters
        ----------
        pcs : int, optional
            Number of principal components, capped by the features shape.
            Default is None.

        neighbors : int, optional
            Number of neighbours of the kNN graph, capped by ``N - 1``. Default
            is 50.

        resolution : float, optional
            Leiden resolution. Default is 0.3.

        solver : str, optional
            PCA `svd_solver`. Default is ``"arpack"``.

        Returns
        -------
        features_clustered : anndata.AnnData
            Annotated features with ``obsm["X_pca"]``, ``obsm["X_umap"]`` and
            obs columns ``leiden`` (categorical) and ``leiden_int`` (int32).
        """
        features_clustered = ad.AnnData(
            X=self.features,
            obs=pd.DataFrame(self.tile_coords, columns=["x", "y", "w", "h"]),
        )
        pcs = min(pcs, *(self.features.shape))
        if pcs > 0:
            sc.pp.pca(features_clustered, n_comps=pcs, svd_solver=solver)

        neighbors = min(neighbors, self.features.shape[0] - 1)
        sc.pp.neighbors(
            features_clustered,
            n_neighbors=neighbors,
            n_pcs=pcs,
        )
        sc.tl.umap(features_clustered)  # Compute UMAP
        sc.tl.leiden(features_clustered, resolution=resolution)
        features_clustered.obs["leiden_int"] = features_clustered.obs["leiden"].astype(
            {"leiden": "int32"}
        )
        return features_clustered

    # add kmeans clustering

    def rgb_patch_features(self, pcs: int = 3, solver: str = "arpack"):
        """Project the patch features to RGB colours with PCA.

        Builds an `anndata.AnnData` from `features` (with `tile_coords` as
        ``x``, ``y``, ``w``, ``h`` obs columns), runs PCA and min-max scales the
        components to ``[0, 1]``.

        Parameters
        ----------
        pcs : int, optional
            Number of principal components, capped by the features shape.
            Default is 3.

        solver : str, optional
            PCA `svd_solver`, forced to ``"full"`` when fewer than 3 components
            are used. Default is ``"arpack"``.

        Returns
        -------
        features_rgb : anndata.AnnData
            Annotated features with ``obsm["X_pca"]`` and obs columns ``r``,
            ``g``, ``b`` (scaled components).
        """
        features_rgb = ad.AnnData(
            X=self.features,
            obs=pd.DataFrame(self.tile_coords, columns=["x", "y", "w", "h"]),
        )
        pcs = min(pcs, *(self.features.shape))
        if pcs < 3:
            solver = "full"
        sc.pp.pca(features_rgb, n_comps=pcs, svd_solver=solver)

        scaler = MinMaxScaler()
        scaled_features_rgb = scaler.fit_transform(features_rgb.obsm["X_pca"])
        features_rgb.obs[["r", "g", "b"]] = scaled_features_rgb
        return features_rgb

    def visualize_embeddings(
        self,
        method: str = "leiden",
        level_to_view: int = 4,
        alpha: float = 1,
        save_vis: str | None = None,
        show: bool = False,
        **kwargs,
    ) -> str:
        """Overlay the patch embeddings on the slide thumbnail.

        Computes `leiden_patch_features` or `rgb_patch_features` and draws them
        with `visualise_tile_feat` (Leiden clusters) or `visualise_tile_rgb`
        (PCA colours).

        Parameters
        ----------
        method : str, optional
            Visualization method: ``"leiden"`` or ``"rgb"``. Default is
            ``"leiden"``.

        level_to_view : int, optional
            Pyramid level of the background image. Default is 4.

        alpha : float, optional
            Overlay opacity. Default is 1.

        save_vis : str, optional
            Output directory; if given, the figure is saved as
            ``{save_vis}/{slide.name}_{method}.jpg``. Default is None.

        show : bool, optional
            Whether to show the figure. Default is False.

        **kwargs
            Passed to the selected ``*_patch_features`` method.

        Returns
        -------
        vis_path : str
            Path of the saved figure.

        Raises
        ------
        ValueError
            If `method` is not ``"leiden"`` or ``"rgb"`` (only reached for
            case variants such as ``"RGB"``).
        """
        # Retrieve sampler function
        if method.lower() in ["leiden", "rgb"]:
            visualization = partial(
                getattr(self, method + "_patch_features"),
            )
        emb_data = visualization(**kwargs)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")
        if method == "leiden":
            visualise_tile_feat(
                self.slide,
                data=emb_data.obs,
                feat="leiden_int",
                analyse_level=self.level,
                level_to_view=level_to_view,
                cmap=plt.get_cmap("gist_rainbow"),
                alpha=alpha,
                ax=ax,
                title=f"{self.encoder.enc_name} {method}",
                loc=None,
                show=False,
            )
        elif method == "rgb":
            visualise_tile_rgb(
                self.slide,
                data=emb_data.obs,
                analyse_level=self.level,
                level_to_view=level_to_view,
                alpha=alpha,
                ax=ax,
                title=f"{self.encoder.enc_name} {method}",
                show=False,
            )
        else:
            raise ValueError("embeddings visualization method unknown")

        if show:
            plt.show
        # Save visualization
        if save_vis:
            os.makedirs(save_vis, exist_ok=True)
            vis_path = os.path.join(save_vis, f"{self.slide.name}_{method}.jpg")
            fig.savefig(vis_path)
        return vis_path

    def save_features(self, dst, save_as="h5"):
        """Save features and coordinates to ``{dst}/features_{enc_name}/{name}.h5``.

        Parameters
        ----------
        dst : str
            Output root directory; the ``features_{enc_name}`` subdirectory is
            created if needed.

        save_as : str, optional
            Output format, only ``"h5"`` is supported. Default is ``"h5"``.

        Returns
        -------
        features_path : str
            Path of the written file.

        Raises
        ------
        ValueError
            If `save_as` is not ``"h5"``.

        Notes
        -----
        See the class docstring for the h5 layout.
        """
        # Save the features to disk
        features_dir = os.path.join(dst, f"features_{self.encoder.enc_name}")
        os.makedirs(features_dir, exist_ok=True)
        if save_as == "h5":
            features_path = os.path.join(features_dir, f"{self.name}.{save_as}")
            assets = {"features": self.features, "coords": self.tile_coords}
            attributes = {
                "features": {
                    "encoder": self.encoder.enc_name,
                    "name": self.name,
                    "dst": features_dir,
                },
                "coords": self.tile_attr,
            }
            save_h5(
                features_path,
                assets=assets,
                attributes=attributes,
                mode="w",
            )
        else:
            raise ValueError(f"Invalid save_as: {save_as}. Only h5 is supported.")
        return features_path
