"""Fit an `IncrementalPCA` over a directory of patch-encoding h5 files.
"""

# General libraries
from glob import glob
import os

# Data libraries
import numpy as np

# Log libraries
from tqdm import tqdm

# ML libraries
from sklearn.decomposition import IncrementalPCA

# Project modules
from histonaut.patcher.io import read_h5_features


def check_batch(batch):
    """Check whether a batch holds enough patches for `partial_fit`.

    `IncrementalPCA.partial_fit` needs at least as many samples as features.

    Parameters
    ----------
    batch : list[np.ndarray]
        Feature matrices, one `(N_i, D)` array per slide.

    Returns
    -------
    is_ready : bool
        True if the total number of patches is at least `D` (taken from the
        last matrix); False for an empty batch.
    """
    if batch:
        n_tiles = np.sum([x.shape[0] for x in batch])
        n_features = batch[-1].shape[1]
        ans = n_tiles >= n_features
    else:
        ans = False
    return ans


def get_feat_files(feat_dir):
    """List the h5 feature files of a directory.

    Parameters
    ----------
    feat_dir : str
        Directory holding the feature files.

    Returns
    -------
    files : list[str]
        Paths matching `feat_dir/*.h5` (not recursive, unsorted).
    """
    files = glob(os.path.join(feat_dir, "*.h5"))
    return files


def fit(feat_dir):
    """Fit an incremental PCA on the patch features of a directory.

    1-D feature arrays are treated as a single row and all-zero matrices are
    skipped. Matrices are accumulated until `check_batch` is satisfied, then
    stacked and passed to `IncrementalPCA.partial_fit`.

    Parameters
    ----------
    feat_dir : str
        Directory holding the `*.h5` feature files.

    Returns
    -------
    files : list[str]
        Paths of all the h5 files found in `feat_dir`.
    ipca : sklearn.decomposition.IncrementalPCA
        Fitted incremental PCA (all components kept).
    """
    files = get_feat_files(feat_dir)
    ipca = IncrementalPCA()
    batch = []
    for file in tqdm(files):
        _, mat = read_h5_features(file)
        if len(mat.shape) == 1:
            mat = np.expand_dims(mat, 0)
        if mat.sum() == 0:
            continue
        if check_batch(batch):
            batch = np.vstack(batch)
            ipca.partial_fit(X=batch)
            batch = []
        else:
            batch.append(mat)
    return files, ipca
