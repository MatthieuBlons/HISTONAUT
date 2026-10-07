"""Read and save patch coordinates and patch feature vectors.

Coordinates are stored in the h5 dataset `"coords"` (`(N, 4)` xywh at level 0)
and patch encodings in the h5 dataset `"features"` (`(N, D)`); metadata is
stored as dataset attributes.
"""

# General libraries
import json
import h5py
from pathlib import Path

# Data libraries
import numpy as np

def safe_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def read_h5(path, key: str, mode: str = "r") -> tuple[dict, np.ndarray]:
    """Read a dataset and its attributes from an h5 file.

    Parameters
    ----------
    path : str | Path
        Path to the h5 file.
    key : str
        Name of the dataset to read (e.g. `"coords"` or `"features"`).
    mode : str, optional
        Mode passed to `h5py.File`. Default is `"r"`.

    Returns
    -------
    attrs : dict
        Attributes of the dataset.
    data : np.ndarray
        Full content of the dataset.

    Raises
    ------
    KeyError
        If `key` is not a dataset of the file.
    """
    with h5py.File(path, mode) as f:
        attrs = dict(f[key].attrs)
        coords = f[key][:]
    return attrs, coords


def read_h5_coords(coords_path, mode: str = "r") -> tuple[dict, np.ndarray]:
    """Read patch coordinates and their attributes from an h5 file.

    Parameters
    ----------
    coords_path : str | Path
        Path to the h5 file holding a `"coords"` dataset.
    mode : str, optional
        Mode passed to `h5py.File`. Default is `"r"`.

    Returns
    -------
    attrs : dict
        Attributes of the `"coords"` dataset.
    coords : np.ndarray
        Patch coordinates, shape `(N, 4)` xywh at level 0.
    """
    key = "coords"
    attrs, coords = read_h5(coords_path, key, mode)
    return attrs, coords


def read_h5_features(embs_path: str, mode: str = "r") -> tuple[dict, np.ndarray]:
    """Read patch (or slide) encodings and their attributes from an h5 file.

    Parameters
    ----------
    embs_path : str
        Path to the h5 file holding a `"features"` dataset.
    mode : str, optional
        Mode passed to `h5py.File`. Default is `"r"`.

    Returns
    -------
    attrs : dict
        Attributes of the `"features"` dataset (e.g. `"name"`, `"encoder"`).
    features : np.ndarray
        Encodings, shape `(N, D)` for patch features or `(D,)` for a slide
        embedding.
    """
    # Defining key for retrieving the embeddings in h5 file
    key = "features"
    attrs, coords = read_h5(embs_path, key, mode)
    return attrs, coords


def print_attrs(obj, name=None):
    """Print the attributes of an h5 object.

    Parameters
    ----------
    obj : h5py.Group | h5py.Dataset
        Object exposing an `attrs` mapping.
    name : str, optional
        Name printed in the header line. Default is None.

    Notes
    -----
    The `(name, obj)` order is reversed with respect to the callback signature
    expected by `h5py.Group.visititems`.
    """
    print(f"Object: {name}")
    for key, value in obj.attrs.items():
        print(f"    Attribute - {key}: {value}")


def print_dict(dict, name=None):
    """Print the key/value pairs of a dictionary.

    Parameters
    ----------
    dict : dict
        Dictionary to print.
    name : str, optional
        Name printed in the header line; `"Dictionary"` is used when not given.
        Default is None.
    """
    if name:
        print(f"In {name}: ")
    else:
        print(f"In Dictionary: ")
    for key, value in dict.items():
        print(f"    {key}: {value}")


def save_h5(save_path, assets, attributes=None, mode="a"):
    """Save a dictionary of arrays to an h5 file, one dataset per key.


    Parameters
    ----------
    save_path : str | Path
        Path to the h5 file.

    assets : dict[str, np.ndarray]
        Dataset name -> array to write. Arrays of `object` dtype (e.g. strings)
        are stored as UTF-8 strings.

    attributes : dict[str, dict[str, Any]], optional
        Dataset name -> attributes to attach to it. Dictionaries are serialised
        to JSON and `None` values are stored as the string `"None"`. Default is
        None.

    mode : str, optional
        Mode passed to `h5py.File`. Default is `"a"` (append).

    Notes
    -----
    Attributes are only written when a dataset is created, not when it is
    extended. Attributes that cannot be saved are skipped and the exception is
    printed.
    """

    with h5py.File(save_path, mode) as file:
        for key, val in assets.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                if data_type == object:  # when saving arrays of str for instance
                    data_type = h5py.string_dtype(encoding="utf-8")
                chunk_shape = (1,) + data_shape[1:]
                maxshape = (None,) + data_shape[1:]
                dset = file.create_dataset(
                    key,
                    shape=data_shape,
                    maxshape=maxshape,
                    chunks=chunk_shape,
                    dtype=data_type,
                )
                dset[:] = val
                if attributes is not None:
                    if key in attributes.keys():
                        for attr_key, attr_val in attributes[key].items():
                            try:
                                # Serialize if the attribute value is a dictionary
                                if isinstance(attr_val, dict):
                                    attr_val = json.dumps(attr_val)
                                # Serialize Nones
                                elif attr_val is None:
                                    attr_val = "None"
                                dset.attrs[attr_key] = attr_val
                            except Exception as e:
                                print(e)

            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0] :] = val
