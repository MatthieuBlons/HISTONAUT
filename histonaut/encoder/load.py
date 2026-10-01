"""Resolve local checkpoint paths from `model_zoo/local_ckpts.json`.
"""

# General libraries
import os
import json

# Registry lives in model_zoo/
MODEL_ZOO_ROOT = os.path.join(os.path.dirname(__file__), "model_zoo")
REGISTRY_PATH = os.path.join(MODEL_ZOO_ROOT, "local_ckpts.json")
ENCODER_LEVEL_DIRS = {
    "tile": "patch_level",
    "slide": "slide_level",
}


def get_weights_path(encoder_type, encoder_name):
    """Resolve the local weights path of an encoder from the registry.

    Parameters
    ----------
    encoder_type : str
        `"tile"` or `"slide"` (keys of `ENCODER_LEVEL_DIRS`); selects the
        directory relative paths resolve against.
    encoder_name : str
        Encoder key in `local_ckpts.json`.

    Returns
    -------
    path : str
        Absolute path to the weights. A warning is printed if it does not
        exist.

    Raises
    ------
    AssertionError
        If `encoder_type` is not `"tile"` or `"slide"`.
    ValueError
        If `encoder_name` has no (or an empty) entry in the registry.
    FileNotFoundError
        If `local_ckpts.json` does not exist.
    """
    assert (
        encoder_type in ENCODER_LEVEL_DIRS
    ), f"Encoder type must be 'tile' or 'slide', not '{encoder_type}'"
    root = os.path.join(MODEL_ZOO_ROOT, ENCODER_LEVEL_DIRS[encoder_type])
    registry_path = REGISTRY_PATH
    with open(registry_path, "r") as f:
        registry = json.load(f)
    path = registry.get(encoder_name)
    if not path:
        raise ValueError(
            f"Please specify the weights path to '{encoder_name}' in '{registry_path}'"
        )
    path = (
        path if os.path.isabs(path) else os.path.abspath(os.path.join(root, path))
    )  # Make path absolute
    if not os.path.exists(path):
        print(
            f"WARNING: Path at '{path}' does not exist. Please double-check the registry in '{registry_path}'"
        )
    return path


def get_model_path(encoder_type, encoder_name):
    """Resolve the local model (code or directory) path of an encoder.

    Same lookup as `get_weights_path`, used for encoders whose registry entry
    points to a model directory rather than a weights file.

    Parameters
    ----------
    encoder_type : str
        `"tile"` or `"slide"` (keys of `ENCODER_LEVEL_DIRS`); selects the
        directory relative paths resolve against.
    encoder_name : str
        Encoder key in `local_ckpts.json`.

    Returns
    -------
    path : str
        Absolute path to the model. A warning is printed if it does not exist.

    Raises
    ------
    AssertionError
        If `encoder_type` is not `"tile"` or `"slide"`.
    ValueError
        If `encoder_name` has no (or an empty) entry in the registry.
    FileNotFoundError
        If `local_ckpts.json` does not exist.
    """
    assert (
        encoder_type in ENCODER_LEVEL_DIRS
    ), f"Encoder type must be 'tile' or 'slide', not '{encoder_type}'"
    root = os.path.join(MODEL_ZOO_ROOT, ENCODER_LEVEL_DIRS[encoder_type])
    registry_path = REGISTRY_PATH
    with open(registry_path, "r") as f:
        registry = json.load(f)
    path = registry.get(encoder_name)
    if not path:
        raise ValueError(
            f"Please specify the model path to '{encoder_name}' in '{registry_path}'"
        )
    path = (
        path if os.path.isabs(path) else os.path.abspath(os.path.join(root, path))
    )  # Make path absolute
    if not os.path.exists(path):
        print(
            f"WARNING: Path at '{path}' does not exist. Please double-check the registry in '{registry_path}'"
        )
    return path
