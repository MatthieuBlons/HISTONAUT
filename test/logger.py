import pyvips

print(f"pyvips == {pyvips.__version__}")

from typing import Any
import logging
import warnings
import yaml
from pathlib import Path

import torch
from argparse import (
    Namespace,
)

import time


warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)

def configure_logging(verbose: bool = False) -> None:
    """
    Configure console logging.

    Parameters
    ----------
    verbose : bool, default=False
        Use informational logging when enabled and warning-level logging
        otherwise.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

def log_configuration(
    config: dict,
    title: str = "Resolved base configuration",
) -> None:
    """
    Log the contents of a configuration mapping.

    Parameters
    ----------
    config : dict
        Configuration values.

    title : str, default="Resolved base configuration"
        Heading written before the configuration entries.
    """
    LOGGER.info("%s:", title)

    for key, value in config.items():
        LOGGER.info("  %s: %s", key, value)


def load_yaml_config(
    config_path: str | Path | None,
) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str, pathlib.Path, or None
        Path to the YAML file. If None, return an empty configuration.

    Returns
    -------
    dict
        YAML configuration.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.

    TypeError
        If the YAML root object is not a mapping.
    """
    if config_path is None:
        return {}

    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise TypeError("The YAML configuration must contain a mapping at its root.")

    return config

class timetracker(object):
    """
    Stopwatch  timer
    """

    def __init__(self, name=None, verbose=False):
        self.name = name
        self.verbose = verbose
        self.TicToc = self.TicTocGenerator()

    def TicTocGenerator(self):  # add verbose arg
        # Generator that returns time differences
        ti = 0  # initial time
        tf = time.time()  # final time
        while True:
            ti = tf
            tf = time.time()
            yield tf - ti  # returns the time difference

    def toc(self, tempBool=True):
        tempTimeInterval = next(self.TicToc)
        if tempBool:
            if self.verbose:
                print("Elapsed time: %f seconds." % tempTimeInterval)
        return tempTimeInterval

    def tic(self):
        # Records a time in TicToc, marks the beginning of a time interval
        self.toc(False)


def resolve_device(args: Namespace) -> str:
    """
    Resolve the compute device passed to each training run.

    Explicit --device takes precedence. Otherwise, --gpu requests
    CUDA, optionally with a selected device index. If no device option is
    provided, CUDA is used when available and CPU otherwise.

    Parameters
    ----------
    args : argparse.Namespace
        Launcher arguments.

    Returns
    -------
    str
        Resolved PyTorch device string.
    """
    if args.device is not None:
        return args.device

    if args.gpu:
        if not torch.cuda.is_available():
            LOGGER.warning("CUDA was requested but is unavailable; using CPU.")
            return "cpu"

        if args.gpu_id is None:
            return "cuda"

        if args.gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {args.gpu_id} was requested, but only "
                f"{torch.cuda.device_count()} device(s) are available."
            )

        return f"cuda:{args.gpu_id}"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"