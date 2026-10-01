"""Torch Dataset over the patches of a `SlidePatcher`."""

# Tensor op
from torch.utils.data import Dataset


class PatchSampler(Dataset):
    """Torch Dataset reading the patches selected by a `SlidePatcher`.

    Parameters
    ----------
    patcher : SlidePatcher
        Indexable patch selection returning `(tile, (x, y, w, h))`.

    transform : Callable | None
        Transform applied to each tile (e.g. torchvision eval transforms);
        skipped when falsy.

    Attributes
    ----------
    patcher : SlidePatcher
        Patch selection the samples are read from.
    transform : Callable | None
        Transform applied to each tile.
    """

    def __init__(self, patcher, transform):
        """Store the patcher and the transform.

        Parameters
        ----------
        patcher : SlidePatcher
            Indexable patch selection returning `(tile, (x, y, w, h))`.

        transform : Callable | None
            Transform applied to each tile; skipped when falsy.
        """
        self.patcher = patcher
        self.transform = transform

    def __len__(self):
        """Return the number of patches.

        Returns
        -------
        length : int
            Number of patches in `patcher`.
        """
        return len(self.patcher)

    def __getitem__(self, index):
        """Read one patch and apply the transform.

        Parameters
        ----------
        index : int
            Index of the patch in `patcher`.

        Returns
        -------
        tile : PIL.Image.Image | torch.Tensor
            Patch image, transformed if `transform` is set.

        coords : tuple[int, int, int, int]
            Patch coordinates `(x, y, w, h)` at level 0.
        """
        tile, (x, y, w, h) = self.patcher[index]
        if self.transform:
            tile = self.transform(tile)
        return tile, (x, y, w, h)
