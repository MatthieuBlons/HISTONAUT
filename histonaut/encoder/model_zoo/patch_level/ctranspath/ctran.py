"""CTransPath model definition (vendored).

Swin-Tiny transformer whose patch embedding is replaced by a convolutional stem
(`ConvStem`). Requires the external `timm_ctp` package (CTransPath's patched
fork of `timm`). Weights are not loaded here.

Credits to original CTransPath implementation:
https://github.com/Xiyue-Wang/TransPath/blob/main/ctran.py
"""

from timm_ctp.models.layers.helpers import to_2tuple
from timm_ctp import create_model as ctp_create_model
import torch.nn as nn


class ConvStem(nn.Module):
    """Convolutional patch embedding used by CTransPath.

    Two stride-2 `Conv2d` + `BatchNorm2d` + `ReLU` blocks (channels
    `embed_dim // 8` then `embed_dim // 4`) followed by a 1x1 convolution to
    `embed_dim`, i.e. an overall downsampling by 4.

    Parameters
    ----------
    img_size : int | tuple[int, int], optional
        Input image size. Default is 224.
    patch_size : int, optional
        Patch size; must be 4. Default is 4.
    in_chans : int, optional
        Number of input channels (unused, the stem expects 3). Default is 3.
    embed_dim : int, optional
        Embedding dimension; must be a multiple of 8. Default is 768.
    norm_layer : Callable[[int], torch.nn.Module] | None, optional
        Normalisation layer applied after the projection. Default is None
        (`nn.Identity`).
    flatten : bool, optional
        If True, flatten the spatial dimensions to tokens. Default is True.

    Attributes
    ----------
    img_size : tuple[int, int]
        Input image size.
    patch_size : tuple[int, int]
        Patch size.
    grid_size : tuple[int, int]
        Number of patches along each axis.
    num_patches : int
        Total number of patches.
    flatten : bool
        Whether the output is flattened to `(B, N, C)`.
    proj : torch.nn.Sequential
        Convolutional stem.
    norm : torch.nn.Module
        Normalisation layer.

    Raises
    ------
    AssertionError
        If `patch_size` is not 4 or `embed_dim` is not a multiple of 8.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        """Build the convolutional stem.

        Parameters
        ----------
        img_size : int | tuple[int, int], optional
            Input image size. Default is 224.
        patch_size : int, optional
            Patch size; must be 4. Default is 4.
        in_chans : int, optional
            Number of input channels (unused). Default is 3.
        embed_dim : int, optional
            Embedding dimension; must be a multiple of 8. Default is 768.
        norm_layer : Callable[[int], torch.nn.Module] | None, optional
            Normalisation layer. Default is None.
        flatten : bool, optional
            Flatten the output to tokens. Default is True.

        Raises
        ------
        AssertionError
            If `patch_size` is not 4 or `embed_dim` is not a multiple of 8.
        """
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """Embed an image batch into patch tokens.

        Parameters
        ----------
        x : torch.Tensor
            Images of shape `(B, 3, H, W)`; the size is not checked.

        Returns
        -------
        x : torch.Tensor
            Tokens `(B, H/4 * W/4, embed_dim)` if `flatten`, else feature maps
            `(B, embed_dim, H/4, W/4)`.
        """
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def ctranspath(img_size = 224, **kwargs):
    """Create an untrained CTransPath model.

    Parameters
    ----------
    img_size : int, optional
        Input image size. Default is 224.
    **kwargs
        Extra arguments passed to `timm_ctp.create_model`.

    Returns
    -------
    model : torch.nn.Module
        `swin_tiny_patch4_window7_224` with a `ConvStem` patch embedding and
        random weights (`pretrained=False`).
    """
    model = ctp_create_model('swin_tiny_patch4_window7_224', 
                                  embed_layer=ConvStem, 
                                  pretrained=False,
                                  img_size=img_size,
                                  **kwargs)
    return model
