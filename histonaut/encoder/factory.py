"""Build patch-level and slide-level encoders.

Mostly a fork of TRIDENT
(https://github.com/mahmoodlab/TRIDENT/blob/main/trident/patch_encoder_models/load.py).

Each encoder is a `torch.nn.Module` whose `_build()` method creates the model
(TRIDENT-like design).

Notes
-----
Weights are read from local checkpoints registered in
`model_zoo/local_ckpts.json` (see `histonaut.encoder.load`) when available;
otherwise they are downloaded from the Hugging Face Hub. Gated
models need an access grant and a registered HF token.
Model-specific third-party packages are imported lazily inside `_build()`,
so they are only required for the encoders that use them.
"""

# General libraries
import os
import traceback
from abc import abstractmethod
from functools import partial
import sys

# Tensor op
import torch
from torchinfo import summary as model_summary

# DL libraries
import timm

# Project modules
from histonaut.encoder.load import get_weights_path, get_model_path

# ---------------------------
# Patch-level
# ---------------------------


def get_encoder_mapping():
    """Return the registry of patch encoders.

    Returns
    -------
    dict[str, type | functools.partial]
        Mapping from patch encoder name to its `BasePatchEncoder`.
    """

    encoder_mapping = {
        "resnet50": ResNet50InferenceEncoder,
        "ctranspath": CTransPathInferenceEncoder,
        "prov_gigapath": GigaPathInferenceEncoder,
        "uni_v1": UNIInferenceEncoder,
        "uni_v2": UNIv2InferenceEncoder,
        "univ2_seal_vision": partial(
            SealVisionInferenceEncoder, backbone="univ2"
        ),  # not tested yet
        "conch_v1": Conchv1InferenceEncoder,
        "conch_v15": Conchv15InferenceEncoder,
        "conch_seal_vision": partial(
            SealVisionInferenceEncoder, backbone="conch"
        ),  # not tested yet
        "hoptimus0": HOptimus0InferenceEncoder,
        "h0mini": H0MiniInferenceEncoder,
        "hoptimus1": HOptimus1InferenceEncoder,
        "PROiFHESi_H1": PROiFHESiH1InferenceEncoder,
        "virchow": VirchowInferenceEncoder,
        "virchow2": Virchow2InferenceEncoder,
        "phikon": PhikonInferenceEncoder,
        "phikon_v2": Phikonv2InferenceEncoder,
        "musk": MuskInferenceEncoder,
    }

    return encoder_mapping


def fetch_embeddings_dim(tile_encoder):
    """Return the known embedding dimension of a patch encoder.

    Parameters
    ----------
    tile_encoder : str
        Patch encoder name.

    Returns
    -------
    int | None
        Embedding dimension
    """
    embedding_dim = None
    if tile_encoder == "conch_v1":
        embedding_dim = 768
    elif tile_encoder == "conch_v15":
        embedding_dim = 768
    elif tile_encoder == "uni_v1":
        embedding_dim = 1024
    elif tile_encoder == "uni_v2":
        embedding_dim = 1536
    elif tile_encoder == "ctranspath":
        embedding_dim = 768
    elif tile_encoder == "phikon":
        embedding_dim = 768
    elif tile_encoder == "phikon_v2":
        embedding_dim = 1024
    elif tile_encoder == "resnet50":
        embedding_dim = 1024
    elif tile_encoder == "gigapath":
        embedding_dim = 1536
    elif tile_encoder == "virchow":
        embedding_dim = 2560
    elif tile_encoder == "virchow2":
        embedding_dim = 2560
    elif tile_encoder == "hoptimus0":
        embedding_dim = 1536
    elif tile_encoder == "hoptimus1":
        embedding_dim = 1536
    elif tile_encoder == "musk":
        embedding_dim = 1024
    else:
        print(
            f"WARNING: Tile embedding dimension is not known a priori for {tile_encoder}. Setting to None."
        )
    return embedding_dim


def patch_encoder_factory(model_name, **kwargs):
    """Build a patch encoder by name.

    Parameters
    ----------
    model_name : str
        Patch encoder name, one of the keys of `get_encoder_mapping()`.

    **kwargs
        Keyword arguments forwarded to the encoder constructor (and then to its
        `_build`).

    Returns
    -------
    BasePatchEncoder
        The instantiated patch encoder.

    Raises
    ------
    ValueError
        If `model_name` is not a key of `get_encoder_mapping()`.
    """

    # Retrieving encoder mapping dict
    encoder_mapping = get_encoder_mapping()

    if model_name in encoder_mapping.keys():
        return encoder_mapping[model_name](**kwargs)
    else:
        raise ValueError(f"Unknown encoder name {model_name}")


class BasePatchEncoder(torch.nn.Module):
    """Base class for patch-level (tile) encoders.

    `_build` is called once at construction; it creates the model, its evaluation
    transforms and the inference precision.

    Parameters
    ----------
    **build_kwargs
        Keyword arguments forwarded to `_build`.

    Attributes
    ----------
    enc_name : str | None
        Encoder name, set by `_build`.
    model : torch.nn.Module
        Underlying network.
    eval_transforms : Callable
        Transform turning a PIL image into a normalised `(3, H, W)` tensor.
    precision : torch.dtype
        Recommended inference dtype (e.g. `torch.float16`).
    """

    def __init__(self, **build_kwargs):
        """Build the encoder through `_build` (see class docstring)."""
        super().__init__()
        self.enc_name = None
        self.model, self.eval_transforms, self.precision = self._build(**build_kwargs)

    def forward(self, x):
        """Encode a batch of patches.

        Can be overridden if the model requires a special forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of transformed patches, shape `(B, 3, H, W)`.

        Returns
        -------
        torch.Tensor
            Patch embeddings, shape `(B, D)`.
        """
        z = self.model(x)
        return z

    def forward_features(self, x):
        """Return the backbone features of a batch of patches.

        Parameters
        ----------
        x : torch.Tensor
            Batch of transformed patches, shape `(B, 3, H, W)`.

        Returns
        -------
        torch.Tensor
            Output of `self.model.forward_features(x)` (typically token features for
            `timm` models).
        """
        z = self.model.forward_features(x)
        return z

    def print_summary(self, depth=4, verbose=0):
        """Print a `torchinfo` summary of the model.

        Parameters
        ----------
        depth : int, optional
            Depth of nested layers to display. Default is 4.

        verbose : int, optional
            `torchinfo` verbosity level. Default is 0.
        """
        model_summary(self.model, depth=depth, verbose=verbose)

    @abstractmethod
    def _build(self, **build_kwargs):
        """Create the model, its transforms and precision (to implement in subclasses).

        Parameters
        ----------
        **build_kwargs
            Encoder-specific options.

        Returns
        -------
        model : torch.nn.Module
            Patch encoder network.

        eval_transforms : Callable
            Evaluation transforms.

        precision : torch.dtype
            Inference dtype.
        """
        pass


class CustomInferenceEncoder(BasePatchEncoder):
    """Wrap an already built model as a patch encoder.

    Parameters
    ----------
    enc_name : str
        Encoder name.

    model : torch.nn.Module
        Patch encoder network.

    transforms : Callable
        Evaluation transforms.

    precision : torch.dtype
        Inference dtype.

    **kwargs
        Ignored.
    """

    def __init__(self, enc_name, model, transforms, precision, **kwargs):
        """Store the given model, transforms and precision (see class docstring)."""
        super().__init__()
        self.enc_name = enc_name
        self.model = model
        self.eval_transforms = transforms
        self.precision = precision

    def _build(self):
        """Return placeholders; attributes are set in `__init__`.

        Returns
        -------
        tuple[None, None, None]
            `(None, None, None)`.
        """
        return None, None, None


class ResNet50InferenceEncoder(BasePatchEncoder):
    """ResNet-50 (ImageNet, `timm` `resnet50.tv_in1k`) patch encoder.

    Uses the output of the third residual stage (`out_indices=[3]`), optionally
    average-pooled. Embedding dim: 1024 (`fetch_embeddings_dim`).

    Parameters
    ----------
    pretrained : bool, optional
        Load ImageNet weights. Default is True.

    timm_kwargs : dict[str, Any], optional
        Extra `timm.create_model` arguments. Default is
        `{"features_only": True, "out_indices": [3], "num_classes": 0}`.

    image_size : int, optional
        Resize target of the eval transforms. Default is 224.

    pool : bool, optional
        Apply global average pooling in `forward`. Default is True.
    """

    def _build(
        self,
        pretrained=True,
        timm_kwargs={"features_only": True, "out_indices": [3], "num_classes": 0},
        image_size=224,
        pool=True,
        **kwargs,
    ):
        """Build the `timm` ResNet-50 with ImageNet normalisation.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)`, precision is `torch.float32`.
        """
        import timm
        from torchvision.transforms import Compose, Resize, InterpolationMode

        self.enc_name = "resnet50"
        model = timm.create_model(
            "resnet50.tv_in1k", pretrained=pretrained, **timm_kwargs
        )
        eval_transform = get_eval_transforms(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            target_img_size=image_size,
            interpolation=InterpolationMode.BILINEAR,
            max_size=None,
            antialias=True,
        )

        precision = torch.float32
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None

        return model, eval_transform, precision

    def forward(self, x):
        """Encode patches, pooling the feature map if `pool` is set.

        Parameters
        ----------
        x : torch.Tensor
            Batch of patches, shape `(B, 3, H, W)`.

        Returns
        -------
        torch.Tensor
            `(B, 1024)` if pooled, else the feature map `(B, 1024, h, w)`.
        """
        out = self.forward_features(x)
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out

    def forward_features(self, x):
        """Return the stage-3 feature map.

        Parameters
        ----------
        x : torch.Tensor
            Batch of patches, shape `(B, 3, H, W)`.

        Returns
        -------
        torch.Tensor
            Feature map, shape `(B, 1024, h, w)` (single `features_only` output).
        """
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        return out


class CTransPathInferenceEncoder(BasePatchEncoder):
    """CTransPath patch encoder (Swin transformer, CHIEF weights).

    Local weights from `local_ckpts.json`; if missing, `CHIEF_CTransPath.pth`
    is downloaded from the `MahmoodLab/hest-bench` HF dataset. Requires the
    `timm_ctp` package (vendored `model_zoo/patch_level/ctranspath/ctran.py`).

    Embedding dim: 768.
    """

    def _build(self, **kwargs):
        """Build CTransPath with its head removed and load its weights.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)`, precision is `torch.float32`.

        Raises
        ------
        Exception
            If the CTransPath model cannot be imported.
        """
        from torchvision.transforms import Compose, Resize, InterpolationMode
        from torch import nn

        try:
            from histonaut.encoder.model_zoo.patch_level.ctranspath.ctran import (
                ctranspath,
            )
        except:
            traceback.print_exc()
            raise Exception(
                "Failed to import CTransPath model, make sure timm_ctp is installed. `pip install timm_ctp`"
            )

        self.enc_name = "ctranspath"
        weights_path = get_weights_path("tile", self.enc_name)
        model = ctranspath(img_size=224)
        model.head = nn.Identity()

        weights_dir = os.path.dirname(weights_path)
        os.makedirs(weights_dir, exist_ok=True)

        if not os.path.isfile(weights_path):
            from huggingface_hub import hf_hub_download
            import shutil

            downloaded_file_path = hf_hub_download(
                repo_id="MahmoodLab/hest-bench",
                repo_type="dataset",
                filename="CHIEF_CTransPath.pth",
                subfolder="fm_v1/ctranspath",
                local_dir=weights_dir,
            )
            shutil.move(downloaded_file_path, weights_path)
            subfolder_path = os.path.join(weights_dir, "fm_v1", "ctranspath")
            if os.path.exists(subfolder_path):
                os.removedirs(subfolder_path)

        state_dict = torch.load(weights_path, weights_only=True)["model"]
        state_dict = {
            key: val for key, val in state_dict.items() if "attn_mask" not in key
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        assert (
            len(unexpected) == 0
        ), f"Unexpected keys found in state dict: {unexpected}"
        assert missing == [
            "layers.0.blocks.1.attn_mask",
            "layers.1.blocks.1.attn_mask",
            "layers.2.blocks.1.attn_mask",
            "layers.2.blocks.3.attn_mask",
            "layers.2.blocks.5.attn_mask",
        ], f"Unexpected missing keys: {missing}"

        eval_transform = get_eval_transforms(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            target_img_size=224,
            interpolation=InterpolationMode.BILINEAR,
            max_size=None,
            antialias=True,
        )

        precision = torch.float32

        return model, eval_transform, precision


class UNIInferenceEncoder(BasePatchEncoder):
    """UNI patch encoder (ViT-L/16, HF `MahmoodLab/uni`, gated).

    Embedding dim: 1024.

    Parameters
    ----------
    timm_kwargs : dict[str, Any], optional
        Extra `timm.create_model` arguments. Default is
        `{"dynamic_img_size": True, "num_classes": 0, "init_values": 1.0}`.
    """

    def _build(
        self,
        timm_kwargs={"dynamic_img_size": True, "num_classes": 0, "init_values": 1.0},
        **kwargs,
    ):
        """Download UNI from the HF Hub through `timm`.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)`, precision is `torch.float16`.

        Raises
        ------
        Exception
            If the model cannot be downloaded (no access or no HF token).
        """
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = "uni_v1"

        try:
            print("try to create uni_v1 model")
            model = timm.create_model(
                "hf-hub:MahmoodLab/uni", pretrained=True, **timm_kwargs
            )
            print("ok")
        except:
            traceback.print_exc()
            raise Exception(
                "Failed to download UNI model, make sure that you were granted access and that you correctly registered your token"
            )

        eval_transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        precision = torch.float16
        return model, eval_transform, precision


class UNIv2InferenceEncoder(BasePatchEncoder):
    """UNI2-h patch encoder (ViT-H/14 with registers, HF `MahmoodLab/UNI2-h`, gated).

    Embedding dim: 1536.
    """

    def _build(self, **kwargs):
        """Download UNI2-h from the HF Hub through `timm`.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)`, precision is `torch.bfloat16`.

        Raises
        ------
        Exception
            If the model cannot be downloaded (no access or no HF token).
        """
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = "uni_v2"

        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        try:
            model = timm.create_model(
                "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
            )
        except:
            traceback.print_exc()
            raise Exception(
                "Failed to download UNI model, make sure that you were granted access and that you correctly registered your token"
            )

        eval_transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        precision = torch.bfloat16
        return model, eval_transform, precision


class SealVisionInferenceEncoder(BasePatchEncoder):
    """Vision encoder fine-tuned with SEAL (HF `MahmoodLab/SEAL`, gated).

    Loads the official SEAL vision checkpoint using SEAL's own model
    construction/loading code (`seal.seal_factory`), preserving the LoRA /
    PatchRecEncoder structure expected by the checkpoint. Requires the
    MahmoodLab/SEAL repository to be installed. Registered as
    `univ2_seal_vision` and `conch_seal_vision`. Output shape: `(B, 1536)`.

    Parameters
    ----------
    backbone : str
        SEAL backbone, `"univ2"` or `"conch"` (as used in `get_encoder_mapping`).
    **build_kwargs
        Not forwarded to `_build` (see Notes).

    Notes
    -----
    `_build` options (`source`, `hf_repo_id`, `hf_revision`, `hf_token`,
    `cache_dir`) are only reachable through their defaults, since `__init__`
    calls `super().__init__()` without `build_kwargs`.
    """

    def __init__(self, backbone, **build_kwargs):
        """Store `backbone` then build the encoder (see class docstring)."""
        self.backbone = backbone
        super().__init__()

    def _build(
        self,
        source="auto",
        hf_repo_id="MahmoodLab/SEAL",
        hf_revision="main",
        hf_token=None,
        cache_dir=None,
        **kwargs,
    ):
        """Load the SEAL vision encoder for `self.backbone`.

        Parameters
        ----------
        source : str, optional
            Checkpoint source passed to `seal_factory`. Default is `"auto"`.
        hf_repo_id : str, optional
            HF repository id. Default is `"MahmoodLab/SEAL"`.
        hf_revision : str, optional
            HF revision. Default is `"main"`.
        hf_token : str | None, optional
            HF token; read with `huggingface_hub.get_token()` if None. Default is None.
        cache_dir : str | None, optional
            HF cache directory. Default is None.
        **kwargs
            Ignored.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)` as returned by `seal_factory`; the
            model is set to eval mode.

        Raises
        ------
        ImportError
            If the `seal` package is not installed.
        RuntimeError
            If loading fails or SEAL returns no vision model for `self.backbone`.
        """
        self.enc_name = f"{self.backbone}_seal_vision"

        if hf_token is None:
            from huggingface_hub import get_token

            hf_token = get_token()

        try:
            from seal import seal_factory
        except ImportError:
            traceback.print_exc()
            raise ImportError(
                f"SEAL is required for `{self.enc_name}`. "
                "Install the MahmoodLab/SEAL repository first."
            )

        try:
            print(f"backbone = {self.backbone}")
            (model, eval_transform, precision), _ = seal_factory(
                backbone=self.backbone,
                source=source,
                hf_repo_id=hf_repo_id,
                hf_revision=hf_revision,
                hf_token=hf_token,
                cache_dir=cache_dir,
            )  # seal_factory also returns a gene_encoder
        except Exception:
            traceback.print_exc()
            raise RuntimeError(
                "Failed to load the SEAL vision encoder. "
                "Make sure you have access to MahmoodLab/SEAL on "
                "Hugging Face and are authenticated."
            )

        if model is None:
            raise RuntimeError(
                f"SEAL did not return a vision model for backbone='{self.backbone}'. "
                f"Check that the SEAL vision checkpoint is available for backbone='{self.backbone}'."
            )

        model.eval()

        return model, eval_transform, precision


class Conchv1InferenceEncoder(BasePatchEncoder):
    """CONCH v1 patch encoder (ViT-B/16, HF `MahmoodLab/conch`, gated).

    Requires the `conch` package (`pip install
    git+https://github.com/Mahmoodlab/CONCH.git`). Embedding dim: 768.

    Parameters
    ----------
    with_proj : bool, optional
        Apply the contrastive projection head in `encode_image`. Default is False.
    normalize : bool, optional
        L2-normalise the embeddings. Default is False.
    """

    def _build(self, with_proj=False, normalize=False, **kwargs):
        """Load CONCH v1 and its preprocessing from the HF Hub.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)`, precision is `torch.float32`.

        Raises
        ------
        Exception
            If `conch` is not installed or the model cannot be downloaded.
        """
        self.enc_name = "conch_v1"
        self.with_proj = with_proj
        self.normalize = normalize
        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except:
            traceback.print_exc()
            raise Exception(
                "Please install CONCH `pip install git+https://github.com/Mahmoodlab/CONCH.git`"
            )

        try:
            model, preprocess = create_model_from_pretrained(
                "conch_ViT-B-16", "hf_hub:MahmoodLab/conch"
            )
        except:
            traceback.print_exc()
            raise Exception(
                "Failed to download CONCH model, make sure that you were granted access and that you correctly registered your token"
            )

        eval_transform = preprocess
        precision = torch.float32

        return model, eval_transform, precision

    def forward(self, x):
        """Encode patches with `model.encode_image`.

        Parameters
        ----------
        x : torch.Tensor
            Batch of patches, shape `(B, 3, H, W)`.

        Returns
        -------
        torch.Tensor
            Image embeddings, shape `(B, D)`.
        """
        return self.model.encode_image(
            x, proj_contrast=self.with_proj, normalize=self.normalize
        )


class Conchv15InferenceEncoder(BasePatchEncoder):
    """CONCH v1.5 patch encoder (vendored in `model_zoo/patch_level/conchv1_5`).

    Local weights from `local_ckpts.json`; if missing, downloaded from HF
    `MahmoodLab/conchv1_5` (gated) and saved to the local path. Embedding dim: 768.

    Parameters
    ----------
    img_size : int, optional
        Input image size. Default is 448.
    """

    def _build(self, img_size=448, **kwargs):
        """Load CONCH v1.5 from local weights or the HF Hub.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)`, precision is `torch.float16`.
        """

        from histonaut.encoder.model_zoo.patch_level.conchv1_5.conchv1_5 import (
            create_model_from_pretrained,
        )

        self.enc_name = "conch_v15"
        weights_path = get_weights_path("tile", self.enc_name)
        weights_dir = os.path.dirname(weights_path)
        os.makedirs(weights_dir, exist_ok=True)

        if os.path.isfile(weights_path):
            model, eval_transform = create_model_from_pretrained(
                checkpoint_path=weights_path, img_size=img_size
            )
        else:
            print("Downloading model weights from HuggingFace...")
            model, eval_transform = create_model_from_pretrained(
                checkpoint_path="hf_hub:MahmoodLab/conchv1_5", img_size=img_size
            )
            torch.save(model.state_dict(), weights_path)

        precision = torch.float16
        return model, eval_transform, precision


class GigaPathInferenceEncoder(BasePatchEncoder):
    """Prov-GigaPath patch encoder (HF `prov-gigapath/prov-gigapath`, via `timm`).

    Requires `timm==0.9.16`. Embedding dim: 1536.

    Parameters
    ----------
    timm_kwargs : dict[str, Any], optional
        Extra `timm.create_model` arguments. Default is `{}`.
    """

    def _build(self, timm_kwargs={}, **kwargs):
        """Download Prov-GigaPath from the HF Hub through `timm`.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)`, precision is `torch.float32`.

        Raises
        ------
        AssertionError
            If the installed `timm` version is not 0.9.16.
        """
        import timm

        assert (
            timm.__version__ == "0.9.16"
        ), f"Gigapath requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        from torchvision import transforms

        self.enc_name = "prov_gigapath"

        model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True, **timm_kwargs
        )

        eval_transform = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        precision = torch.float32
        return model, eval_transform, precision


class VirchowInferenceEncoder(BasePatchEncoder):
    """Virchow patch encoder (ViT-H/14, HF `paige-ai/Virchow`, gated).

    Embedding dim: 2560 (class token concatenated with the mean patch token).

    Parameters
    ----------
    return_cls : bool, optional
        Return only the class token instead of the concatenation. Default is
        False.
    timm_kwargs : dict[str, Any], optional
        Extra `timm.create_model` arguments. Default is
        `{"mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU}`.
    """

    import timm

    def _build(
        self,
        return_cls=False,
        timm_kwargs={"mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU},
        **kwargs,
    ):
        """Download Virchow from the HF Hub through `timm`.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)`, precision is `torch.float16`.
        """
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = "virchow"

        model = timm.create_model(
            "hf-hub:paige-ai/Virchow", pretrained=True, **timm_kwargs
        )
        eval_transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        precision = torch.float16
        self.return_cls = return_cls

        return model, eval_transform, precision

    def forward(self, x):
        """Encode patches as class token or `[class token, mean patch tokens]`.

        Parameters
        ----------
        x : torch.Tensor
            Batch of patches, shape `(B, 3, H, W)`.

        Returns
        -------
        torch.Tensor
            `(B, D)` if `return_cls`, else `(B, 2 * D)`, with `D` the token dim.
        """
        output = self.model(x)
        class_token = output[:, 0]

        if self.return_cls:
            return class_token
        else:
            patch_tokens = output[:, 1:]
            embeddings = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            return embeddings


class Virchow2InferenceEncoder(BasePatchEncoder):
    """Virchow2 patch encoder (ViT-H/14 with registers, HF `paige-ai/Virchow2`, gated).

    Embedding dim: 2560 (class token concatenated with the mean patch token).

    Parameters
    ----------
    return_cls : bool, optional
        Return only the class token instead of the concatenation. Default is
        False.
    timm_kwargs : dict[str, Any], optional
        Extra `timm.create_model` arguments. Default is
        `{"mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU}`.
    """

    import timm

    def _build(
        self,
        return_cls=False,
        timm_kwargs={"mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU},
        **kwargs,
    ):
        """Download Virchow2 from the HF Hub through `timm`.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)`, precision is `torch.float16`.
        """
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = "virchow2"

        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2", pretrained=True, **timm_kwargs
        )
        eval_transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        precision = torch.float16
        self.return_cls = return_cls

        return model, eval_transform, precision

    def forward(self, x):
        """Encode patches as class token or `[class token, mean patch tokens]`.

        Tokens 1 to 4 (registers) are skipped; patch tokens start at index 5.

        Parameters
        ----------
        x : torch.Tensor
            Batch of patches, shape `(B, 3, H, W)`.

        Returns
        -------
        torch.Tensor
            `(B, D)` if `return_cls`, else `(B, 2 * D)`, with `D` the token dim.
        """
        output = self.model(x)

        class_token = output[:, 0]
        if self.return_cls:
            return class_token

        patch_tokens = output[:, 5:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        return embedding


class HOptimus0InferenceEncoder(BasePatchEncoder):
    """H-optimus-0 patch encoder (HF `bioptimus/H-optimus-0`, via `timm`).

    Requires `timm==0.9.16`. Embedding dim: 1536.

    Parameters
    ----------
    timm_kwargs : dict[str, Any], optional
        Extra `timm.create_model` arguments. Default is
        `{"init_values": 1e-5, "dynamic_img_size": False}`.
    """

    def _build(
        self, timm_kwargs={"init_values": 1e-5, "dynamic_img_size": False}, **kwargs
    ):
        """Download H-optimus-0 from the HF Hub through `timm`.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)`, precision is `torch.float16`.

        Raises
        ------
        AssertionError
            If the installed `timm` version is not 0.9.16.
        """
        import timm

        assert (
            timm.__version__ == "0.9.16"
        ), f"H-Optimus requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        from torchvision import transforms

        self.enc_name = "hoptimus0"

        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0", pretrained=True, **timm_kwargs
        )

        eval_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )

        precision = torch.float16
        return model, eval_transform, precision


class H0MiniInferenceEncoder(BasePatchEncoder):
    """H0-mini patch encoder (HF `bioptimus/H0-mini`, via `timm`).

    Parameters
    ----------
    timm_kwargs : dict[str, Any], optional
        Extra `timm.create_model` arguments. Default is
        `{"mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU}`.
    """

    def _build(
        self,
        timm_kwargs={"mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU},
        **kwargs,
    ):
        """Download H0-mini from the HF Hub through `timm`.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)`, precision is `torch.float32`.
        """
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = "h0mini"

        model = timm.create_model(
            "hf-hub:bioptimus/H0-mini", pretrained=True, **timm_kwargs
        )
        eval_transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )

        precision = torch.float32
        return model, eval_transform, precision


class HOptimus1InferenceEncoder(BasePatchEncoder):
    """H-optimus-1 patch encoder (HF `bioptimus/H-optimus-1`, via `timm`, gated).

    Embedding dim: 1536.

    Parameters
    ----------
    timm_kwargs : dict[str, Any], optional
        Extra `timm.create_model` arguments. Default is
        `{"init_values": 1e-5, "dynamic_img_size": False}`.
    """

    def _build(
        self, timm_kwargs={"init_values": 1e-5, "dynamic_img_size": False}, **kwargs
    ):
        """Download H-optimus-1 from the HF Hub through `timm`.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)`, precision is `torch.float16`.
        """
        import timm
        from torchvision import transforms

        self.enc_name = "hoptimus1"

        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1", pretrained=True, **timm_kwargs
        )

        eval_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )

        precision = torch.float16
        return model, eval_transform, precision


class PROiFHESiH1InferenceEncoder(BasePatchEncoder):
    """PROiFHESi_H1 patch encoder (private model, local checkpoint only).

    Loaded with `load_model_from_path` from
    `model_zoo/patch_level/PROiFHESi_H1/model.py`, path from `local_ckpts.json`.

    Parameters
    ----------
    device : str, optional
        Device to load the model on (`"cuda"` if available). Default is `"cuda"`.
    embeddings : bool, optional
        Return the backbone (embeddings, `enc_name="PROiFHESi_H1_emb"`) instead
        of the full network (outputs, `enc_name="PROiFHESi_H1_out"`). Default is
        False.

    Notes
    -----
    The lazy import is broken on purpose: `model_zoo/patch_level/PROiFHESi_H1/`
    does not exist yet, so building this encoder raises `ModuleNotFoundError`.
    """

    def _build(self, **kwargs):
        """Load the PROiFHESi_H1 network (or its backbone) from the local checkpoint.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)` taken from the loaded network.
        """
        from histonaut.encoder.model_zoo.patch_level.PROiFHESi_H1.model import (
            HESingIF,  # to change name
            load_model_from_path,
        )

        self.device = (
            kwargs.get("device", "cuda")
            if torch.cuda.is_available()
            else kwargs.get("device", "cpu")
        )
        self.how = kwargs.get("embeddings", False)
        model_path = get_model_path("tile", "PROiFHESi_H1")
        # use load_model_from_path?
        model = load_model_from_path(model_path=model_path, device=self.device)
        network = model.network

        if self.how:
            self.enc_name = "PROiFHESi_H1_emb"
            return network.backbone, network.transform, network.precision["backbone"]
        else:
            self.enc_name = "PROiFHESi_H1_out"
            return network, network.transform, model.precision["backbone"]


class PhikonInferenceEncoder(BasePatchEncoder):
    """Phikon patch encoder (ViT-B/16, HF `owkin/phikon`, via `transformers`).

    Local weights from `local_ckpts.json`; if missing, downloaded from the HF Hub
    and saved locally. Embedding dim: 768 (class token).
    """

    def _build(self, **kwargs):
        """Load Phikon `ViTModel` (no pooling layer) from local weights or the HF Hub.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)`, precision is `torch.float32`.
        """

        from transformers import ViTModel
        from torchvision.transforms import Compose, Resize, InterpolationMode

        self.enc_name = "phikon"
        weights_path = get_weights_path("tile", self.enc_name)

        if os.path.exists(weights_path):
            model = ViTModel.from_pretrained(weights_path, add_pooling_layer=False)
        else:
            model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
            os.makedirs(weights_path, exist_ok=True)
            model.save_pretrained(weights_path)
        eval_transform = get_eval_transforms(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            target_img_size=224,
            interpolation=InterpolationMode.BILINEAR,
            max_size=None,
            antialias=True,
        )
        precision = torch.float32
        return model, eval_transform, precision

    def forward(self, x):
        """Encode patches as the class token of the last hidden state.

        Parameters
        ----------
        x : torch.Tensor
            Batch of patches, shape `(B, 3, 224, 224)`.

        Returns
        -------
        torch.Tensor
            Class-token embeddings, shape `(B, 768)`.
        """
        out = self.forward_features(x)
        out = out.last_hidden_state[:, 0, :]
        return out

    def forward_features(self, x):
        """Run the `ViTModel` on the patches.

        Parameters
        ----------
        x : torch.Tensor
            Batch of patches, shape `(B, 3, 224, 224)`.

        Returns
        -------
        transformers.modeling_outputs.BaseModelOutputWithPooling
            Model output; `last_hidden_state` has shape `(B, N_tokens, 768)`.
        """
        out = self.model(pixel_values=x)
        return out


class Phikonv2InferenceEncoder(BasePatchEncoder):
    """Phikon-v2 patch encoder (ViT-L/16, HF `owkin/phikon-v2`, via `transformers`).

    Local weights from `local_ckpts.json`; if missing, downloaded from the HF Hub
    and saved locally. Embedding dim: 1024 (class token).
    """

    def _build(self, **kwargs):
        """Load Phikon-v2 `AutoModel` from local weights or the HF Hub.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)`, precision is `torch.float32`.
        """

        from transformers import AutoModel
        import torchvision.transforms as T

        self.enc_name = "phikon_v2"
        weights_path = get_weights_path("tile", self.enc_name)

        if os.path.exists(weights_path):
            model = AutoModel.from_pretrained(weights_path)
        else:
            model = AutoModel.from_pretrained("owkin/phikon-v2")
            os.makedirs(weights_path, exist_ok=True)
            model.save_pretrained(weights_path)

        eval_transform = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),  # Normalize with specified mean and std
            ]
        )

        precision = torch.float32
        return model, eval_transform, precision

    def forward(self, x):
        """Encode patches as the class token of the last hidden state.

        Parameters
        ----------
        x : torch.Tensor
            Batch of patches, shape `(B, 3, 224, 224)`.

        Returns
        -------
        torch.Tensor
            Class-token embeddings, shape `(B, 1024)`.
        """
        out = self.model(x)
        out = out.last_hidden_state[:, 0, :]
        return out


class MuskInferenceEncoder(BasePatchEncoder):
    """MUSK patch encoder (`musk_large_patch16_384`, HF `xiangjx/musk`, gated).

    Requires the `musk` package and `fairscale` (`pip install fairscale
    git+https://github.com/lilab-stanford/MUSK`). Embedding dim: 1024.

    Parameters
    ----------
    inference_aug : bool, optional
        Use test-time multiscale augmentation. Default is False to allow for fair
        comparison with other models.
    with_proj : bool, optional
        Apply the projection head. Default is False.
    out_norm : bool, optional
        L2-normalise the output. Default is False.
    return_global : bool, optional
        Return the global (class) representation. Default is True.
    """

    def _build(
        self,
        inference_aug=False,
        with_proj=False,
        out_norm=False,
        return_global=True,
        **kwargs,
    ):
        """Create MUSK with `timm` and load its HF weights.

        Returns
        -------
        tuple[torch.nn.Module, Callable, torch.dtype]
            `(model, eval_transforms, precision)`, precision is `torch.float16`.

        Raises
        ------
        Exception
            If `musk` is not installed or the model cannot be downloaded.
        """
        self.enc_name = "musk"
        self.inference_aug = inference_aug
        self.with_proj = with_proj
        self.out_norm = out_norm
        self.return_global = return_global

        try:
            from musk import utils, modeling
        except:
            traceback.print_exc()
            raise Exception(
                "Please install MUSK `pip install fairscale git+https://github.com/lilab-stanford/MUSK`"
            )

        try:
            from timm.models import create_model

            model = create_model("musk_large_patch16_384")
            utils.load_model_and_may_interpolate(
                "hf_hub:xiangjx/musk", model, "model|module", ""
            )
        except:
            traceback.print_exc()
            raise Exception(
                "Failed to download MUSK model, make sure that you were granted access and that you correctly registered your token"
            )

        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        from torchvision.transforms import Compose, Resize, InterpolationMode

        eval_transform = get_eval_transforms(
            IMAGENET_INCEPTION_MEAN,
            IMAGENET_INCEPTION_STD,
            target_img_size=384,
            center_crop=True,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        precision = torch.float16

        return model, eval_transform, precision

    def forward(self, x):
        """Encode patches and keep the vision output.

        Parameters
        ----------
        x : torch.Tensor
            Batch of patches, shape `(B, 3, 384, 384)`.

        Returns
        -------
        torch.Tensor
            Vision embeddings, shape `(B, D)`.
        """
        return self.model(
            image=x,
            with_head=self.with_proj,
            out_norm=self.out_norm,
            ms_aug=self.inference_aug,
            return_global=self.return_global,
        )[
            0
        ]  # Forward pass yields (vision_cls, text_cls). We only need vision_cls.


def get_eval_transforms(
    mean, std, target_img_size=-1, center_crop=False, **resize_kwargs
):
    """Build evaluation transforms (resize, crop, to tensor, normalise).

    Parameters
    ----------
    mean : Sequence[float] | None
        Per-channel normalisation mean. No normalisation if `mean` or `std` is
        None.
    std : Sequence[float] | None
        Per-channel normalisation standard deviation.
    target_img_size : int, optional
        Resize target; no resize if not positive. Default is -1.
    center_crop : bool, optional
        Center-crop to `target_img_size` after resizing. Default is False.
    **resize_kwargs
        Keyword arguments forwarded to `torchvision.transforms.Resize`
        (e.g. `interpolation`, `max_size`, `antialias`).

    Returns
    -------
    torchvision.transforms.Compose
        Composed transforms mapping a PIL image to a `(3, H, W)` tensor.

    Raises
    ------
    AssertionError
        If `center_crop` is True and `target_img_size` is not positive.
    """

    from torchvision import transforms

    tforms = []

    if target_img_size > 0:
        tforms.append(transforms.Resize(target_img_size, **resize_kwargs))
    if center_crop:
        assert target_img_size > 0, "target_img_size must be set if center_crop is True"
        tforms.append(transforms.CenterCrop(target_img_size))

    tforms.append(transforms.ToTensor())
    if mean is not None and std is not None:
        tforms.append(transforms.Normalize(mean, std))
    tforms = transforms.Compose(tforms)

    return tforms


# ---------------------------
# Slide-level
# ---------------------------

# Slide encoder name -> patch encoder whose features it expects as input.
SLIDE_TO_TILE_ENCODER_MATCH = {
    "threads": "conch_v15",
    "titan": "conch_v15",
    "prism": "virchow",
    "chief": "ctranspath",
    "gigapath": "prov_gigapath",
    "madeleine": "conch_v1",
}


def slide_encoder_factory(model_name, pretrained=True, freeze=True, **kwargs):
    """Build a slide encoder by name.

    Parameters
    ----------
    model_name : str
        Slide encoder name: `"titan"`, `"threads"`, `"prism"`, `"chief"`,
        `"gigapath"`, `"madeleine"`, or `"mean-<patch encoder>"`.

    pretrained : bool, optional
        Load pretrained weights. Default is True.

    freeze : bool, optional
        Freeze the model weights and set it to eval mode. Default is True.

    **kwargs
        Keyword arguments forwarded to the encoder constructor.

    Returns
    -------
    BaseSlideEncoder
        The instantiated slide encoder.

    Raises
    ------
    ValueError
        If `model_name` is not supported.

    Notes
    -----
    For `mean-*` encoders, `pretrained`, `freeze` and `kwargs` are ignored.
    """

    if model_name.startswith("mean-"):
        enc = MeanSlideEncoder
        return enc(model_name=model_name)
    elif model_name == "titan":
        enc = TitanSlideEncoder
    elif model_name == "threads":
        enc = ThreadsSlideEncoder
    elif model_name == "prism":
        enc = PRISMSlideEncoder
    elif model_name == "chief":
        enc = CHIEFSlideEncoder
    elif model_name == "gigapath":
        enc = GigaPathSlideEncoder
    elif model_name == "madeleine":
        enc = MadeleineSlideEncoder
    else:
        raise ValueError(f"Model type {model_name} not supported")

    return enc(pretrained=pretrained, freeze=freeze, **kwargs)


class BaseSlideEncoder(torch.nn.Module):
    """Base class for pretrained slide-level encoders.

    A slide encoder aggregates the patch features of one slide into a slide
    embedding. `_build` is called once at construction.

    Parameters
    ----------
    freeze : bool, optional
        Disable gradients of `model` and set it to eval mode. Default is True.

    **build_kwargs
        Keyword arguments forwarded to `_build` (e.g. `pretrained`).

    Attributes
    ----------
    enc_name : str | None
        Encoder name, set by `_build`.
    model : torch.nn.Module | None
        Underlying network (None for mean pooling).
    precision : torch.dtype | None
        Recommended inference dtype.
    embedding_dim : int | None
        Dimension of the slide embedding.

    Notes
    -----
    Subclasses implement `_build(**build_kwargs)`, which must set
    `self.enc_name` and return `(model, precision, embedding_dim)` (note the
    order differs from `BasePatchEncoder._build`). `forward` takes a batch dict
    with at least `"features"` of shape `(B, N, D_tile)`, and for some models
    `"coords"` `(B, N, 2)` and `"attributes"`; it returns `(B, embedding_dim)`.
    """

    def __init__(self, freeze=True, **build_kwargs):
        """Build the encoder through `_build` and optionally freeze it."""
        super().__init__()
        self.enc_name = None
        self.model, self.precision, self.embedding_dim = self._build(**build_kwargs)

        # Set all parameters to be non-trainable
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def print_summary(self, depth=4, verbose=1):
        """Print a `torchinfo` summary of the model, if any.

        Parameters
        ----------
        depth : int, optional
            Depth of nested layers to display. Default is 4.

        verbose : int, optional
            `torchinfo` verbosity level. Default is 1.
        """
        if self.model is not None:
            model_summary(self.model, depth=depth, verbose=verbose)
        else:
            print("Slide encoder has no summary (must be mean-pooling)")

    def forward(self, batch):
        """Encode a slide from its batch.

        Can be overridden if the model requires a special forward pass.

        Parameters
        ----------
        batch : Any
            Model input, passed as is to `self.model`.

        Returns
        -------
        torch.Tensor
            Slide embedding, shape `(B, embedding_dim)`.
        """
        z = self.model(batch)
        return z

    @abstractmethod
    def _build(self, **build_kwargs):
        """Create the model, precision and embedding dim (to implement in subclasses).

        Parameters
        ----------
        **build_kwargs
            Encoder-specific options.

        Returns
        -------
        model : torch.nn.Module | None
            Slide encoder network.

        precision : torch.dtype | None
            Inference dtype.

        embedding_dim : int | None
            Slide embedding dimension.
        """
        pass


# Add a class for attention mil slide representation
# not implemented yet


class PRISMSlideEncoder(BaseSlideEncoder):
    """PRISM slide encoder (HF `paige-ai/Prism`, `transformers`, Virchow features).

    Requires Python >= 3.10 and `environs`, `sacremoses`, `transformers`. The text
    decoder is dropped. Embedding dim: 1280.

    Parameters
    ----------
    pretrained : bool, optional
        Load pretrained weights, else build from config. Default is True.
    """

    def _build(self, pretrained=True):
        """Load PRISM from the HF Hub (remote code).

        Returns
        -------
        tuple[torch.nn.Module, torch.dtype, int]
            `(model, precision, embedding_dim)`: `torch.float16`, 1280.

        Raises
        ------
        RuntimeError
            If Python is older than 3.10.
        Exception
            If PRISM dependencies cannot be imported.
        """

        self.enc_name = "prism"

        if sys.version_info < (3, 10):
            raise RuntimeError(
                "PRISM requires Python 3.10 or above. Please update your Python interpreter."
            )

        try:
            import environs  # weird dependencies required by PRISM
            import sacremoses
            from transformers import AutoModel, AutoConfig
        except:
            traceback.print_exc()
            raise Exception(
                "Please run `pip install environs==11.0.0 transformers==4.42.4 sacremoses==0.1.1` "
                "and ensure Python version is 3.10 or above."
            )

        if pretrained:
            model = AutoModel.from_pretrained("paige-ai/Prism", trust_remote_code=True)
        else:
            model = AutoModel.from_config(AutoConfig.from_pretrained("paige-ai/Prism"))
        model.text_decoder = None
        precision = torch.float16
        embedding_dim = 1280
        return model, precision, embedding_dim

    def forward(self, batch, device="cuda"):
        """Encode a slide with `model.slide_representations`.

        Parameters
        ----------
        batch : dict[str, Any]
            Must contain `"features"`, shape `(B, N, D_tile)`.
        device : str, optional
            Device to move the features to. Default is `"cuda"`.

        Returns
        -------
        torch.Tensor
            Image embedding, shape `(B, 1280)`.
        """
        # input should be of shape (batch_size, tile_seq_len, tile_embed_dim)
        # how tiles' positions are assessed
        x = batch["features"].to(device)
        z = self.model.slide_representations(x)
        z = z["image_embedding"]
        return z


class CHIEFSlideEncoder(BaseSlideEncoder):
    """CHIEF slide encoder (CTransPath features).

    Needs a local clone of https://github.com/hms-dbmi/CHIEF registered in
    `local_ckpts.json` (`"slide"` entry), with `model_weight/Text_emdding.pth` and
    `model_weight/CHIEF_pretraining.pth` downloaded manually, and `addict`
    installed. Embedding dim: 768.

    Parameters
    ----------
    pretrained : bool, optional
        Load `CHIEF_pretraining.pth`. Default is True.
    """

    def _build(self, pretrained=True):
        """Import CHIEF from the local repository and load its weights.

        Temporarily changes the working directory to the CHIEF repository.

        Returns
        -------
        tuple[torch.nn.Module, torch.dtype, int]
            `(model, precision, embedding_dim)`: `torch.float32`, 768.

        Raises
        ------
        Exception
            If the CHIEF repository cannot be imported or a weight file is missing.
        """

        self.enc_name = "chief"
        weights_path = get_weights_path("slide", self.enc_name)

        # Ensure model can be built.
        try:
            sys.path.append(weights_path)
            from models.CHIEF import CHIEF
        except Exception:
            traceback.print_exc()
            raise Exception(
                f"\nError: Unable to import the CHIEF repository from '{weights_path}'.\n\n"
                "To resolve this issue:\n"
                "1. Ensure you have cloned the CHIEF repository to a convenient location:\n"
                "   `git clone https://github.com/hms-dbmi/CHIEF/`\n"
                "2. Set the path to CHIEF repo in `trident/slide_encoder_models/load_ckpts.json`, e.g., `./CHIEF`.\n"
                "3. Verify that CHIEF dependencies are installed:\n"
                "   `pip install addict`\n\n"
            )

        # Ensure weights can be loaded.
        try:
            current_wd = os.getcwd()  # Get current working directory
            os.chdir(weights_path)  # Change to CHIEF repo directory
            os.makedirs(os.path.join(weights_path, "model_weight"), exist_ok=True)

            required_files = {
                "Text_emdding.pth": "https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV",
                "CHIEF_pretraining.pth": "https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV",
            }

            for file_name, download_link in required_files.items():
                file_path = os.path.join(weights_path, "model_weight", file_name)
                if not os.path.exists(file_path):
                    raise Exception(
                        f"\nError: Missing required file '{file_name}'.\n\n"
                        "To resolve this issue:\n"
                        f"1. Download the file from:\n   {download_link}\n"
                        f"2. Copy '{file_name}' to the following directory:\n   {file_path}\n\n"
                        "Ensure the file is correctly placed before retrying."
                    )

            print("All necessary files are present. CHIEF setup is complete!")

        except Exception as e:
            print("\nAn error occurred during CHIEF setup:")
            traceback.print_exc()
            raise e

        model = CHIEF(size_arg="small", dropout=True, n_classes=2)

        # Load pretrained weights
        if pretrained:
            td = torch.load(
                os.path.join("model_weight", "CHIEF_pretraining.pth"),
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(td, strict=True)

        # Return to original working directory
        os.chdir(current_wd)

        precision = torch.float32
        embedding_dim = 768
        return model, precision, embedding_dim

    def forward(self, batch, device="cuda"):
        """Encode one slide and return its `WSI_feature`.

        Parameters
        ----------
        batch : dict[str, Any]
            Must contain `"features"`, shape `(1, N, 768)`.
        device : str, optional
            Device to move the features to. Default is `"cuda"`.

        Returns
        -------
        torch.Tensor
            Slide embedding, shape `(1, 768)`.
        """
        # how tiles' positions are assessed
        x = batch["features"].squeeze(0).to(device)
        z = self.model(x, torch.tensor([0]))
        z = z["WSI_feature"]  # Shape (1,768)
        return z


class GigaPathSlideEncoder(BaseSlideEncoder):
    """Prov-GigaPath slide encoder (`gigapath`, HF `prov-gigapath/prov-gigapath`).

    Uses `gigapath_slide_enc12l768d` with 1536-dim Prov-GigaPath patch features.
    Requires `fairscale`, `gigapath` and `flash_attn==2.5.8`. Embedding dim: 768.

    Parameters
    ----------
    pretrained : bool, optional
        Load pretrained weights from the HF Hub. Default is True.
    """

    def _build(self, pretrained=True):
        """Create the GigaPath slide encoder with global pooling.

        Returns
        -------
        tuple[torch.nn.Module, torch.dtype, int]
            `(model, precision, embedding_dim)`: `torch.float16`, 768.

        Raises
        ------
        Exception
            If `gigapath` is not installed or `flash_attn` is not version 2.5.8.
        """

        self.enc_name = "gigapath"

        try:
            from gigapath.slide_encoder import create_model
        except:
            traceback.print_exc()
            raise Exception(
                "Please install fairscale and gigapath using `pip install fairscale git+https://github.com/prov-gigapath/prov-gigapath.git`."
            )

        # Make sure flash_attn is correct version
        try:
            import flash_attn

            assert flash_attn.__version__ == "2.5.8"
        except:
            traceback.print_exc()
            raise Exception(
                "Please install flash_attn version 2.5.8 using `pip install flash_attn==2.5.8`."
            )

        if pretrained:
            model = create_model(
                "hf_hub:prov-gigapath/prov-gigapath",
                "gigapath_slide_enc12l768d",
                1536,
                global_pool=True,
            )
        else:
            model = create_model(
                "", "gigapath_slide_enc12l768d", 1536, global_pool=True
            )

        precision = torch.float16
        embedding_dim = 768
        return model, precision, embedding_dim

    def forward(self, batch, device="cuda"):
        """Encode a slide and return the output of the last (12th) layer.

        Parameters
        ----------
        batch : dict[str, Any]
            Must contain `"features"` `(B, N, 1536)`, `"coords"` `(B, N, 2)` and
            `"attributes"["patch_size_level0"]` (sets `model.tile_size`).
        device : str, optional
            Device to move the inputs to. Default is `"cuda"`.

        Returns
        -------
        torch.Tensor
            Slide embedding, shape `(B, 768)`.
        """
        self.model.tile_size = batch["attributes"]["patch_size_level0"]
        z = self.model(
            batch["features"].to(device),
            batch["coords"].to(device),
            all_layer_embed=True,
        )[11]
        return z


class MadeleineSlideEncoder(BaseSlideEncoder):
    """MADELEINE slide encoder (`madeleine` package, CONCH v1 features).

    Weights path from `local_ckpts.json`. Requires `pip install
    git+https://github.com/mahmoodlab/MADELEINE.git`. Embedding dim: 512.

    Parameters
    ----------
    pretrained : bool, optional
        Must be True (no randomly initialised model). Default is True.
    """

    def _build(self, pretrained=True):
        """Load MADELEINE from its local weights.

        Returns
        -------
        tuple[torch.nn.Module, torch.dtype, int]
            `(model, precision, embedding_dim)`, `embedding_dim` is 512.

        Raises
        ------
        AssertionError
            If `pretrained` is False.
        Exception
            If `madeleine` is not installed.
        """

        assert (
            pretrained
        ), "MadeleineSlideEncoder has no non-pretrained models. Please load with pretrained=True."

        self.enc_name = "madeleine"
        weights_path = get_weights_path("slide", self.enc_name)
        embedding_dim = 512

        try:
            from madeleine.models.factory import create_model_from_pretrained
        except:
            traceback.print_exc()
            raise Exception(
                "Please install Madeleine using `pip install git+https://github.com/mahmoodlab/MADELEINE.git`"
            )

        model, precision = create_model_from_pretrained(weights_path)

        return model, precision, embedding_dim

    def forward(self, x, device="cuda"):
        """Encode a slide with `model.encode_he`.

        Parameters
        ----------
        x : dict[str, Any]
            Must contain `"features"`, shape `(B, N, 512)` (CONCH v1 features).
        device : str, optional
            Device used by `encode_he`. Default is `"cuda"`.

        Returns
        -------
        torch.Tensor
            Slide embedding, shape `(B, 512)`.
        """
        # how tiles' positions are assessed
        z = self.model.encode_he(x["features"], device)
        return z


class ThreadsSlideEncoder(BaseSlideEncoder):
    """THREADS slide encoder (`threadsmodel` package, CONCH v1.5 features).

    Not implemented yet: `_build` returns no model and `forward` returns None.

    Parameters
    ----------
    **build_kwargs
        Forwarded to `BaseSlideEncoder` (`freeze`, `pretrained`).
    """

    def __init__(self, **build_kwargs):
        """Initialise THREADS (see class docstring)."""
        super().__init__(**build_kwargs)

    def _build(self, pretrained=True):
        """Check that `threadsmodel` is importable (placeholder).

        Returns
        -------
        tuple[None, None, None]
            `(None, None, None)`.

        Raises
        ------
        Exception
            If `threadsmodel` cannot be imported.
        """

        self.enc_name = "threads"

        try:
            from threadsmodel.inference import (
                create_model,
                create_model_from_pretrained,
            )
        except:
            traceback.print_exc()
            raise Exception("Coming Soon! Thanks for your patience.")

        return None, None, None

    def forward(self, batch, device="cuda", return_raw_attention=False):
        """Not implemented; return None.

        Parameters
        ----------
        batch : dict[str, Any]
            Slide batch.
        device : str, optional
            Device. Default is `"cuda"`.
        return_raw_attention : bool, optional
            Return attention weights. Default is False.

        Returns
        -------
        None
        """
        pass


class TitanSlideEncoder(BaseSlideEncoder):
    """TITAN slide encoder (HF `MahmoodLab/TITAN`, gated, CONCH v1.5 features).

    Loaded with `transformers.AutoModel` (remote code). Embedding dim: 768.

    Parameters
    ----------
    pretrained : bool, optional
        Must be True (no randomly initialised model). Default is True.
    """

    def _build(self, pretrained=True):
        """Load TITAN from the HF Hub.

        Returns
        -------
        tuple[torch.nn.Module, torch.dtype, int]
            `(model, precision, embedding_dim)`: `torch.float16`, 768.

        Raises
        ------
        AssertionError
            If `pretrained` is False.
        """
        self.enc_name = "titan"
        assert (
            pretrained
        ), "TitanSlideEncoder has no non-pretrained models. Please load with pretrained=True."
        from transformers import AutoModel

        model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
        precision = torch.float16
        embedding_dim = 768
        return model, precision, embedding_dim

    def forward(self, batch, device="cuda"):
        """Encode a slide with `model.encode_slide_from_patch_features`.

        Parameters
        ----------
        batch : dict[str, Any]
            Must contain `"features"` `(B, N, 768)`, `"coords"` `(B, N, 2)` and
            `"attributes"["level_patch_size"]`.
        device : str, optional
            Device to move the inputs to. Default is `"cuda"`.

        Returns
        -------
        torch.Tensor
            Slide embedding, shape `(B, 768)`.
        """
        # do I have to pass coords at level 0 ?
        z = self.model.encode_slide_from_patch_features(
            batch["features"].to(device),
            batch["coords"].to(device),
            batch["attributes"]["level_patch_size"],
        )
        return z


class MeanSlideEncoder(BaseSlideEncoder):
    """Mean-pooling slide encoder (no learned weights).

    Parameters
    ----------
    model_name : str, optional
        `"mean-<patch encoder>"`, used to infer `embedding_dim`. Default is
        `"mean-default"`.
    """

    def _build(self, model_name="mean-default"):
        """Infer the embedding dim from `model_name`.

        Returns
        -------
        tuple[None, None, int | None]
            `(None, None, embedding_dim)`; `embedding_dim` is None (with a printed
            warning) for unknown patch encoders.
        """
        self.enc_name = model_name

        if model_name == "mean-conch_v1":
            embedding_dim = 768
        elif model_name == "mean-conch_v15":
            embedding_dim = 768
        elif model_name == "mean-uni_v1":
            embedding_dim = 1024
        elif model_name == "mean-uni_v2":
            embedding_dim = 1536
        elif model_name == "mean-ctranspath":
            embedding_dim = 768
        elif model_name == "mean-phikon":
            embedding_dim = 768
        elif model_name == "mean-phikon_v2":
            embedding_dim = 1024
        elif model_name == "mean-resnet50":
            embedding_dim = 1024
        elif model_name == "mean-gigapath":
            embedding_dim = 1536
        elif model_name == "mean-virchow":
            embedding_dim = 2560
        elif model_name == "mean-virchow2":
            embedding_dim = 2560
        elif model_name == "mean-hoptimus0":
            embedding_dim = 1536
        elif model_name == "mean-hoptimus1":
            embedding_dim = 1536
        elif model_name == "mean-musk":
            embedding_dim = 1024
        else:
            print(
                f"WARNING: Could not automatically infer embedding_dim for mean encoder {self.enc_name}. Setting to None."
            )
            embedding_dim = None
        return None, None, embedding_dim

    def forward(self, batch, device="cuda"):
        """Average the patch features of each slide.

        Parameters
        ----------
        batch : dict[str, Any]
            Must contain `"features"`, shape `(B, N, D)`.

        device : str, optional
            Device to move the features to. Default is `"cuda"`.

        Returns
        -------
        torch.Tensor
            Mean feature, shape `(B, D)`.
        """
        z = batch["features"].to(device).mean(dim=1)  # Just mean pooling
        return z
