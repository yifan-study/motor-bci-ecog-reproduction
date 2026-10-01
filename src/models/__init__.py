"""Model module — registry and builder.

Usage:
    from src.models import build_model
    model = build_model(config, dataset_info)
"""

from src.models.unet_lomtev import AutoEncoder1D
from src.models.unet_raw import AutoEncoder1DRaw
from src.models.tcn import TCN
from src.models.transformer import TransformerECoG, MultiscaleTransformerECoG, HybridTransformerECoG
from src.models.nested_unet import NestedUNet
from src.models.dtcnet import DTCNet
from src.models.dtcnet_attn import DTCNetAttn
from src.models.conformer import ConformerECoG
from src.models.conv_transformer import ConvTransformerECoG
from src.models.transformer_v2 import TransformerV2ECoG
from src.models.ssm_decoder import SSMDecoder


def _build_unet_lomtev(model_cfg, n_channels, n_input_features, n_targets):
    return AutoEncoder1D(
        n_electrodes=n_channels,
        n_freqs=n_input_features,
        n_channels_out=n_targets,
        channels=model_cfg["channels"],
        kernel_sizes=model_cfg["kernel_sizes"],
        strides=model_cfg["strides"],
        dilation=model_cfg["dilation"],
    )


def _build_unet_raw(model_cfg, n_channels, n_input_features, n_targets):
    return AutoEncoder1DRaw(
        n_channels_in=n_channels,
        n_channels_out=n_targets,
        channels=model_cfg["channels"],
        kernel_sizes=model_cfg["kernel_sizes"],
        strides=model_cfg["strides"],
        dilation=model_cfg["dilation"],
    )


def _build_tcn(model_cfg, n_channels, n_input_features, n_targets):
    return TCN(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        hidden_channels=model_cfg["hidden_channels"],
        kernel_size=model_cfg["kernel_size"],
        n_blocks=model_cfg["n_blocks"],
        dilation_base=model_cfg.get("dilation_base", 2),
        dropout=model_cfg.get("dropout", 0.1),
    )


def _build_transformer(model_cfg, n_channels, n_input_features, n_targets):
    return TransformerECoG(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        dim_feedforward=model_cfg["dim_feedforward"],
        spatial_kernel_size=model_cfg.get("spatial_kernel_size", 3),
        spatial_bottleneck_dim=model_cfg.get("spatial_bottleneck_dim", 0),
        ffn_type=model_cfg.get("ffn_type", "gelu"),
        dropout=model_cfg.get("dropout", 0.1),
        eval_window=model_cfg.get("eval_window", 256),
        channel_dropout_prob=model_cfg.get("channel_dropout_prob", 0.0),
    )


def _build_multiscale_transformer(model_cfg, n_channels, n_input_features, n_targets):
    return MultiscaleTransformerECoG(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        dim_feedforward=model_cfg["dim_feedforward"],
        spatial_kernel_size=model_cfg.get("spatial_kernel_size", 1),
        dropout=model_cfg.get("dropout", 0.1),
        downsample_factor=model_cfg.get("downsample_factor", 4),
        eval_window=model_cfg.get("eval_window", 256),
    )


def _build_hybrid_transformer(model_cfg, n_channels, n_input_features, n_targets):
    return HybridTransformerECoG(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        dim_feedforward=model_cfg["dim_feedforward"],
        spatial_kernel_size=model_cfg.get("spatial_kernel_size", 1),
        dropout=model_cfg.get("dropout", 0.1),
        eval_window=model_cfg.get("eval_window", 256),
    )


def _build_nested_unet(model_cfg, n_channels, n_input_features, n_targets):
    return NestedUNet(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        base_ch=model_cfg.get("base_ch", 32),
        kernel_size=model_cfg.get("kernel_size", 3),
        dropout=model_cfg.get("dropout", 0.1),
        decoder_kernel_size=model_cfg.get("decoder_kernel_size", 3),
        dense_output=model_cfg.get("dense_output", True),
    )


def _build_dtcnet(model_cfg, n_channels, n_input_features, n_targets):
    return DTCNet(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        dropout=model_cfg.get("dropout", 0.1),
    )


def _build_dtcnet_attn(model_cfg, n_channels, n_input_features, n_targets):
    return DTCNetAttn(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        dropout=model_cfg.get("dropout", 0.1),
        n_attn_layers=model_cfg.get("n_attn_layers", 2),
        n_attn_heads=model_cfg.get("n_attn_heads", 8),
        attn_dim_feedforward=model_cfg.get("attn_dim_feedforward", 2048),
    )


def _build_conv_transformer(model_cfg, n_channels, n_input_features, n_targets):
    return ConvTransformerECoG(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        dim_feedforward=model_cfg["dim_feedforward"],
        conv_channels=model_cfg.get("conv_channels", [128, 128, 256]),
        conv_kernels=model_cfg.get("conv_kernels", [7, 7, 5]),
        conv_dilations=model_cfg.get("conv_dilations", [1, 2, 3]),
        dropout=model_cfg.get("dropout", 0.1),
        eval_window=model_cfg.get("eval_window", 256),
        channel_dropout_prob=model_cfg.get("channel_dropout_prob", 0.0),
    )


def _build_conformer(model_cfg, n_channels, n_input_features, n_targets):
    return ConformerECoG(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        dim_feedforward=model_cfg["dim_feedforward"],
        conv_kernel_size=model_cfg.get("conv_kernel_size", 31),
        stem_kernel_size=model_cfg.get("stem_kernel_size", 3),
        dropout=model_cfg.get("dropout", 0.1),
        eval_window=model_cfg.get("eval_window", 256),
        channel_dropout_prob=model_cfg.get("channel_dropout_prob", 0.0),
    )


def _build_transformer_v2(model_cfg, n_channels, n_input_features, n_targets):
    return TransformerV2ECoG(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        dim_feedforward=model_cfg["dim_feedforward"],
        spatial_kernel_size=model_cfg.get("spatial_kernel_size", 1),
        conv_kernel_size=model_cfg.get("conv_kernel_size", 31),
        dropout=model_cfg.get("dropout", 0.1),
        eval_window=model_cfg.get("eval_window", 256),
        channel_dropout_prob=model_cfg.get("channel_dropout_prob", 0.0),
    )


def _build_ssm(model_cfg, n_channels, n_input_features, n_targets):
    return SSMDecoder(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        d_model=model_cfg.get("d_model", 256),
        d_state=model_cfg.get("d_state", 64),
        n_layers=model_cfg.get("n_layers", 6),
        dropout=model_cfg.get("dropout", 0.1),
    )


MODEL_REGISTRY = {
    "ssm": _build_ssm,
    "unet_lomtev": _build_unet_lomtev,
    "unet_raw": _build_unet_raw,
    "tcn": _build_tcn,
    "transformer": _build_transformer,
    "multiscale_transformer": _build_multiscale_transformer,
    "hybrid_transformer": _build_hybrid_transformer,
    "nested_unet": _build_nested_unet,
    "dtcnet": _build_dtcnet,
    "dtcnet_attn": _build_dtcnet_attn,
    "conv_transformer": _build_conv_transformer,
    "conformer": _build_conformer,
    "transformer_v2": _build_transformer_v2,
}


def build_model(config, dataset_info):
    """Construct a model from config and dataset metadata.

    Parameters
    ----------
    config : dict
        Full YAML config. Must have config["model"]["name"].
    dataset_info : dict
        Return value from build_data(). Must have 'n_channels' and
        'n_input_features'.

    Returns
    -------
    nn.Module
    """
    model_cfg = config["model"]
    name = model_cfg["name"]

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    n_channels = dataset_info["n_channels"]
    n_input_features = dataset_info.get("n_input_features", 1)
    n_targets = config["data"]["n_targets"]

    return MODEL_REGISTRY[name](model_cfg, n_channels, n_input_features, n_targets)
