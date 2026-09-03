"""Checkpoint conversion and loading for the MLX Privacy Filter port.

Reads the reference ``original/`` safetensors checkpoint published on the Hugging
Face Hub and rewrites it into MLX-native weights, optionally quantized with
``mx.quantize``.
"""

from __future__ import annotations

import copy
import glob
import json
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .model import Model, ModelArgs, SwitchLinear

__all__ = ["DEFAULT_CHECKPOINT", "convert", "download_reference_checkpoint", "load_model"]

DEFAULT_CHECKPOINT = Path.home() / ".opf" / "privacy_filter"
"""Directory used by the reference ``opf`` CLI, reused here to avoid a second copy."""

HF_REPO = "openai/privacy-filter"

DTYPES: dict[str, mx.Dtype] = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
}

_SIDECAR_FILES = ("viterbi_calibration.json",)

# Kept at full precision: the MoE router drives expert selection and the
# classification head has only 33 output rows, so neither is worth quantizing.
_UNQUANTIZED = ("mlp.gate", "score")


def download_reference_checkpoint(target: Path | str = DEFAULT_CHECKPOINT) -> Path:
    """Download the reference checkpoint from the Hub if it is not present yet.

    Args:
        target: Local directory that should hold ``config.json`` and the weights.

    Returns:
        The resolved checkpoint directory.
    """
    target = Path(target).expanduser()
    if (target / "config.json").is_file() and any(target.glob("*.safetensors")):
        return target

    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=HF_REPO, local_dir=str(target), allow_patterns=["original/*"])
    original = target / "original"
    if original.is_dir():
        for path in original.iterdir():
            shutil.move(str(path), str(target / path.name))
        original.rmdir()
    return target


def _read_weights(path: Path) -> dict[str, mx.array]:
    """Load and merge every safetensors shard in a checkpoint directory."""
    shards = sorted(glob.glob(str(path / "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No .safetensors file in {path}")
    weights: dict[str, mx.array] = {}
    for shard in shards:
        weights.update(mx.load(shard))
    return weights


def _quantize_predicate(name: str, module: nn.Module) -> bool:
    """Return whether one module should be quantized."""
    if any(name.endswith(suffix) for suffix in _UNQUANTIZED):
        return False
    return hasattr(module, "to_quantized")


def load_model(
    path: Path | str = DEFAULT_CHECKPOINT,
    *,
    moe_precision: str | None = None,
    attention_chunk_size: int | None = None,
) -> tuple[Model, ModelArgs]:
    """Instantiate the MLX model and load one checkpoint into it.

    Accepts either the reference checkpoint layout or a directory produced by
    :func:`convert`.

    Args:
        path: Checkpoint directory.
        moe_precision: Optional dtype name for the expert weights. ``"float32"``
            reproduces the reference implementation's expert arithmetic exactly;
            the default keeps the stored dtype.
        attention_chunk_size: Optional override for the attention tile width.

    Returns:
        The loaded model, already evaluated, and its resolved arguments.
    """
    path = Path(path).expanduser()
    with (path / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    args = ModelArgs.from_dict(config)
    if attention_chunk_size is not None:
        args.attention_chunk_size = attention_chunk_size

    model = Model(args)
    if args.quantization is not None:
        nn.quantize(
            model,
            group_size=args.quantization["group_size"],
            bits=args.quantization["bits"],
            mode=args.quantization.get("mode", "affine"),
            class_predicate=_quantize_predicate,
        )

    model.load_weights(list(Model.sanitize(_read_weights(path)).items()))

    if moe_precision is not None:
        dtype = DTYPES[moe_precision]
        for layer in model.model.layers:
            if isinstance(layer.mlp.w1, SwitchLinear):
                layer.mlp.w1.weight = layer.mlp.w1.weight.astype(dtype)
                layer.mlp.w1.bias = layer.mlp.w1.bias.astype(dtype)
                layer.mlp.w2.weight = layer.mlp.w2.weight.astype(dtype)
                layer.mlp.w2.bias = layer.mlp.w2.bias.astype(dtype)

    model.eval()
    mx.eval(model.parameters())
    return model, args


def convert(
    source: Path | str = DEFAULT_CHECKPOINT,
    output: Path | str = "mlx_model",
    *,
    dtype: str = "bfloat16",
    quantize: bool = False,
    group_size: int = 64,
    bits: int = 4,
) -> Path:
    """Convert a reference checkpoint into MLX-native weights.

    Args:
        source: Reference checkpoint directory.
        output: Destination directory, created if needed.
        dtype: Storage dtype for the unquantized tensors.
        quantize: Whether to quantize with ``mx.quantize``.
        group_size: Quantization group size along the input axis.
        bits: Quantization bit width.

    Returns:
        The destination directory.
    """
    source = Path(source).expanduser()
    output = Path(output).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    with (source / "config.json").open(encoding="utf-8") as handle:
        config: dict[str, Any] = json.load(handle)
    args = ModelArgs.from_dict(config)

    model = Model(args)
    model.load_weights(list(Model.sanitize(_read_weights(source)).items()))

    target = DTYPES[dtype]
    model.set_dtype(target, lambda t: mx.issubdtype(t, mx.floating) and t != mx.float32)

    out_config = copy.deepcopy(config)
    if quantize:
        nn.quantize(model, group_size=group_size, bits=bits, class_predicate=_quantize_predicate)
        out_config["quantization"] = {"group_size": group_size, "bits": bits, "mode": "affine"}
    out_config["mlx_dtype"] = dtype

    weights = dict(_flatten(model.parameters()))
    mx.eval(weights)
    mx.save_safetensors(str(output / "model.safetensors"), weights, metadata={"format": "mlx"})
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(out_config, handle, indent=2)
    for name in _SIDECAR_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    return output


def _flatten(tree: Any, prefix: str = "") -> list[tuple[str, mx.array]]:
    """Flatten a parameter tree into ``(dotted_name, array)`` pairs."""
    if isinstance(tree, dict):
        out: list[tuple[str, mx.array]] = []
        for key, value in tree.items():
            out.extend(_flatten(value, f"{prefix}.{key}" if prefix else key))
        return out
    if isinstance(tree, list):
        out = []
        for index, value in enumerate(tree):
            out.extend(_flatten(value, f"{prefix}.{index}"))
        return out
    return [(prefix, tree)]
