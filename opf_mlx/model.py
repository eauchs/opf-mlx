"""Native MLX implementation of the OpenAI Privacy Filter token classifier.

The model is a pre-norm transformer encoder with banded bidirectional attention,
attention sinks, YaRN-scaled rotary embeddings and sparse mixture-of-experts
feed-forward blocks. It emits one BIOES label logit vector per input token in a
single forward pass.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "ModelArgs",
    "Model",
    "RMSNorm",
    "RotaryEmbedding",
    "Attention",
    "banded_attention",
    "banded_attention_reference",
    "SwitchLinear",
    "SparseMoeBlock",
    "TransformerBlock",
]

# Ratio between the checkpoint's log2-space sink logits and natural log space.
_LN2 = math.log(2.0)


@dataclass
class ModelArgs:
    """Configuration for one Privacy Filter checkpoint.

    Field names mirror the keys of the checkpoint's ``config.json`` so that
    :meth:`from_dict` can build an instance straight from the JSON payload.
    """

    model_type: str = "privacy_filter"
    num_hidden_layers: int = 8
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 200064
    num_labels: int = 33
    hidden_size: int = 640
    intermediate_size: int = 640
    head_dim: int = 64
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    sliding_window: int = 257
    bidirectional_context: bool = True
    bidirectional_left_context: int = 128
    bidirectional_right_context: int = 128
    initial_context_length: int = 4096
    max_position_embeddings: int = 131072
    default_n_ctx: int = 128000
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
    swiglu_limit: float = 7.0
    rms_norm_eps: float = 1e-5
    encoding: str = "o200k_base"
    attention_chunk_size: int = 256
    # Run the unfused reference attention instead of the tiled fast path.
    unfused_attention: bool = False
    quantization: dict[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate the invariants the reference runtime also enforces."""
        if not self.bidirectional_context:
            raise ValueError("Only bidirectional Privacy Filter checkpoints are supported")
        expected = self.bidirectional_left_context + self.bidirectional_right_context + 1
        if self.sliding_window != expected:
            raise ValueError(
                "sliding_window must equal left+right+1 "
                f"(got {self.sliding_window}, expected {expected})"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ModelArgs:
        """Build model arguments from a checkpoint config mapping."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in known})

    @classmethod
    def from_path(cls, path: str | Path) -> ModelArgs:
        """Load ``config.json`` from a checkpoint directory."""
        with (Path(path) / "config.json").open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @property
    def queries_per_kv(self) -> int:
        """Number of query heads sharing one key/value head."""
        return self.num_attention_heads // self.num_key_value_heads


class RMSNorm(nn.Module):
    """Root-mean-square normalisation with a float32 accumulation path."""

    def __init__(self, dims: int, eps: float = 1e-5) -> None:
        """Create a scale vector of ``dims`` entries."""
        super().__init__()
        self.weight = mx.ones((dims,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """Normalise ``x`` over its last axis and rescale, returning ``x``'s dtype."""
        t = x.astype(mx.float32)
        t = t * mx.rsqrt(mx.mean(mx.square(t), axis=-1, keepdims=True) + self.eps)
        return (t * self.weight).astype(x.dtype)


class RotaryEmbedding(nn.Module):
    """Interleaved rotary embeddings with YaRN "NTK by parts" frequency scaling."""

    def __init__(self, args: ModelArgs) -> None:
        """Precompute the YaRN concentration factor and inverse frequencies."""
        super().__init__()
        self.head_dim = args.head_dim
        self.base = float(args.rope_theta)
        self.initial_context_length = args.initial_context_length
        self.scaling_factor = float(args.rope_scaling_factor)
        self.ntk_alpha = float(args.rope_ntk_alpha)
        self.ntk_beta = float(args.rope_ntk_beta)
        self._concentration, self._inv_freq = self._concentration_and_inv_freq()
        self._cache_len = 0
        self._cos: mx.array | None = None
        self._sin: mx.array | None = None

    def _concentration_and_inv_freq(self) -> tuple[float, mx.array]:
        """Return the YaRN attention concentration and per-pair inverse frequencies.

        Follows the YaRN formulation (https://arxiv.org/abs/2309.00071): frequencies
        below the ``ntk_beta`` wavelength are extrapolated, frequencies above the
        ``ntk_alpha`` wavelength are interpolated, with a linear ramp in between.
        """
        freq = self.base ** (mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim)
        if self.scaling_factor <= 1.0:
            return 1.0, 1.0 / freq

        concentration = 0.1 * math.log(self.scaling_factor) + 1.0
        d_half = self.head_dim / 2
        low = (
            d_half
            * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
            / math.log(self.base)
        )
        high = (
            d_half
            * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
            / math.log(self.base)
        )
        if not 0 < low < high < d_half - 1:
            raise ValueError(f"Invalid YaRN ramp bounds: low={low}, high={high}")

        interpolation = 1.0 / (self.scaling_factor * freq)
        extrapolation = 1.0 / freq
        ramp = (mx.arange(d_half, dtype=mx.float32) - low) / (high - low)
        mask = 1 - mx.clip(ramp, 0, 1)
        return concentration, interpolation * (1 - mask) + extrapolation * mask

    def _cos_sin(self, num_tokens: int) -> tuple[mx.array, mx.array]:
        """Return cached ``(cos, sin)`` tables covering ``num_tokens`` positions."""
        if self._cos is None or num_tokens > self._cache_len:
            positions = mx.arange(num_tokens, dtype=mx.float32)[:, None]
            freqs = positions * self._inv_freq[None, :]
            self._cos = mx.cos(freqs) * self._concentration
            self._sin = mx.sin(freqs) * self._concentration
            self._cache_len = num_tokens
            mx.eval(self._cos, self._sin)
        return self._cos[:num_tokens], self._sin[:num_tokens]

    def __call__(self, q: mx.array, k: mx.array) -> tuple[mx.array, mx.array]:
        """Rotate ``q`` and ``k``, both shaped ``[B, T, heads, head_dim]``."""
        num_tokens = q.shape[1]
        cos, sin = self._cos_sin(num_tokens)
        cos = cos[None, :, None, :].astype(q.dtype)
        sin = sin[None, :, None, :].astype(q.dtype)
        return _rotate(q, cos, sin), _rotate(k, cos, sin)


def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply one interleaved rotation to ``x`` shaped ``[B, T, heads, head_dim]``."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return mx.stack([o1, o2], axis=-1).reshape(x.shape)


def banded_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    sinks: mx.array,
    *,
    left: int,
    right: int,
    scale: float = 1.0,
    key_mask: mx.array | None = None,
    chunk_size: int = 512,
) -> mx.array:
    """Attend over an asymmetric local band with a per-head attention sink.

    The band is resolved one query tile at a time rather than as a dense
    ``[T, T]`` mask, which would be O(T^2): at 32k tokens that mask alone costs
    more than the weights. Every query in a tile can reach only keys within the
    band, so the tile carries all of them and each softmax, sink included, is
    exactly the one a dense pass would compute.

    Args:
        q: Queries shaped ``[B, n_heads, T, head_dim]``.
        k: Keys shaped ``[B, n_kv_heads, T, head_dim]``.
        v: Values shaped ``[B, n_kv_heads, T, head_dim]``.
        sinks: Per-head sink logits in natural log space, shaped ``[n_heads]``.
        left: Number of past tokens each query may attend to.
        right: Number of future tokens each query may attend to.
        scale: Multiplier applied to the attention scores.
        key_mask: Optional ``[B, T]`` boolean mask of valid key positions.
        chunk_size: Number of query positions processed per tile.

    Returns:
        Context vectors shaped ``[B, n_heads, T, head_dim]``.
    """
    num_tokens = q.shape[2]
    dtype = q.dtype
    sinks = sinks.astype(dtype)
    keep = mx.array(0.0, dtype)
    drop = mx.array(-mx.inf, dtype)
    outputs: list[mx.array] = []

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        key_start = max(0, start - left)
        key_end = min(num_tokens, end + right)

        query_pos = mx.arange(start, end)[:, None]
        key_pos = mx.arange(key_start, key_end)[None, :]
        band = (key_pos >= query_pos - left) & (key_pos <= query_pos + right)
        mask = mx.where(band, keep, drop)[None, None, :, :]
        if key_mask is not None:
            valid = mx.where(key_mask[:, key_start:key_end], keep, drop)
            mask = mask + valid[:, None, None, :]

        outputs.append(
            mx.fast.scaled_dot_product_attention(
                q[:, :, start:end, :],
                k[:, :, key_start:key_end, :],
                v[:, :, key_start:key_end, :],
                scale=scale,
                mask=mask,
                sinks=sinks,
            )
        )

    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)


def banded_attention_reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    sinks: mx.array,
    *,
    left: int,
    right: int,
    scale: float = 1.0,
    key_mask: mx.array | None = None,
) -> mx.array:
    """Unfused banded attention, written to mirror the reference implementation.

    Scores are accumulated in the input dtype and the softmax runs in float32,
    exactly as the PyTorch reference does. This is quadratic in the sequence
    length and is kept as an executable specification for
    :func:`banded_attention`, which the tests check it against; it is not used
    at inference time.

    Args and returns match :func:`banded_attention`.
    """
    batch, n_heads, num_tokens, head_dim = q.shape
    n_kv_heads = k.shape[1]
    group = n_heads // n_kv_heads

    grouped_q = q.reshape(batch, n_kv_heads, group, num_tokens, head_dim)
    grouped_k = k[:, :, None]
    grouped_v = v[:, :, None]

    scores = mx.matmul(grouped_q, mx.swapaxes(grouped_k, -1, -2)).astype(mx.float32)
    scores = scores * scale

    query_pos = mx.arange(num_tokens)[:, None]
    key_pos = mx.arange(num_tokens)[None, :]
    valid = ((key_pos >= query_pos - left) & (key_pos <= query_pos + right))[None, None, None]
    if key_mask is not None:
        valid = valid & key_mask[:, None, None, None, :]
    scores = mx.where(valid, scores, mx.array(-mx.inf, mx.float32))

    sink_column = sinks.astype(mx.float32).reshape(n_kv_heads, group)[None, :, :, None, None]
    sink_column = mx.broadcast_to(sink_column, (*scores.shape[:-1], 1))
    weights = mx.softmax(mx.concatenate([scores, sink_column], axis=-1), axis=-1)
    context = mx.matmul(weights[..., :-1].astype(v.dtype), grouped_v)
    return context.reshape(batch, n_heads, num_tokens, head_dim)


class Attention(nn.Module):
    """Grouped-query banded attention block with rotary embeddings and sinks."""

    def __init__(self, args: ModelArgs) -> None:
        """Build the projections, norm, sink logits and rotary cache."""
        super().__init__()
        self.args = args
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.q_per_kv = args.queries_per_kv
        self.left = args.bidirectional_left_context
        self.right = args.bidirectional_right_context
        # Applied to both queries and keys, so the score scale is 1/sqrt(head_dim).
        self.qk_scale = 1.0 / math.sqrt(math.sqrt(args.head_dim))

        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        qkv_dim = args.head_dim * (args.num_attention_heads + 2 * args.num_key_value_heads)
        self.qkv_proj = nn.Linear(args.hidden_size, qkv_dim, bias=True)
        self.o_proj = nn.Linear(
            args.head_dim * args.num_attention_heads, args.hidden_size, bias=True
        )
        self.sinks = mx.zeros((args.num_attention_heads,), dtype=mx.float32)
        self.rope = RotaryEmbedding(args)

    def __call__(self, x: mx.array, key_mask: mx.array | None = None) -> mx.array:
        """Return the attention branch output for ``x`` shaped ``[B, T, hidden]``."""
        batch, num_tokens, _ = x.shape
        qkv = self.qkv_proj(self.norm(x))

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        q = qkv[..., :q_dim].reshape(batch, num_tokens, self.n_heads, self.head_dim)
        k = qkv[..., q_dim : q_dim + kv_dim]
        k = k.reshape(batch, num_tokens, self.n_kv_heads, self.head_dim)
        v = qkv[..., q_dim + kv_dim :]
        v = v.reshape(batch, num_tokens, self.n_kv_heads, self.head_dim)

        q, k = self.rope(q, k)
        # The reference scales both sides rather than the scores, so the score
        # multiplier stays 1 and the rounding matches.
        q = (q * self.qk_scale).transpose(0, 2, 1, 3)
        k = (k * self.qk_scale).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Checkpoint sinks are stored in log2 space.
        sinks = self.sinks * _LN2
        if self.args.unfused_attention:
            context = banded_attention_reference(
                q, k, v, sinks, left=self.left, right=self.right, key_mask=key_mask
            )
        else:
            context = banded_attention(
                q,
                k,
                v,
                sinks,
                left=self.left,
                right=self.right,
                key_mask=key_mask,
                chunk_size=self.args.attention_chunk_size,
            )
        context = context.transpose(0, 2, 1, 3).reshape(batch, num_tokens, q_dim)
        return self.o_proj(context)


def swiglu(x: mx.array, limit: float) -> mx.array:
    """Apply the clamped SwiGLU nonlinearity used by the reference MoE experts."""
    x_glu, x_linear = mx.split(x, 2, axis=-1)
    x_glu = mx.minimum(x_glu, limit)
    x_linear = mx.clip(x_linear, -limit, limit)
    return (x_glu * mx.sigmoid(1.702 * x_glu)) * (x_linear + 1)


class SwitchLinear(nn.Module):
    """Per-expert affine projection gathered by routing indices."""

    def __init__(self, input_dims: int, output_dims: int, num_experts: int) -> None:
        """Allocate ``num_experts`` weight and bias slabs."""
        super().__init__()
        self.weight = mx.zeros((num_experts, output_dims, input_dims))
        self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dtype(self) -> mx.Dtype:
        """Dtype this layer's inputs must be cast to."""
        return self.weight.dtype

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        """Project ``x`` through the experts named by ``indices``.

        Args:
            x: One ``[..., 1, in]`` row per routed token, already in expert order.
            indices: Expert id per row of ``x``.
            sorted_indices: Whether ``indices`` is sorted, which lets MLX run one
                grouped matmul per expert instead of one per routed token.

        Returns:
            The projected rows, shaped ``[..., 1, out]``.
        """
        y = mx.gather_mm(
            x,
            mx.swapaxes(self.weight, -1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        return y + self.bias[indices][..., None, :]

    def to_quantized(
        self, group_size: int = 64, bits: int = 4, mode: str = "affine"
    ) -> QuantizedSwitchLinear:
        """Return a quantized copy of this layer."""
        return QuantizedSwitchLinear.from_switch_linear(self, group_size, bits, mode)


class QuantizedSwitchLinear(nn.Module):
    """Quantized counterpart of :class:`SwitchLinear` backed by ``mx.gather_qmm``."""

    def __init__(
        self,
        weight: mx.array,
        scales: mx.array,
        biases: mx.array,
        bias: mx.array,
        group_size: int,
        bits: int,
        mode: str = "affine",
    ) -> None:
        """Store the packed expert weights and the per-expert output bias."""
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.bias = bias
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.freeze()

    @classmethod
    def from_switch_linear(
        cls, layer: SwitchLinear, group_size: int, bits: int, mode: str = "affine"
    ) -> QuantizedSwitchLinear:
        """Quantize the expert weights of ``layer`` along their input axis."""
        weight, scales, biases = mx.quantize(layer.weight, group_size, bits, mode=mode)
        return cls(weight, scales, biases, layer.bias, group_size, bits, mode)

    @property
    def input_dtype(self) -> mx.Dtype:
        """Dtype this layer's inputs must be cast to.

        The packed ``weight`` is an integer array, so the scales carry the dtype
        the layer actually computes in.
        """
        return self.scales.dtype

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        """Project ``x`` through the quantized experts named by ``indices``."""
        y = mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        return y + self.bias[indices][..., None, :]


class SparseMoeBlock(nn.Module):
    """Top-k routed mixture-of-experts feed-forward block."""

    def __init__(self, args: ModelArgs) -> None:
        """Build the router and the two per-expert projections."""
        super().__init__()
        self.args = args
        self.top_k = args.experts_per_token
        self.swiglu_limit = args.swiglu_limit
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=True)
        self.w1 = SwitchLinear(args.hidden_size, 2 * args.intermediate_size, args.num_experts)
        self.w2 = SwitchLinear(args.intermediate_size, args.hidden_size, args.num_experts)

    def __call__(self, x: mx.array) -> mx.array:
        """Return the MoE branch output for ``x`` shaped ``[B, T, hidden]``."""
        shape = x.shape
        hidden = shape[-1]
        top_k = self.top_k
        t = self.norm(x).reshape(-1, hidden)

        # The router always runs in float32, matching the reference implementation.
        gates = mx.matmul(
            t.astype(mx.float32), self.gate.weight.astype(mx.float32).T
        ) + self.gate.bias.astype(mx.float32)
        indices = mx.stop_gradient(mx.argpartition(-gates, top_k - 1, axis=-1))
        indices = indices[..., :top_k]
        weights = mx.softmax(mx.take_along_axis(gates, indices, axis=-1), axis=-1)

        # Routed tokens are grouped by expert so that each expert runs as a single
        # matmul over its whole caseload rather than one matmul per token. The rows
        # are gathered into expert order up front: letting gather_mm do that
        # indexing itself costs 4x more here, 24.2 ms against 5.9 ms per layer on
        # an 8k-token document.
        order = mx.argsort(indices.reshape(-1))
        experts = indices.reshape(-1)[order]
        rows = order // top_k

        h = t.astype(self.w1.input_dtype)[rows][:, None, :]
        h = self.w1(h, experts, sorted_indices=True)
        h = swiglu(h, self.swiglu_limit)
        h = self.w2(h.astype(self.w2.input_dtype), experts, sorted_indices=True)

        h = h.squeeze(-2)[mx.argsort(order)].reshape(-1, top_k, hidden)
        out = (h.astype(mx.float32) * weights[..., None]).sum(axis=-2)
        return out.astype(x.dtype).reshape(shape)


class TransformerBlock(nn.Module):
    """One attention plus mixture-of-experts layer with residual connections."""

    def __init__(self, args: ModelArgs) -> None:
        """Build the attention and MoE branches."""
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = SparseMoeBlock(args)

    def __call__(self, x: mx.array, key_mask: mx.array | None = None) -> mx.array:
        """Run both residual branches over ``x``."""
        x = x + self.self_attn(x, key_mask)
        return x + self.mlp(x)


class PrivacyFilterEncoder(nn.Module):
    """Embedding table, transformer stack and final normalisation."""

    def __init__(self, args: ModelArgs) -> None:
        """Build the encoder stack."""
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, input_ids: mx.array, key_mask: mx.array | None = None) -> mx.array:
        """Return the final hidden states for ``input_ids`` shaped ``[B, T]``."""
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, key_mask)
        return self.norm(h)


class Model(nn.Module):
    """Privacy Filter token classifier producing ``[B, T, num_labels]`` logits."""

    def __init__(self, args: ModelArgs) -> None:
        """Build the encoder and the token-classification head."""
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = PrivacyFilterEncoder(args)
        self.score = nn.Linear(args.hidden_size, args.num_labels, bias=False)

    def __call__(self, input_ids: mx.array, key_mask: mx.array | None = None) -> mx.array:
        """Classify every token of ``input_ids`` in a single bidirectional pass.

        Args:
            input_ids: Token ids shaped ``[B, T]``.
            key_mask: Optional ``[B, T]`` boolean mask; ``False`` marks padding.

        Returns:
            Label logits shaped ``[B, T, num_labels]``.
        """
        return self.score(self.model(input_ids, key_mask))

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Rename reference checkpoint tensors onto this module tree.

        The reference checkpoint stores expert weights as ``[experts, in, out]``;
        they are transposed here to the ``[experts, out, in]`` layout expected by
        ``mx.gather_mm`` and ``mx.quantize``.
        """
        renamed: dict[str, mx.array] = {}
        for key, value in weights.items():
            parts = key.split(".")
            if key == "embedding.weight":
                renamed["model.embed_tokens.weight"] = value
            elif key == "norm.scale":
                renamed["model.norm.weight"] = value
            elif key == "unembedding.weight":
                renamed["score.weight"] = value
            elif parts[0] == "block":
                name, tensor = _sanitize_block_key(parts, value)
                renamed[name] = tensor
            else:
                renamed[key] = value
        # RMSNorm scales are float32 parameters in the reference implementation.
        for name in list(renamed):
            if name.endswith("norm.weight"):
                renamed[name] = renamed[name].astype(mx.float32)
        return renamed


def _sanitize_block_key(parts: list[str], value: mx.array) -> tuple[str, mx.array]:
    """Map one ``block.<i>.*`` checkpoint tensor onto its module path."""
    index, branch, *rest = parts[1:]
    tail = ".".join(rest)
    if branch == "attn":
        prefix = f"model.layers.{index}.self_attn"
        mapping = {
            "norm.scale": "norm.weight",
            "qkv.weight": "qkv_proj.weight",
            "qkv.bias": "qkv_proj.bias",
            "out.weight": "o_proj.weight",
            "out.bias": "o_proj.bias",
            "sinks": "sinks",
        }
        return f"{prefix}.{mapping[tail]}", value
    if branch == "mlp":
        prefix = f"model.layers.{index}.mlp"
        if tail == "norm.scale":
            return f"{prefix}.norm.weight", value
        if tail in ("gate.weight", "gate.bias"):
            return f"{prefix}.{tail}", value
        # swiglu -> w1 (first MoE projection), out -> w2 (second MoE projection).
        target = {"swiglu": "w1", "out": "w2"}[rest[0]]
        if rest[1] == "weight":
            return f"{prefix}.{target}.weight", mx.swapaxes(value, -1, -2)
        return f"{prefix}.{target}.bias", value
    raise KeyError(f"Unexpected checkpoint tensor: {'.'.join(parts)}")
