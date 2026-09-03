"""Unit tests for the MLX modules that need no checkpoint."""

from __future__ import annotations

import mlx.core as mx
import pytest

from opf_mlx.model import (
    ModelArgs,
    banded_attention,
    banded_attention_reference,
    swiglu,
)

LEFT = RIGHT = 5


def _inputs(dtype: mx.Dtype = mx.float32, batch: int = 1, num_tokens: int = 40):
    """Build queries, keys, values and sinks for a small grouped-query layer."""
    mx.random.seed(0)
    n_heads, n_kv_heads, head_dim = 14, 2, 8
    q = mx.random.normal((batch, n_heads, num_tokens, head_dim)).astype(dtype)
    k = mx.random.normal((batch, n_kv_heads, num_tokens, head_dim)).astype(dtype)
    v = mx.random.normal((batch, n_kv_heads, num_tokens, head_dim)).astype(dtype)
    sinks = mx.random.normal((n_heads,)).astype(mx.float32)
    return q, k, v, sinks


@pytest.mark.parametrize("chunk_size", [7, 16, 40, 128])
def test_tiled_attention_matches_reference(chunk_size: int) -> None:
    """The tiled fast path must compute the same attention as the reference."""
    q, k, v, sinks = _inputs()
    expected = banded_attention_reference(q, k, v, sinks, left=LEFT, right=RIGHT)
    got = banded_attention(q, k, v, sinks, left=LEFT, right=RIGHT, chunk_size=chunk_size)
    mx.eval(expected, got)
    assert mx.allclose(expected, got, atol=1e-4).item(), f"diverged at chunk_size={chunk_size}"


def test_attention_sinks_change_the_result() -> None:
    """Guard the comparison above: it must be able to fail."""
    q, k, v, sinks = _inputs()
    with_sinks = banded_attention(q, k, v, sinks, left=LEFT, right=RIGHT, chunk_size=16)
    without = banded_attention(
        q, k, v, mx.full(sinks.shape, -30.0), left=LEFT, right=RIGHT, chunk_size=16
    )
    mx.eval(with_sinks, without)
    assert not mx.allclose(with_sinks, without, atol=1e-4).item()


def test_tiled_attention_honours_key_padding() -> None:
    """Masked key positions must stay masked when the band is tiled."""
    q, k, v, sinks = _inputs(batch=2, num_tokens=24)
    key_mask = mx.array([[True] * 24, [True] * 15 + [False] * 9])
    expected = banded_attention_reference(q, k, v, sinks, left=LEFT, right=RIGHT, key_mask=key_mask)
    got = banded_attention(q, k, v, sinks, left=LEFT, right=RIGHT, key_mask=key_mask, chunk_size=8)
    mx.eval(expected, got)
    assert mx.allclose(expected[0], got[0], atol=1e-4).item()
    assert mx.allclose(expected[1, :, :15], got[1, :, :15], atol=1e-4).item()


def test_attention_band_is_asymmetric_aware() -> None:
    """A query must not see beyond its own left and right reach."""
    q, k, v, sinks = _inputs(num_tokens=32)
    narrow = banded_attention(q, k, v, sinks, left=0, right=0, chunk_size=8)
    wide = banded_attention(q, k, v, sinks, left=8, right=8, chunk_size=8)
    mx.eval(narrow, wide)
    assert not mx.allclose(narrow, wide, atol=1e-4).item()


def test_swiglu_clamps_both_halves() -> None:
    """The gate is clamped above only, the linear half on both sides."""
    x = mx.array([[100.0, -100.0, 100.0, -100.0]])
    out = swiglu(x, limit=7.0)
    expected_gate = 7.0 * float(mx.sigmoid(mx.array(1.702 * 7.0)).item())
    assert out.shape == (1, 2)
    assert out[0, 0].item() == pytest.approx(expected_gate * (7.0 + 1), rel=1e-4)
    assert out[0, 1].item() == pytest.approx(
        -100.0 * float(mx.sigmoid(mx.array(1.702 * -100.0)).item()) * (-7.0 + 1), abs=1e-6
    )


def test_model_args_reject_inconsistent_window() -> None:
    """The band width must agree with the declared left and right context."""
    with pytest.raises(ValueError, match="sliding_window"):
        ModelArgs(
            sliding_window=128, bidirectional_left_context=128, bidirectional_right_context=128
        )
    with pytest.raises(ValueError, match="bidirectional"):
        ModelArgs(bidirectional_context=False)
