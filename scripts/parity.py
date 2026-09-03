#!/usr/bin/env python
"""Report parity of the MLX port against the official PyTorch reference.

Runs every sample in ``tests/samples/samples.json`` through both implementations
and prints, per configuration, how many label sequences and how many span sets
match, plus the largest absolute logit difference.

The reference is also compared against itself on two devices, which measures the
numerical noise floor that any bfloat16 implementation has to live with.

Usage:
    OPF_MOE_TRITON=0 python scripts/parity.py --quantized path/to/mlx-8bit
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("OPF_MOE_TRITON", "0")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from opf._core.decoding import build_sequence_decoder  # noqa: E402
from opf._core.runtime import load_inference_runtime, predict_text  # noqa: E402

from opf_mlx import PrivacyFilter  # noqa: E402
from opf_mlx.convert import DEFAULT_CHECKPOINT  # noqa: E402

SAMPLES = json.loads(
    (Path(__file__).resolve().parents[1] / "tests" / "samples" / "samples.json").read_text(
        encoding="utf-8"
    )
)


def _reference(checkpoint: str | Path, device: str) -> tuple[Any, Any]:
    """Build the reference runtime and decoder on one device."""
    runtime = load_inference_runtime(
        checkpoint=str(checkpoint),
        device_name=device,
        n_ctx_override=4096,
        trim_span_whitespace=True,
        discard_overlapping_predicted_spans=False,
        output_mode="typed",
    )
    decoder, _ = build_sequence_decoder(
        decode_mode="viterbi",
        label_info=runtime.label_info,
        viterbi_calibration_path=None,
        checkpoint_dir=str(checkpoint),
    )
    return runtime, decoder


def _reference_outputs(checkpoint: str | Path, device: str) -> list[dict[str, Any]]:
    """Run every sample through the reference implementation."""
    runtime, decoder = _reference(checkpoint, device)
    outputs: list[dict[str, Any]] = []
    for sample in SAMPLES:
        token_ids = runtime.encoding.encode(sample["text"], allowed_special="all")
        tokens = torch.tensor([token_ids], device=runtime.device, dtype=torch.int32)
        with torch.inference_mode():
            logits = runtime.model(tokens, attention_mask=torch.ones_like(tokens, dtype=torch.bool))
        prediction = predict_text(runtime, sample["text"], decoder=decoder)
        outputs.append(
            {
                "ids": token_ids,
                "logits": logits.float().cpu().numpy()[0],
                "spans": [(s.label, s.start, s.end) for s in prediction.spans],
            }
        )
    return outputs


def _compare(
    name: str, baseline: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]
) -> None:
    """Print how closely one run reproduces the baseline."""
    pairs = list(zip(baseline, candidate, strict=True))
    labels = sum(
        a["logits"].argmax(-1).tolist() == b["logits"].argmax(-1).tolist() for a, b in pairs
    )
    spans = sum(a["spans"] == b["spans"] for a, b in pairs)
    worst = max(float(np.abs(a["logits"] - b["logits"]).max()) for a, b in pairs)
    diverging = [
        sample["id"]
        for sample, (a, b) in zip(SAMPLES, pairs, strict=True)
        if a["spans"] != b["spans"]
    ]
    print(f"| {name} | {labels}/50 | {spans}/50 | {worst:.3g} |")
    if diverging:
        print(f"|   diverging: {', '.join(diverging)} | | | |")


def _mlx_outputs(
    checkpoint: str | Path, *, float32: bool = False, **kwargs: Any
) -> list[dict[str, Any]]:
    """Run every sample through the MLX port."""
    pf = PrivacyFilter(checkpoint, **kwargs)
    if float32:
        pf.model.set_dtype(mx.float32)
        mx.eval(pf.model.parameters())
    return [
        {
            "logits": pf.logits(pf.tokenizer.encode(sample["text"])),
            "spans": [(s.label, s.start, s.end) for s in pf.predict(sample["text"]).spans],
        }
        for sample in SAMPLES
    ]


def main() -> int:
    """Print the parity table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--quantized", action="append", default=[], metavar="DIR")
    parser.add_argument(
        "--float32-checkpoint", default=None, help="a checkpoint whose config sets param_dtype=fp32"
    )
    args = parser.parse_args()

    print("| Comparison (baseline: reference, CPU) | labels | spans | max abs logit diff |")
    print("| --- | ---: | ---: | ---: |")

    baseline = _reference_outputs(args.checkpoint, "cpu")
    if torch.backends.mps.is_available():
        _compare(
            "reference on MPS (noise floor)", baseline, _reference_outputs(args.checkpoint, "mps")
        )
    _compare("MLX bf16", baseline, _mlx_outputs(args.checkpoint))
    _compare(
        "MLX bf16, float32 experts",
        baseline,
        _mlx_outputs(args.checkpoint, moe_precision="float32"),
    )
    for directory in args.quantized:
        _compare(f"MLX {Path(directory).name}", baseline, _mlx_outputs(directory))

    if args.float32_checkpoint:
        print(
            "\n| Comparison (baseline: reference in float32) "
            "| labels | spans | max abs logit diff |"
        )
        print("| --- | ---: | ---: | ---: |")
        _compare(
            "MLX float32",
            _reference_outputs(args.float32_checkpoint, "cpu"),
            _mlx_outputs(args.checkpoint, float32=True),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
