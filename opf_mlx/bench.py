"""Throughput and memory benchmarks for the MLX Privacy Filter port.

Two workloads are measured, both of which map onto how the model is actually
deployed: one long document processed in a single bidirectional pass, and a batch
of short messages processed together.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx

from . import PrivacyFilter
from .convert import DEFAULT_CHECKPOINT

__all__ = ["build_batch", "build_document", "run_benchmark", "time_call"]

_LOREM = (
    "On 14 March 1987 Sarah Lindqvist wrote to marcus.webb@example.org from "
    "42 Rosewood Lane, Portland, to confirm that invoice 4455-9021 had cleared. "
    "The support desk replied the next morning and asked her to call 020 7946 0958 "
    "before the end of the quarter, quoting reference 88213 and the token "
    "ghp_16C7e42F292c6912E7710c838347Ae178B4a that had just been rotated. "
)

_SMS = [
    "Hey it's Tom, running late, start without me, back around eight tonight ok",
    "Bonjour, je confirme le rendez-vous du 15/01/2026 a neuf heures trente merci",
    "Your code is 448210, it expires in ten minutes, do not share it with anyone",
    "Colis livre au 17 bis rue des Acacias, sonner chez Fournier au 3e etage svp",
    "Call me back on 06 12 34 56 78 when you get out of the meeting please thanks",
]


def build_document(tokenizer, target_tokens: int) -> str:
    """Build one document whose tokenization is close to ``target_tokens`` tokens."""
    text = _LOREM
    while len(tokenizer.encode(text)) < target_tokens:
        text += _LOREM
    return tokenizer.decode(tokenizer.encode(text)[:target_tokens])


def build_batch(size: int, char_length: int = 160) -> list[str]:
    """Build ``size`` SMS-length messages of roughly ``char_length`` characters."""
    batch: list[str] = []
    for index in range(size):
        base = _SMS[index % len(_SMS)]
        message = f"[{index:03d}] {base} "
        while len(message) < char_length:
            message += base + " "
        batch.append(message[:char_length])
    return batch


def time_call(fn: Callable[[], Any], repeats: int) -> tuple[float, float, Any]:
    """Time ``fn`` after one warm-up call.

    Args:
        fn: The callable to time; it must be side-effect free across calls.
        repeats: Number of timed repetitions.

    Returns:
        The median and minimum wall-clock seconds, plus the last returned value.
    """
    result = fn()
    timings: list[float] = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - started)
    return statistics.median(timings), min(timings), result


def _mlx_document(pf: PrivacyFilter, token_ids: Sequence[int], repeats: int) -> dict[str, Any]:
    """Measure a single-document forward pass, excluding tokenization and decoding."""
    ids = mx.array([list(token_ids)], dtype=mx.uint32)

    def forward() -> None:
        out = pf.model(ids)
        mx.eval(out)

    mx.reset_peak_memory()
    median, best, _ = time_call(forward, repeats)
    return {
        "tokens": len(token_ids),
        "seconds_median": median,
        "tokens_per_s": len(token_ids) / median,
        "tokens_per_s_best": len(token_ids) / best,
        "peak_memory_gb": mx.get_peak_memory() / 2**30,
    }


def _mlx_batch(pf: PrivacyFilter, batch: Sequence[str], repeats: int) -> dict[str, Any]:
    """Measure a padded batched forward pass over short messages."""
    encoded = [pf.tokenizer.encode(text) for text in batch]
    total = sum(len(item) for item in encoded)

    def forward() -> None:
        pf.batch_logits(encoded)

    mx.reset_peak_memory()
    median, best, _ = time_call(forward, repeats)
    return {
        "messages": len(batch),
        "tokens": total,
        "seconds_median": median,
        "tokens_per_s": total / median,
        "messages_per_s": len(batch) / median,
        "peak_memory_gb": mx.get_peak_memory() / 2**30,
    }


def _mlx_end_to_end(pf: PrivacyFilter, batch: Sequence[str], repeats: int) -> dict[str, Any]:
    """Measure the full batched pipeline: tokenize, forward, Viterbi, spans."""
    median, _, _ = time_call(lambda: pf.predict_batch(list(batch)), repeats)
    return {"seconds_median": median, "messages_per_s": len(batch) / median}


def _torch_benchmark(
    checkpoint: str | Path,
    token_ids: Sequence[int],
    batch_tokens: Sequence[Sequence[int]],
    repeats: int,
) -> dict[str, Any]:
    """Time the reference PyTorch implementation on the MPS backend."""
    import os

    os.environ.setdefault("OPF_MOE_TRITON", "0")
    import torch
    from opf._model.model import Transformer

    if not torch.backends.mps.is_available():
        return {"available": False}

    model = Transformer.from_checkpoint(str(checkpoint), device="mps")
    model.eval()

    def run(ids_batch: Sequence[Sequence[int]]) -> float:
        width = max(len(item) for item in ids_batch)
        padded = [list(item) + [199999] * (width - len(item)) for item in ids_batch]
        tokens = torch.tensor(padded, dtype=torch.int32, device="mps")
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for row, item in enumerate(ids_batch):
            mask[row, : len(item)] = True

        def forward() -> None:
            with torch.inference_mode():
                model(tokens, attention_mask=mask)
            torch.mps.synchronize()

        median, _, _ = time_call(forward, repeats)
        return median

    # torch.mps exposes no peak counter, so this is the driver allocation
    # observed right after the workload rather than a true high-water mark.
    torch.mps.empty_cache()
    document_seconds = run([list(token_ids)])
    document_memory = torch.mps.driver_allocated_memory() / 2**30
    batch_seconds = run([list(item) for item in batch_tokens])
    batch_memory = torch.mps.driver_allocated_memory() / 2**30
    batch_total = sum(len(item) for item in batch_tokens)

    return {
        "available": True,
        "backend": "mps",
        "document": {
            "tokens": len(token_ids),
            "seconds_median": document_seconds,
            "tokens_per_s": len(token_ids) / document_seconds,
            "driver_memory_gb": document_memory,
        },
        "batch": {
            "messages": len(batch_tokens),
            "tokens": batch_total,
            "seconds_median": batch_seconds,
            "tokens_per_s": batch_total / batch_seconds,
            "messages_per_s": len(batch_tokens) / batch_seconds,
            "driver_memory_gb": batch_memory,
        },
    }


def run_benchmark(
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    document_tokens: int = 32768,
    batch_size: int = 64,
    repeats: int = 3,
    moe_precision: str | None = None,
    compare_torch: bool = False,
) -> dict[str, Any]:
    """Benchmark the MLX model, optionally against the PyTorch reference.

    Args:
        checkpoint: Checkpoint directory to load.
        document_tokens: Length of the synthetic long document, in tokens.
        batch_size: Number of SMS-length messages in the batch workload.
        repeats: Timed repetitions per workload, after one warm-up pass.
        moe_precision: Optional expert dtype override.
        compare_torch: Whether to also time the PyTorch reference on MPS.

    Returns:
        A nested mapping of measurements, ready to serialize as JSON.
    """
    pf = PrivacyFilter(checkpoint, moe_precision=moe_precision)
    document = build_document(pf.tokenizer, document_tokens)
    document_ids = pf.tokenizer.encode(document)[:document_tokens]
    batch = build_batch(batch_size)
    batch_ids = [pf.tokenizer.encode(text) for text in batch]

    results: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "moe_precision": moe_precision or "checkpoint",
        "mlx": {
            "document": _mlx_document(pf, document_ids, repeats),
            "batch": _mlx_batch(pf, batch, repeats),
            "batch_end_to_end": _mlx_end_to_end(pf, batch, repeats),
        },
    }
    if compare_torch:
        results["torch"] = _torch_benchmark(checkpoint, document_ids, batch_ids, repeats)
        torch_doc = results["torch"].get("document")
        if torch_doc:
            results["speedup"] = {
                "document": results["mlx"]["document"]["tokens_per_s"] / torch_doc["tokens_per_s"],
                "batch": (
                    results["mlx"]["batch"]["tokens_per_s"]
                    / results["torch"]["batch"]["tokens_per_s"]
                ),
            }
    return results
