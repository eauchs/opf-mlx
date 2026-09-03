#!/usr/bin/env python
"""Run the full benchmark matrix and print it as a Markdown table.

Measures the MLX model on a long document and on a batch of SMS-length messages,
in every requested precision, and optionally times the PyTorch reference on the
MPS backend for comparison.

Usage:
    python scripts/bench.py --compare-torch --quantized path/to/mlx-8bit
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from opf_mlx.bench import run_benchmark
from opf_mlx.convert import DEFAULT_CHECKPOINT


def _machine() -> dict[str, str]:
    """Describe the host so that published numbers stay attributable."""

    def sysctl(key: str) -> str:
        try:
            return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    memory = sysctl("hw.memsize")
    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_gb": f"{int(memory) / 2**30:.0f}" if memory.isdigit() else "unknown",
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


def _row(name: str, result: dict[str, Any]) -> str:
    """Format one Markdown table row from a benchmark result."""
    document = result["mlx"]["document"]
    batch = result["mlx"]["batch"]
    return (
        f"| {name} | {document['tokens_per_s']:,.0f} | {document['peak_memory_gb']:.2f} | "
        f"{batch['tokens_per_s']:,.0f} | {batch['messages_per_s']:,.0f} | "
        f"{batch['peak_memory_gb']:.2f} |"
    )


def main() -> int:
    """Run the benchmark matrix and print the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--quantized",
        action="append",
        default=[],
        metavar="DIR",
        help="additional converted checkpoint to benchmark, repeatable",
    )
    parser.add_argument("--document-tokens", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compare-torch", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    runs: dict[str, dict[str, Any]] = {}
    runs["MLX bf16"] = run_benchmark(
        checkpoint=args.checkpoint,
        document_tokens=args.document_tokens,
        batch_size=args.batch_size,
        repeats=args.repeats,
        compare_torch=args.compare_torch,
    )
    runs["MLX bf16, float32 experts"] = run_benchmark(
        checkpoint=args.checkpoint,
        document_tokens=args.document_tokens,
        batch_size=args.batch_size,
        repeats=args.repeats,
        moe_precision="float32",
    )
    for directory in args.quantized:
        label = Path(directory).name
        runs[f"MLX {label}"] = run_benchmark(
            checkpoint=directory,
            document_tokens=args.document_tokens,
            batch_size=args.batch_size,
            repeats=args.repeats,
        )

    host = _machine()
    print(f"\nHost: {host['chip']}, {host['memory_gb']} GB, {host['os']}, Python {host['python']}")
    print(
        f"Document: {args.document_tokens} tokens. "
        f"Batch: {args.batch_size} messages of 160 characters.\n"
    )
    print("| Variant | doc tok/s | doc peak GB | batch tok/s | batch msg/s | batch peak GB |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, result in runs.items():
        print(_row(name, result))

    torch_result = runs["MLX bf16"].get("torch")
    if torch_result and torch_result.get("available"):
        document = torch_result["document"]
        batch = torch_result["batch"]
        print(
            f"| PyTorch reference, MPS | {document['tokens_per_s']:,.0f} | "
            f"{document['driver_memory_gb']:.2f}* | {batch['tokens_per_s']:,.0f} | "
            f"{batch['messages_per_s']:,.0f} | {batch['driver_memory_gb']:.2f}* |"
        )
        speedup = runs["MLX bf16"]["speedup"]
        print(
            f"\nMLX bf16 over PyTorch-MPS: document x{speedup['document']:.1f}, "
            f"batch x{speedup['batch']:.1f}"
        )
        print("* driver-allocated memory after the run; torch.mps exposes no peak counter.")

    if args.json_out:
        args.json_out.write_text(
            json.dumps({"host": host, "runs": runs}, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
