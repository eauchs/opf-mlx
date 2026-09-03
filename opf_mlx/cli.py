"""Command line interface for the MLX Privacy Filter port."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import PrivacyFilter, __version__
from .convert import DEFAULT_CHECKPOINT, convert

__all__ = ["main"]

_LABEL_COLORS = {
    "account_number": 33,
    "private_address": 35,
    "private_date": 36,
    "private_email": 34,
    "private_person": 32,
    "private_phone": 31,
    "private_url": 94,
    "secret": 91,
    "redacted": 90,
}


def _read_input(source: str) -> str:
    """Read one input document from a path or from stdin when ``source`` is ``-``."""
    if source == "-":
        return sys.stdin.read()
    return Path(source).read_text(encoding="utf-8")


def _build_filter(args: argparse.Namespace) -> PrivacyFilter:
    """Instantiate a :class:`PrivacyFilter` from parsed arguments."""
    return PrivacyFilter(
        args.checkpoint,
        n_ctx=args.n_ctx,
        decode_mode=args.decode_mode,
        trim_whitespace=not args.no_trim,
        discard_overlapping_spans=args.discard_overlapping_spans,
        output_mode=args.output_mode,
        moe_precision=args.moe_precision,
    )


def _colorize(text: str, spans, use_color: bool) -> str:
    """Return ``text`` with each detected span highlighted for a terminal."""
    if not use_color or not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        pieces.append(text[cursor : span.start])
        color = _LABEL_COLORS.get(span.label, 37)
        pieces.append(f"\033[{color}m{text[span.start : span.end]}\033[0m")
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _cmd_redact(args: argparse.Namespace) -> int:
    """Print the redacted form of one document."""
    text = _read_input(args.input)
    started = time.perf_counter()
    prediction = _build_filter(args).predict(text)
    elapsed_ms = (time.perf_counter() - started) * 1000

    if args.json:
        print(json.dumps(prediction.to_dict(), indent=2, ensure_ascii=False))
    else:
        sys.stdout.write(prediction.redacted_text)
        if not prediction.redacted_text.endswith("\n"):
            sys.stdout.write("\n")
    counts: dict[str, int] = {}
    for span in prediction.spans:
        counts[span.label] = counts.get(span.label, 0) + 1
    summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "-"
    print(
        f"summary: spans={len(prediction.spans)} by_label={summary} "
        f"latency_ms={elapsed_ms:.1f} decoded_mismatch={prediction.decoded_mismatch}",
        file=sys.stderr,
    )
    return 0


def _cmd_spans(args: argparse.Namespace) -> int:
    """Print the detected spans of one document as JSON."""
    text = _read_input(args.input)
    prediction = _build_filter(args).predict(text)
    payload = prediction.to_dict()
    if args.color:
        print(_colorize(prediction.text, prediction.spans, sys.stdout.isatty()), file=sys.stderr)
    print(json.dumps(payload["spans"], indent=2, ensure_ascii=False))
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    """Run the throughput benchmark bundled with the package."""
    from .bench import run_benchmark

    results = run_benchmark(
        checkpoint=args.checkpoint,
        document_tokens=args.document_tokens,
        batch_size=args.batch_size,
        repeats=args.repeats,
        moe_precision=args.moe_precision,
        compare_torch=args.compare_torch,
    )
    print(json.dumps(results, indent=2))
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    """Convert a reference checkpoint into MLX-native weights."""
    output = convert(
        args.checkpoint,
        args.output,
        dtype=args.dtype,
        quantize=args.bits is not None,
        group_size=args.group_size,
        bits=args.bits or 4,
    )
    print(f"wrote {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="opf-mlx", description=__doc__)
    parser.add_argument("--version", action="version", version=f"opf-mlx {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
        target.add_argument("--n-ctx", type=int, default=None)
        target.add_argument("--decode-mode", choices=("viterbi", "argmax"), default="viterbi")
        target.add_argument("--output-mode", choices=("typed", "redacted"), default="typed")
        target.add_argument("--no-trim", action="store_true")
        target.add_argument("--discard-overlapping-spans", action="store_true")
        target.add_argument(
            "--moe-precision",
            choices=("bfloat16", "float32"),
            default=None,
            help="expert arithmetic precision; float32 matches the reference exactly",
        )

    redact = sub.add_parser("redact", help="print the redacted text of a document")
    redact.add_argument("input", help="path to a file, or - for stdin")
    redact.add_argument("--json", action="store_true", help="emit the full structured result")
    add_common(redact)
    redact.set_defaults(func=_cmd_redact)

    spans = sub.add_parser("spans", help="print detected spans as JSON")
    spans.add_argument("input", help="path to a file, or - for stdin")
    spans.add_argument("--color", action="store_true", help="echo colour-coded text on stderr")
    add_common(spans)
    spans.set_defaults(func=_cmd_spans)

    bench = sub.add_parser("bench", help="measure throughput and peak memory")
    bench.add_argument("--document-tokens", type=int, default=32768)
    bench.add_argument("--batch-size", type=int, default=64)
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument(
        "--compare-torch", action="store_true", help="also time the PyTorch reference"
    )
    bench.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    bench.add_argument("--moe-precision", choices=("bfloat16", "float32"), default=None)
    bench.set_defaults(func=_cmd_bench)

    conv = sub.add_parser("convert", help="write MLX-native weights, optionally quantized")
    conv.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    conv.add_argument("-o", "--output", default="mlx_model")
    conv.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    conv.add_argument("--bits", type=int, default=None, help="quantize to this bit width")
    conv.add_argument("--group-size", type=int, default=64)
    conv.set_defaults(func=_cmd_convert)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        The process exit status.
    """
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
