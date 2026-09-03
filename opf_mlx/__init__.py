"""MLX port of OpenAI Privacy Filter, a bidirectional PII span tagger.

Typical use:

    >>> from opf_mlx import redact
    >>> redacted, spans = redact("My name is Harry Potter")  # doctest: +SKIP
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import numpy as np

from .convert import DEFAULT_CHECKPOINT, download_reference_checkpoint, load_model
from .decode import (
    LabelSpace,
    Span,
    ViterbiDecoder,
    load_transition_biases,
    redact_text,
    spans_from_labels,
)
from .model import Model, ModelArgs
from .tokenizer import Tokenizer

__all__ = [
    "DEFAULT_CHECKPOINT",
    "PrivacyFilter",
    "Prediction",
    "Span",
    "__version__",
    "redact",
    "spans",
]

__version__ = "0.1.0"


@dataclass(frozen=True)
class Prediction:
    """Result of one redaction call."""

    text: str
    redacted_text: str
    spans: tuple[Span, ...]
    decoded_mismatch: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the prediction."""
        return {
            "text": self.text,
            "redacted_text": self.redacted_text,
            "decoded_mismatch": self.decoded_mismatch,
            "spans": [
                {
                    "label": span.label,
                    "start": span.start,
                    "end": span.end,
                    "byte_start": span.byte_start,
                    "byte_end": span.byte_end,
                    "text": span.text,
                    "placeholder": span.placeholder,
                }
                for span in self.spans
            ],
        }


class PrivacyFilter:
    """Loaded model, tokenizer and decoder, reusable across calls."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        *,
        n_ctx: int | None = None,
        decode_mode: str = "viterbi",
        trim_whitespace: bool = True,
        discard_overlapping_spans: bool = False,
        output_mode: str = "typed",
        moe_precision: str | None = None,
        attention_chunk_size: int | None = None,
    ) -> None:
        """Load one checkpoint and prepare the decoding pipeline.

        Args:
            checkpoint: Checkpoint directory, downloaded on first use if missing.
            n_ctx: Window length in tokens; defaults to the checkpoint's own value.
            decode_mode: ``"viterbi"`` for constrained decoding, ``"argmax"`` for none.
            trim_whitespace: Whether to strip whitespace from span edges.
            discard_overlapping_spans: Whether to drop same-label overlapping spans.
            output_mode: ``"typed"`` or ``"redacted"``.
            moe_precision: Optional expert dtype override, e.g. ``"float32"``.
            attention_chunk_size: Optional attention tile width.
        """
        self.checkpoint = Path(download_reference_checkpoint(checkpoint))
        self.model, self.args = load_model(
            self.checkpoint,
            moe_precision=moe_precision,
            attention_chunk_size=attention_chunk_size,
        )
        with (self.checkpoint / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)

        self.labels = LabelSpace.from_config(config)
        self.tokenizer = Tokenizer(self.args.encoding)
        self.n_ctx = int(n_ctx or self.args.default_n_ctx)
        self.trim_whitespace = trim_whitespace
        self.discard_overlapping_spans = discard_overlapping_spans
        self.output_mode = output_mode

        if decode_mode not in ("viterbi", "argmax"):
            raise ValueError(f"Unsupported decode_mode: {decode_mode!r}")
        self.decode_mode = decode_mode
        calibration = self.checkpoint / "viterbi_calibration.json"
        self.decoder = (
            ViterbiDecoder(
                self.labels,
                **load_transition_biases(calibration if calibration.is_file() else None),
            )
            if decode_mode == "viterbi"
            else None
        )

    def logits(self, token_ids: Sequence[int]) -> np.ndarray:
        """Run one forward pass and return ``[len(token_ids), num_labels]`` logits."""
        out = self.model(mx.array([list(token_ids)], dtype=mx.uint32))
        mx.eval(out)
        return np.array(out.astype(mx.float32))[0]

    def batch_logits(self, batch: Sequence[Sequence[int]]) -> list[np.ndarray]:
        """Run one padded forward pass over several sequences.

        Args:
            batch: Token id sequences, of possibly different lengths.

        Returns:
            One ``[len_i, num_labels]`` logit array per input sequence.
        """
        if not batch:
            return []
        lengths = [len(item) for item in batch]
        width = max(lengths)
        pad = self.tokenizer.pad_token_id
        padded = [list(item) + [pad] * (width - len(item)) for item in batch]
        mask = np.zeros((len(batch), width), dtype=bool)
        for row, length in enumerate(lengths):
            mask[row, :length] = True

        out = self.model(mx.array(padded, dtype=mx.uint32), mx.array(mask))
        mx.eval(out)
        logits = np.array(out.astype(mx.float32))
        return [logits[row, :length] for row, length in enumerate(lengths)]

    def _token_log_probs(self, token_ids: Sequence[int]) -> np.ndarray:
        """Return per-token label log-probabilities, aggregating over windows."""
        if len(token_ids) <= self.n_ctx:
            return _log_softmax(self.logits(token_ids))

        # Longer inputs are split into disjoint windows, exactly like the reference.
        chunks = [
            _log_softmax(self.logits(token_ids[start : start + self.n_ctx]))
            for start in range(0, len(token_ids), self.n_ctx)
        ]
        return np.concatenate(chunks, axis=0)

    def decode_labels(self, log_probs: np.ndarray) -> list[int]:
        """Decode per-token label ids from label log-probabilities."""
        if self.decoder is None:
            return log_probs.argmax(axis=1).tolist()
        decoded = self.decoder.decode(log_probs)
        if len(decoded) != log_probs.shape[0]:
            return log_probs.argmax(axis=1).tolist()
        return decoded

    def predict(self, text: str) -> Prediction:
        """Detect and redact privacy spans in one string.

        Args:
            text: The text to analyse.

        Returns:
            The prediction, including the redacted text and the detected spans.
        """
        encoded = self.tokenizer.encode_with_offsets(text)
        if not encoded.token_ids:
            return Prediction(text=text, redacted_text=text, spans=(), decoded_mismatch=False)

        labels = self.decode_labels(self._token_log_probs(encoded.token_ids))
        detected = spans_from_labels(
            labels,
            self.labels,
            text=encoded.text,
            char_starts=encoded.char_starts,
            char_ends=encoded.char_ends,
            byte_starts=encoded.byte_starts,
            byte_ends=encoded.byte_ends,
            trim_whitespace=self.trim_whitespace,
            discard_overlapping=self.discard_overlapping_spans,
            output_mode=self.output_mode,
        )
        return Prediction(
            text=encoded.text,
            redacted_text=redact_text(encoded.text, detected),
            spans=tuple(detected),
            decoded_mismatch=encoded.mismatch,
        )

    def predict_batch(self, texts: Sequence[str]) -> list[Prediction]:
        """Detect and redact privacy spans in several strings in one forward pass.

        Sequences longer than ``n_ctx`` fall back to the windowed single-text path.

        Args:
            texts: The texts to analyse.

        Returns:
            One prediction per input text, in input order.
        """
        encoded = [self.tokenizer.encode_with_offsets(text) for text in texts]
        batchable = [
            index
            for index, item in enumerate(encoded)
            if item.token_ids and len(item.token_ids) <= self.n_ctx
        ]
        results: list[Prediction | None] = [None] * len(texts)

        if batchable:
            logits = self.batch_logits([encoded[index].token_ids for index in batchable])
            for index, item_logits in zip(batchable, logits, strict=True):
                item = encoded[index]
                labels = self.decode_labels(_log_softmax(item_logits))
                detected = spans_from_labels(
                    labels,
                    self.labels,
                    text=item.text,
                    char_starts=item.char_starts,
                    char_ends=item.char_ends,
                    byte_starts=item.byte_starts,
                    byte_ends=item.byte_ends,
                    trim_whitespace=self.trim_whitespace,
                    discard_overlapping=self.discard_overlapping_spans,
                    output_mode=self.output_mode,
                )
                results[index] = Prediction(
                    text=item.text,
                    redacted_text=redact_text(item.text, detected),
                    spans=tuple(detected),
                    decoded_mismatch=item.mismatch,
                )

        for index, result in enumerate(results):
            if result is None:
                results[index] = self.predict(texts[index])
        return [result for result in results if result is not None]


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


@lru_cache(maxsize=1)
def _default_filter() -> PrivacyFilter:
    """Return the process-wide default filter, loading it on first use."""
    return PrivacyFilter()


def redact(text: str) -> tuple[str, tuple[Span, ...]]:
    """Redact one string with the default checkpoint.

    Args:
        text: The text to redact.

    Returns:
        A ``(redacted_text, spans)`` pair.
    """
    prediction = _default_filter().predict(text)
    return prediction.redacted_text, prediction.spans


def spans(text: str) -> tuple[Span, ...]:
    """Return the privacy spans detected in one string."""
    return _default_filter().predict(text).spans


# Re-exported so that `from opf_mlx import Model` keeps working for library users.
_ = (Model, ModelArgs, math)
