"""BIOES label space, constrained Viterbi decoding and span reconstruction.

This mirrors the decoding half of the reference implementation: a linear-chain
CRF whose transitions encode the BIOES grammar, followed by span extraction,
character-offset mapping, whitespace trimming and placeholder rendering.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "BIAS_KEYS",
    "LabelSpace",
    "Span",
    "ViterbiDecoder",
    "labels_to_spans",
    "load_transition_biases",
    "placeholder_for",
    "redact_text",
    "spans_from_labels",
]

BACKGROUND = "O"
BOUNDARIES = ("B", "I", "E", "S")
REDACTED_LABEL = "redacted"
REDACTED_PLACEHOLDER = "<REDACTED>"

# Large negative score standing in for a forbidden CRF transition.
_NEG_INF = -1e9

BIAS_KEYS: tuple[str, ...] = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)

SPAN_CLASSES_BY_VERSION: dict[str, tuple[str, ...]] = {
    "v2": (
        BACKGROUND,
        "account_number",
        "private_address",
        "private_date",
        "private_email",
        "private_person",
        "private_phone",
        "private_url",
        "secret",
    ),
    "v4": (
        BACKGROUND,
        "private_person",
        "other_person",
        "personal_url",
        "other_url",
        "personal_location",
        "other_location",
        "personal_email",
        "other_email",
        "personal_phone",
        "other_phone",
        "personal_date",
        "other_date",
        "personal_id",
        "secret",
    ),
    "v7": (
        BACKGROUND,
        "personal_name",
        "personal_handle",
        "other_person",
        "personal_email",
        "other_email",
        "personal_phone",
        "other_phone",
        "personal_location",
        "other_location",
        "personal_url",
        "other_url",
        "personal_org",
        "personal_gov_id",
        "personal_fin_id",
        "personal_health_id",
        "personal_device_id",
        "personal_vehicle_id",
        "personal_property_id",
        "personal_edu_id",
        "personal_emp_id",
        "personal_membership_id",
        "personal_registry_id",
        "personal_date",
        "secret",
        "secret_url",
    ),
}
"""Span taxonomies keyed by the category version they belong to."""


def _expand(span_classes: Sequence[str]) -> tuple[str, ...]:
    """Expand span classes into the BIOES token-level label vocabulary."""
    labels = [BACKGROUND]
    for name in span_classes:
        if name == BACKGROUND:
            continue
        labels.extend(f"{prefix}-{name}" for prefix in BOUNDARIES)
    return tuple(labels)


TOKEN_CLASSES_BY_VERSION: dict[str, tuple[str, ...]] = {
    version: _expand(classes) for version, classes in SPAN_CLASSES_BY_VERSION.items()
}
_VERSION_BY_NUM_LABELS = {len(v): k for k, v in TOKEN_CLASSES_BY_VERSION.items()}


@dataclass(frozen=True)
class Span:
    """One detected span with character and byte offsets into the source text."""

    label: str
    start: int
    end: int
    text: str
    placeholder: str
    byte_start: int
    byte_end: int


@dataclass(frozen=True)
class LabelSpace:
    """Resolved BIOES label space for one checkpoint."""

    category_version: str
    span_classes: tuple[str, ...]
    token_classes: tuple[str, ...]
    token_to_span: tuple[int, ...]
    token_boundary: tuple[str | None, ...]
    background_token: int
    background_span: int = 0

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> LabelSpace:
        """Resolve the label space described by a checkpoint config.

        Args:
            config: Parsed ``config.json`` payload.

        Returns:
            The resolved label space.

        Raises:
            ValueError: If the config names an unknown taxonomy or label count.
        """
        version = config.get("category_version")
        if version is None:
            num_labels = int(config.get("num_labels") or 0)
            version = _VERSION_BY_NUM_LABELS.get(num_labels)
            if version is None:
                known = ", ".join(f"{k}:{v}" for k, v in sorted(_VERSION_BY_NUM_LABELS.items()))
                raise ValueError(f"num_labels={num_labels} matches no known taxonomy ({known})")
        version = str(version).strip().lower()
        if version not in TOKEN_CLASSES_BY_VERSION:
            raise ValueError(f"Unsupported category_version {version!r}")
        return cls.from_token_classes(version, TOKEN_CLASSES_BY_VERSION[version])

    @classmethod
    def from_token_classes(cls, version: str, token_classes: Sequence[str]) -> LabelSpace:
        """Build lookup tables from an ordered BIOES token-class vocabulary."""
        span_classes = [BACKGROUND]
        span_index = {BACKGROUND: 0}
        token_to_span: list[int] = []
        token_boundary: list[str | None] = []
        background_token: int | None = None

        for index, name in enumerate(token_classes):
            if name == BACKGROUND:
                background_token = index
                token_to_span.append(0)
                token_boundary.append(None)
                continue
            boundary, base = name.split("-", 1)
            if base not in span_index:
                span_index[base] = len(span_classes)
                span_classes.append(base)
            token_to_span.append(span_index[base])
            token_boundary.append(boundary)

        if background_token is None:
            raise ValueError("Token classes must include the background label 'O'")
        return cls(
            category_version=version,
            span_classes=tuple(span_classes),
            token_classes=tuple(token_classes),
            token_to_span=tuple(token_to_span),
            token_boundary=tuple(token_boundary),
            background_token=background_token,
        )


def load_transition_biases(path: str | Path | None) -> dict[str, float]:
    """Load Viterbi transition biases from a calibration artifact.

    Args:
        path: Calibration file, or ``None`` for the all-zero operating point.

    Returns:
        A mapping with exactly the keys of :data:`BIAS_KEYS`.

    Raises:
        ValueError: If the artifact does not expose a ``default`` operating point.
    """
    if path is None:
        return dict.fromkeys(BIAS_KEYS, 0.0)
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        raw = payload["operating_points"]["default"]["biases"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed calibration artifact: {path}") from exc
    missing = set(BIAS_KEYS) - set(raw)
    if missing:
        raise ValueError(f"Calibration artifact is missing biases: {sorted(missing)}")
    return {key: float(raw[key]) for key in BIAS_KEYS}


class ViterbiDecoder:
    """Constrained linear-chain Viterbi decoder over the BIOES label space."""

    def __init__(self, labels: LabelSpace, **biases: float) -> None:
        """Precompute the start, transition and end score tables.

        Args:
            labels: The label space to decode over.
            **biases: Optional transition biases named by :data:`BIAS_KEYS`.
        """
        unknown = set(biases) - set(BIAS_KEYS)
        if unknown:
            raise ValueError(f"Unknown transition biases: {sorted(unknown)}")
        self.labels = labels
        self.biases = {key: float(biases.get(key, 0.0)) for key in BIAS_KEYS}

        n = len(labels.token_classes)
        self.start_scores = np.full(n, _NEG_INF, dtype=np.float32)
        self.end_scores = np.full(n, _NEG_INF, dtype=np.float32)
        self.transitions = np.full((n, n), _NEG_INF, dtype=np.float32)

        for i in range(n):
            tag = labels.token_boundary[i]
            if tag in ("B", "S") or i == labels.background_token:
                self.start_scores[i] = 0.0
            if tag in ("E", "S") or i == labels.background_token:
                self.end_scores[i] = 0.0
            for j in range(n):
                if self._allowed(i, j):
                    self.transitions[i, j] = self._bias(i, j)

    def _is_background(self, index: int) -> bool:
        """Return whether one token class is the background class."""
        return (
            self.labels.token_to_span[index] == self.labels.background_span
            or index == self.labels.background_token
        )

    def _allowed(self, prev: int, nxt: int) -> bool:
        """Return whether the BIOES grammar allows ``prev -> nxt``.

        An open span (``B`` or ``I``) may only continue with ``I`` or ``E`` of the
        same class: it can never fall straight back to background, which is what
        forces every span to be closed explicitly.
        """
        labels = self.labels
        next_tag = labels.token_boundary[nxt]
        next_background = self._is_background(nxt)
        prev_tag = labels.token_boundary[prev]

        if self._is_background(prev) or prev_tag in ("E", "S"):
            return next_background or next_tag in ("B", "S")
        if prev_tag in ("B", "I"):
            return (
                next_tag in ("I", "E") and labels.token_to_span[prev] == labels.token_to_span[nxt]
            )
        return False

    def _bias(self, prev: int, nxt: int) -> float:
        """Return the bias applied to one allowed transition."""
        labels = self.labels
        prev_tag = labels.token_boundary[prev]
        next_tag = labels.token_boundary[nxt]
        next_background = self._is_background(nxt)

        if self._is_background(prev):
            if next_background:
                return self.biases["transition_bias_background_stay"]
            if next_tag in ("B", "S"):
                return self.biases["transition_bias_background_to_start"]
            return 0.0
        if prev_tag in ("B", "I"):
            same = labels.token_to_span[prev] == labels.token_to_span[nxt]
            if next_tag == "I" and same:
                return self.biases["transition_bias_inside_to_continue"]
            if next_tag == "E" and same:
                return self.biases["transition_bias_inside_to_end"]
            return 0.0
        if prev_tag in ("E", "S"):
            if next_background:
                return self.biases["transition_bias_end_to_background"]
            if next_tag in ("B", "S"):
                return self.biases["transition_bias_end_to_start"]
        return 0.0

    def decode(self, log_probs: np.ndarray) -> list[int]:
        """Decode one ``[seq_len, num_classes]`` log-probability matrix.

        Args:
            log_probs: Per-token label log-probabilities.

        Returns:
            The highest-scoring label id per token.
        """
        if log_probs.ndim != 2:
            raise ValueError("log_probs must have shape [seq_len, num_classes]")
        seq_len = log_probs.shape[0]
        if seq_len == 0:
            return []

        emissions = np.asarray(log_probs, dtype=np.float32)
        scores = emissions[0] + self.start_scores
        backpointers = np.empty((seq_len - 1, emissions.shape[1]), dtype=np.int32)
        for step in range(1, seq_len):
            candidates = scores[:, None] + self.transitions
            backpointers[step - 1] = candidates.argmax(axis=0)
            scores = candidates.max(axis=0) + emissions[step]

        if not np.isfinite(scores).any():
            return emissions.argmax(axis=1).tolist()

        path = np.empty(seq_len, dtype=np.int32)
        path[-1] = int((scores + self.end_scores).argmax())
        for step in range(seq_len - 2, -1, -1):
            path[step] = backpointers[step, path[step + 1]]
        return path.tolist()


def labels_to_spans(labels: Sequence[int], space: LabelSpace) -> list[tuple[int, int, int]]:
    """Group per-token BIOES labels into ``(span_class, token_start, token_end)`` triples.

    Args:
        labels: One label id per token, in token order.
        space: The label space the ids belong to.

    Returns:
        Token-index spans, half open on the right.
    """
    spans: list[tuple[int, int, int]] = []
    current: int | None = None
    start: int | None = None

    def flush(end: int) -> None:
        nonlocal current, start
        if current is not None and start is not None:
            spans.append((current, start, end))
        current = None
        start = None

    for index, label_id in enumerate(labels):
        span_class = space.token_to_span[label_id]
        tag = space.token_boundary[label_id]

        if span_class == space.background_span:
            flush(index)
            continue
        if tag == "S":
            flush(index)
            spans.append((span_class, index, index + 1))
        elif tag == "B":
            flush(index)
            current, start = span_class, index
        elif tag == "I":
            if current != span_class:
                flush(index)
                current, start = span_class, index
        elif tag == "E":
            if current != span_class or start is None:
                flush(index)
                spans.append((span_class, index, index + 1))
            else:
                spans.append((current, start, index + 1))
                current, start = None, None
    flush(len(labels))
    return spans


def _token_spans_to_char_spans(
    spans: Sequence[tuple[int, int, int]],
    starts: Sequence[int],
    ends: Sequence[int],
) -> list[tuple[int, int, int]]:
    """Convert token-index spans into offset spans using per-token boundaries."""
    converted: list[tuple[int, int, int]] = []
    for span_class, token_start, token_end in spans:
        if not 0 <= token_start < token_end <= len(starts):
            continue
        start, end = starts[token_start], ends[token_end - 1]
        if end > start:
            converted.append((span_class, start, end))
    return converted


def _trim_whitespace(
    spans: Sequence[tuple[int, int, int]], text: str
) -> list[tuple[int, int, int]]:
    """Trim leading and trailing whitespace from character spans."""
    trimmed: list[tuple[int, int, int]] = []
    for span_class, start, end in spans:
        if not 0 <= start < end <= len(text):
            continue
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            trimmed.append((span_class, start, end))
    return trimmed


def _discard_overlapping_by_label(
    spans: Sequence[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Drop spans that overlap another span carrying the same label."""
    by_label: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    for span_class, start, end in spans:
        by_label[span_class].append((start, end))

    kept: list[tuple[int, int, int]] = []
    for span_class, entries in by_label.items():
        accepted: list[tuple[int, int]] = []
        for start, end in sorted(entries, key=lambda s: (s[0], -(s[1] - s[0]))):
            if any(not (end <= a or start >= b) for a, b in accepted):
                continue
            accepted.append((start, end))
        kept.extend((span_class, start, end) for start, end in accepted)
    kept.sort(key=lambda s: (s[1], s[2], s[0]))
    return kept


def _select_non_overlapping(spans: Sequence[Span]) -> list[Span]:
    """Keep a left-to-right, non-overlapping subset of detected spans."""
    kept: list[Span] = []
    cursor = 0
    for span in sorted(spans, key=lambda s: (s.start, -(s.end - s.start), s.label)):
        if span.start < cursor or span.end <= span.start:
            continue
        kept.append(span)
        cursor = span.end
    return kept


def _char_to_byte_offsets(text: str) -> list[int]:
    """Return the UTF-8 byte offset of every character boundary in ``text``."""
    offsets = [0]
    cursor = 0
    for char in text:
        cursor += len(char.encode("utf-8"))
        offsets.append(cursor)
    return offsets


def placeholder_for(label: str) -> str:
    """Render the placeholder token substituted for one span label."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", label.upper()).strip("_")
    return f"<{normalized or 'REDACTED'}>"


def spans_from_labels(
    labels: Sequence[int],
    space: LabelSpace,
    *,
    text: str,
    char_starts: Sequence[int],
    char_ends: Sequence[int],
    byte_starts: Sequence[int],
    byte_ends: Sequence[int],
    trim_whitespace: bool = True,
    discard_overlapping: bool = False,
    output_mode: str = "typed",
) -> list[Span]:
    """Turn decoded token labels into rendered character spans.

    Args:
        labels: One label id per token.
        space: The label space the ids belong to.
        text: Source text the offsets refer to.
        char_starts: Per-token inclusive character start offsets.
        char_ends: Per-token exclusive character end offsets.
        byte_starts: Unused; kept so callers can pass a full :class:`~opf_mlx.tokenizer.Encoded`.
        byte_ends: Unused; kept so callers can pass a full :class:`~opf_mlx.tokenizer.Encoded`.
        trim_whitespace: Whether to strip whitespace from span edges.
        discard_overlapping: Whether to drop same-label overlapping spans.
        output_mode: ``"typed"`` keeps category labels, ``"redacted"`` collapses them.

    Returns:
        Non-overlapping detected spans in document order.
    """
    if output_mode not in ("typed", "redacted"):
        raise ValueError(f"Unsupported output_mode: {output_mode!r}")

    token_spans = labels_to_spans(labels, space)
    char_spans = _token_spans_to_char_spans(token_spans, char_starts, char_ends)
    if trim_whitespace:
        char_spans = _trim_whitespace(char_spans, text)
    if discard_overlapping:
        char_spans = _discard_overlapping_by_label(char_spans)

    # Byte offsets are derived from the final character offsets so that they stay
    # consistent after whitespace trimming.
    prefix_bytes = _char_to_byte_offsets(text)

    detected: list[Span] = []
    for span_class, start, end in char_spans:
        if not 0 <= start < end <= len(text):
            continue
        label = space.span_classes[span_class]
        detected.append(
            Span(
                label=label,
                start=start,
                end=end,
                text=text[start:end],
                placeholder=placeholder_for(label),
                byte_start=prefix_bytes[start],
                byte_end=prefix_bytes[end],
            )
        )

    selected = _select_non_overlapping(detected)
    if output_mode == "redacted":
        selected = [
            Span(
                label=REDACTED_LABEL,
                start=span.start,
                end=span.end,
                text=span.text,
                placeholder=REDACTED_PLACEHOLDER,
                byte_start=span.byte_start,
                byte_end=span.byte_end,
            )
            for span in selected
        ]
    return selected


def redact_text(text: str, spans: Sequence[Span]) -> str:
    """Substitute each detected span in ``text`` with its placeholder."""
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        pieces.append(text[cursor : span.start])
        pieces.append(span.placeholder)
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces)
