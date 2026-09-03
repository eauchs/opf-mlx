"""Tokenization and character-offset mapping for the Privacy Filter port.

The reference implementation tokenizes with ``tiktoken`` using the encoding named
in the checkpoint config (``o200k_base``), not with a Hugging Face tokenizer, and
reports span boundaries as character offsets into the source string. Both choices
are reproduced here so that spans are byte-for-byte comparable with the official
``opf`` CLI.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

import tiktoken

__all__ = ["Encoded", "Tokenizer"]


@dataclass(frozen=True)
class Encoded:
    """One tokenized string together with its per-token offsets.

    Attributes:
        token_ids: Token ids produced by the encoding.
        text: The text the offsets refer to; the tokenizer round-trip decode when
            it differs from the input, otherwise the input itself.
        char_starts: Inclusive start character offset of each token in ``text``.
        char_ends: Exclusive end character offset of each token in ``text``.
        byte_starts: Inclusive start byte offset of each token in ``text``.
        byte_ends: Exclusive end byte offset of each token in ``text``.
        mismatch: Whether the decoded text differed from the input text.
    """

    token_ids: tuple[int, ...]
    text: str
    char_starts: tuple[int, ...]
    char_ends: tuple[int, ...]
    byte_starts: tuple[int, ...]
    byte_ends: tuple[int, ...]
    mismatch: bool


class Tokenizer:
    """Thin wrapper over a ``tiktoken`` encoding with offset reconstruction."""

    def __init__(self, encoding_name: str = "o200k_base") -> None:
        """Load one named ``tiktoken`` encoding."""
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.pad_token_id = int(self.encoding.eot_token)
        self._bytes_cache: dict[int, bytes] = {}

    def encode(self, text: str) -> list[int]:
        """Encode ``text``, treating special-token strings as ordinary text."""
        return [int(token) for token in self.encoding.encode(text, allowed_special="all")]

    def _token_bytes(self, token_id: int) -> bytes:
        """Return the raw bytes of one token, memoized per encoding instance."""
        cached = self._bytes_cache.get(token_id)
        if cached is None:
            cached = self.encoding.decode_single_token_bytes(token_id)
            self._bytes_cache[token_id] = cached
        return cached

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token ids back into text, replacing invalid byte sequences."""
        return b"".join(self._token_bytes(int(t)) for t in token_ids).decode(
            "utf-8", errors="replace"
        )

    def _offsets(self, token_ids: Sequence[int], text: str) -> tuple[list[int], ...]:
        """Map tokens onto character and byte offsets within ``text``."""
        char_byte_starts: list[int] = []
        char_byte_ends: list[int] = []
        cursor = 0
        for char in text:
            char_byte_starts.append(cursor)
            cursor += len(char.encode("utf-8"))
            char_byte_ends.append(cursor)

        char_starts: list[int] = []
        char_ends: list[int] = []
        byte_starts: list[int] = []
        byte_ends: list[int] = []
        byte_cursor = 0
        for token_id in token_ids:
            start = byte_cursor
            end = start + len(self._token_bytes(int(token_id)))
            byte_cursor = end
            byte_starts.append(start)
            byte_ends.append(end)
            # A token may start mid-character in malformed UTF-8; clamp to whole chars.
            start_idx = bisect_right(char_byte_ends, start)
            end_idx = bisect_left(char_byte_starts, end)
            char_starts.append(start_idx)
            char_ends.append(max(end_idx, start_idx))
        return char_starts, char_ends, byte_starts, byte_ends

    def encode_with_offsets(self, text: str) -> Encoded:
        """Encode ``text`` and compute per-token character and byte offsets.

        Args:
            text: The string to tokenize.

        Returns:
            An :class:`Encoded` record. When the tokenizer round-trip does not
            reproduce the input exactly, offsets refer to the decoded text and
            ``mismatch`` is set, matching the reference implementation.
        """
        token_ids = self.encode(text)
        if not token_ids:
            return Encoded((), text, (), (), (), (), False)

        decoded = self.decode(token_ids)
        mismatch = decoded != text
        source = decoded if mismatch else text
        char_starts, char_ends, byte_starts, byte_ends = self._offsets(token_ids, source)
        return Encoded(
            token_ids=tuple(token_ids),
            text=source,
            char_starts=tuple(char_starts),
            char_ends=tuple(char_ends),
            byte_starts=tuple(byte_starts),
            byte_ends=tuple(byte_ends),
            mismatch=mismatch,
        )
