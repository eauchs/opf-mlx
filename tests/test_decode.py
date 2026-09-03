"""Unit tests for the label space, Viterbi decoder and span reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from opf_mlx.decode import (
    BIAS_KEYS,
    LabelSpace,
    Span,
    ViterbiDecoder,
    labels_to_spans,
    load_transition_biases,
    placeholder_for,
    redact_text,
    spans_from_labels,
)
from opf_mlx.tokenizer import Tokenizer

V2_CONFIG = {"num_labels": 33}


@pytest.fixture(scope="module")
def space() -> LabelSpace:
    """The v2 label space used by the published checkpoint."""
    return LabelSpace.from_config(V2_CONFIG)


@pytest.fixture(scope="module")
def decoder(space: LabelSpace) -> ViterbiDecoder:
    """A decoder with the checkpoint's all-zero operating point."""
    return ViterbiDecoder(space)


def label_id(space: LabelSpace, name: str) -> int:
    """Return the token-class id of one BIOES label name."""
    return space.token_classes.index(name)


def test_label_space_matches_published_ids(space: LabelSpace) -> None:
    """The v2 vocabulary must match the checkpoint's id2label ordering."""
    assert len(space.token_classes) == 33
    assert space.token_classes[0] == "O"
    assert space.token_classes[1:5] == (
        "B-account_number",
        "I-account_number",
        "E-account_number",
        "S-account_number",
    )
    assert space.span_classes[5] == "private_person"
    assert space.background_token == 0


def test_label_space_rejects_unknown_taxonomy() -> None:
    """An unrecognised label count must be refused rather than guessed."""
    with pytest.raises(ValueError):
        LabelSpace.from_config({"num_labels": 7})


def test_transition_grammar_is_bioes(decoder: ViterbiDecoder, space: LabelSpace) -> None:
    """An open span may not fall back to background without an explicit close."""
    allowed = decoder.transitions > -1e8
    b_person = label_id(space, "B-private_person")
    i_person = label_id(space, "I-private_person")
    e_person = label_id(space, "E-private_person")
    s_person = label_id(space, "S-private_person")
    b_email = label_id(space, "B-private_email")

    assert not allowed[b_person, 0], "B-x -> O must be forbidden"
    assert not allowed[i_person, 0], "I-x -> O must be forbidden"
    assert not allowed[b_person, b_email], "a span may not switch class mid-way"
    assert allowed[b_person, i_person]
    assert allowed[b_person, e_person]
    assert allowed[e_person, 0]
    assert allowed[s_person, b_email]
    assert allowed[0, b_person]
    assert not allowed[0, i_person], "a span may not start on I-x"

    assert decoder.start_scores[i_person] < -1e8
    assert decoder.start_scores[b_person] == 0.0
    assert decoder.end_scores[b_person] < -1e8
    assert decoder.end_scores[e_person] == 0.0


def test_decoder_repairs_an_illegal_argmax(decoder: ViterbiDecoder, space: LabelSpace) -> None:
    """A locally-preferred but ungrammatical path must be replaced by a legal one."""
    b_person = label_id(space, "B-private_person")
    e_person = label_id(space, "E-private_person")
    i_person = label_id(space, "I-private_person")

    log_probs = np.full((4, 33), -20.0, dtype=np.float32)
    log_probs[0, 0] = -0.1
    log_probs[1, b_person] = -0.1
    log_probs[2, 0] = -0.1  # illegal: closes the span without an E tag
    log_probs[2, i_person] = -0.5
    log_probs[3, 0] = -0.1
    log_probs[3, e_person] = -0.5

    assert log_probs.argmax(axis=1).tolist() == [0, b_person, 0, 0]
    assert decoder.decode(log_probs) == [0, b_person, i_person, e_person]


def test_decoder_handles_empty_and_single_token(decoder: ViterbiDecoder) -> None:
    """Degenerate sequence lengths must not raise."""
    assert decoder.decode(np.zeros((0, 33), dtype=np.float32)) == []
    assert decoder.decode(np.zeros((1, 33), dtype=np.float32)) == [0]


def test_decoder_rejects_unknown_bias() -> None:
    """Only the six documented transition biases are accepted."""
    with pytest.raises(ValueError):
        ViterbiDecoder(LabelSpace.from_config(V2_CONFIG), transition_bias_made_up=1.0)


def test_transition_bias_is_applied(space: LabelSpace) -> None:
    """A background-to-start bias must show up on exactly those edges."""
    biased = ViterbiDecoder(space, transition_bias_background_to_start=2.5)
    b_person = label_id(space, "B-private_person")
    assert biased.transitions[0, b_person] == pytest.approx(2.5)
    assert biased.transitions[0, 0] == pytest.approx(0.0)


def test_labels_to_spans_boundaries(space: LabelSpace) -> None:
    """BIOES sequences must group into the expected token spans."""
    seq = [
        0,
        label_id(space, "B-private_person"),
        label_id(space, "I-private_person"),
        label_id(space, "E-private_person"),
        0,
        label_id(space, "S-private_email"),
    ]
    assert labels_to_spans(seq, space) == [(5, 1, 4), (4, 5, 6)]


def test_labels_to_spans_closes_dangling_span(space: LabelSpace) -> None:
    """A span left open at the end of the sequence is still emitted."""
    seq = [label_id(space, "B-secret"), label_id(space, "I-secret")]
    assert labels_to_spans(seq, space) == [(8, 0, 2)]


def test_placeholder_rendering() -> None:
    """Placeholders must match the reference CLI's rendering."""
    assert placeholder_for("private_person") == "<PRIVATE_PERSON>"
    assert placeholder_for("account_number") == "<ACCOUNT_NUMBER>"
    assert placeholder_for("!!") == "<REDACTED>"


def test_redact_text_substitutes_in_order() -> None:
    """Placeholders replace spans left to right without shifting later offsets."""
    text = "call Ann on 0600"
    spans = (
        Span("private_person", 5, 8, "Ann", "<PRIVATE_PERSON>", 5, 8),
        Span("private_phone", 12, 16, "0600", "<PRIVATE_PHONE>", 12, 16),
    )
    assert redact_text(text, spans) == "call <PRIVATE_PERSON> on <PRIVATE_PHONE>"
    assert redact_text(text, ()) == text


def test_spans_trim_whitespace_and_report_byte_offsets(space: LabelSpace) -> None:
    """Spans over accented text must carry correct character and byte offsets."""
    tokenizer = Tokenizer()
    text = "Écrire à Élodie Beauchamp demain."
    encoded = tokenizer.encode_with_offsets(text)
    start = text.index("Élodie")
    end = text.index("Beauchamp") + len("Beauchamp")

    labels = [0] * len(encoded.token_ids)
    covered = [
        index
        for index, (a, b) in enumerate(zip(encoded.char_starts, encoded.char_ends, strict=True))
        if a >= start - 1 and b <= end
    ]
    labels[covered[0]] = label_id(space, "B-private_person")
    for index in covered[1:-1]:
        labels[index] = label_id(space, "I-private_person")
    labels[covered[-1]] = label_id(space, "E-private_person")

    detected = spans_from_labels(
        labels,
        space,
        text=encoded.text,
        char_starts=encoded.char_starts,
        char_ends=encoded.char_ends,
        byte_starts=encoded.byte_starts,
        byte_ends=encoded.byte_ends,
    )
    assert len(detected) == 1
    span = detected[0]
    assert span.text == text[span.start : span.end]
    assert not span.text[0].isspace() and not span.text[-1].isspace()
    assert span.byte_start == len(text[: span.start].encode("utf-8"))
    assert span.byte_end == len(text[: span.end].encode("utf-8"))
    assert span.byte_start != span.start, "the accented prefix must shift byte offsets"


def test_redacted_output_mode_collapses_labels(space: LabelSpace) -> None:
    """The redacted output mode hides the model's categories."""
    tokenizer = Tokenizer()
    encoded = tokenizer.encode_with_offsets("Harry")
    labels = [label_id(space, "S-private_person")] + [0] * (len(encoded.token_ids) - 1)
    detected = spans_from_labels(
        labels,
        space,
        text=encoded.text,
        char_starts=encoded.char_starts,
        char_ends=encoded.char_ends,
        byte_starts=encoded.byte_starts,
        byte_ends=encoded.byte_ends,
        output_mode="redacted",
    )
    assert [span.label for span in detected] == ["redacted"]
    assert [span.placeholder for span in detected] == ["<REDACTED>"]


def test_calibration_artifact_round_trip(tmp_path: Path) -> None:
    """Transition biases load from the checkpoint's calibration artifact."""
    assert load_transition_biases(None) == dict.fromkeys(BIAS_KEYS, 0.0)

    artifact = tmp_path / "viterbi_calibration.json"
    biases = {key: float(index) for index, key in enumerate(BIAS_KEYS)}
    artifact.write_text(json.dumps({"operating_points": {"default": {"biases": biases}}}))
    assert load_transition_biases(artifact) == biases

    artifact.write_text(json.dumps({"operating_points": {"default": {"biases": {}}}}))
    with pytest.raises(ValueError):
        load_transition_biases(artifact)


def test_tokenizer_offsets_cover_the_text() -> None:
    """Token offsets must tile the source text exactly."""
    tokenizer = Tokenizer()
    text = "Le rendez-vous du 15/01/2026 à 9 h 30, chez Élodie."
    encoded = tokenizer.encode_with_offsets(text)
    assert not encoded.mismatch
    assert encoded.text == text
    assert encoded.char_starts[0] == 0
    assert encoded.char_ends[-1] == len(text)
    assert encoded.byte_ends[-1] == len(text.encode("utf-8"))
    assert tokenizer.decode(encoded.token_ids) == text


def test_tokenizer_handles_empty_text() -> None:
    """Empty input yields no tokens and no offsets."""
    encoded = Tokenizer().encode_with_offsets("")
    assert encoded.token_ids == ()
    assert encoded.char_starts == ()
