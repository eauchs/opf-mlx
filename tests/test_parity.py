"""Parity tests against the official PyTorch reference implementation.

The reference forward pass runs in bfloat16 and this checkpoint's residual stream
reaches magnitudes around 2e4, where one bfloat16 ulp is 128. Small differences in
matmul accumulation therefore change the output of *any* two implementations, the
reference included: running it on CPU and on MPS already disagrees on one of the
fifty samples below. Exact parity is consequently asserted in float32, where the
arithmetic is well conditioned, and the bfloat16 comparison is asserted against
that measured noise floor.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OPF_MOE_TRITON", "0")

from opf_mlx import PrivacyFilter  # noqa: E402
from opf_mlx.convert import DEFAULT_CHECKPOINT  # noqa: E402

torch = pytest.importorskip("torch", reason="the PyTorch reference is a dev dependency")
pytest.importorskip("opf", reason="the PyTorch reference is a dev dependency")

from opf._core.decoding import build_sequence_decoder  # noqa: E402
from opf._core.runtime import load_inference_runtime, predict_text  # noqa: E402

SAMPLES_PATH = Path(__file__).parent / "samples" / "samples.json"
SAMPLES = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))

# All fifty currently match, but this is a noise-floor assertion rather than a
# target: one sample is decided by a three-way near-tie between private_email,
# private_phone and secret, and the reference itself resolves it differently on
# CPU and on MPS.
BF16_MIN_MATCHING_SPANS = 49

MODEL_CARD_EXAMPLES = [
    # Figure 1 of the OpenAI Privacy Filter model card.
    (
        "Ben Morgan lives at 123rd St. Call him at 1234567890.",
        [("private_person", 0, 10), ("private_address", 20, 29), ("private_phone", 42, 52)],
    ),
    # Table 10 of the model card, "digit_words": a phone number spelled out in words.
    (
        "If you need to reach me directly, my personal cell number is "
        "two six eight-seven two two-one zero four nine.",
        [("private_phone", 61, 107)],
    ),
    # The worked example of the Hugging Face model card.
    (
        "My name is Harry Potter and my email is harry.potter@hogwarts.edu.",
        [("private_person", 11, 23), ("private_email", 40, 65)],
    ),
]

pytestmark = pytest.mark.skipif(
    not (DEFAULT_CHECKPOINT / "config.json").is_file(),
    reason=f"no checkpoint at {DEFAULT_CHECKPOINT}",
)


def _spans(prediction) -> list[tuple[str, int, int]]:
    """Return comparable ``(label, start, end)`` triples for one prediction."""
    return [(span.label, span.start, span.end) for span in prediction.spans]


def _reference(checkpoint: Path):
    """Build the reference runtime and its Viterbi decoder for one checkpoint."""
    runtime = load_inference_runtime(
        checkpoint=str(checkpoint),
        device_name="cpu",
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


def _reference_logits(runtime, token_ids) -> np.ndarray:
    """Run one reference forward pass and return float32 logits."""
    tokens = torch.tensor([list(token_ids)], dtype=torch.int32)
    with torch.inference_mode():
        logits = runtime.model(tokens, attention_mask=torch.ones_like(tokens, dtype=torch.bool))
    return logits.float().numpy()[0]


@pytest.fixture(scope="module")
def float32_checkpoint(tmp_path_factory) -> Path:
    """A view of the checkpoint whose config asks the reference for float32 weights."""
    target = tmp_path_factory.mktemp("opf-float32")
    config = json.loads((DEFAULT_CHECKPOINT / "config.json").read_text(encoding="utf-8"))
    config["param_dtype"] = "fp32"
    (target / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in ("model.safetensors", "viterbi_calibration.json"):
        source = DEFAULT_CHECKPOINT / name
        if source.is_file():
            (target / name).symlink_to(source)
    return target


@pytest.fixture(scope="module")
def mlx_bfloat16() -> PrivacyFilter:
    """The MLX model as shipped, in the checkpoint's own bfloat16."""
    return PrivacyFilter()


@pytest.fixture(scope="module")
def mlx_float32() -> PrivacyFilter:
    """The MLX model promoted to float32 end to end."""
    import mlx.core as mx

    pf = PrivacyFilter()
    pf.model.set_dtype(mx.float32)
    mx.eval(pf.model.parameters())
    return pf


def test_samples_cover_both_languages() -> None:
    """The sample set must stay at fifty strings spanning French and English."""
    assert len(SAMPLES) == 50
    assert len({sample["id"] for sample in SAMPLES}) == 50
    assert {sample["lang"] for sample in SAMPLES} == {"en", "fr"}
    kinds = {sample["kind"] for sample in SAMPLES}
    assert {"phone-spelled", "public-address", "secret", "date"} <= kinds


def test_float32_logit_parity(float32_checkpoint: Path, mlx_float32: PrivacyFilter) -> None:
    """In float32 the port must reproduce the reference logits and every argmax label."""
    runtime, _ = _reference(float32_checkpoint)
    worst = 0.0
    for sample in SAMPLES:
        token_ids = mlx_float32.tokenizer.encode(sample["text"])
        reference = _reference_logits(runtime, token_ids)
        mine = mlx_float32.logits(token_ids)
        worst = max(worst, float(np.abs(reference - mine).max()))
        assert reference.argmax(-1).tolist() == mine.argmax(-1).tolist(), sample["id"]
    print(f"\nfloat32 max |logit difference| over 50 samples: {worst:.3g}")
    assert worst < 1e-3


def test_float32_span_parity(float32_checkpoint: Path, mlx_float32: PrivacyFilter) -> None:
    """In float32 every decoded span must match the reference exactly."""
    runtime, decoder = _reference(float32_checkpoint)
    for sample in SAMPLES:
        expected = predict_text(runtime, sample["text"], decoder=decoder)
        got = mlx_float32.predict(sample["text"])
        assert _spans(got) == [(span.label, span.start, span.end) for span in expected.spans], (
            sample["id"]
        )
        assert got.redacted_text == "".join(_redact(expected.text, expected.spans)), sample["id"]


def _redact(text: str, spans) -> list[str]:
    """Rebuild the reference's redacted text from its detected spans."""
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        pieces.append(text[cursor : span.start])
        pieces.append(span.placeholder)
        cursor = span.end
    pieces.append(text[cursor:])
    return pieces


def test_bfloat16_span_parity(mlx_bfloat16: PrivacyFilter) -> None:
    """In bfloat16 the port must stay at the reference's own numerical noise floor."""
    runtime, decoder = _reference(DEFAULT_CHECKPOINT)
    matching = 0
    worst = 0.0
    diverging: list[str] = []
    for sample in SAMPLES:
        token_ids = mlx_bfloat16.tokenizer.encode(sample["text"])
        worst = max(
            worst,
            float(
                np.abs(_reference_logits(runtime, token_ids) - mlx_bfloat16.logits(token_ids)).max()
            ),
        )
        expected = [
            (span.label, span.start, span.end)
            for span in predict_text(runtime, sample["text"], decoder=decoder).spans
        ]
        if _spans(mlx_bfloat16.predict(sample["text"])) == expected:
            matching += 1
        else:
            diverging.append(sample["id"])
    print(
        f"\nbfloat16 spans identical on {matching}/50 samples "
        f"(diverging: {diverging}), max |logit difference| {worst:.3g}"
    )
    assert matching >= BF16_MIN_MATCHING_SPANS


@pytest.mark.parametrize(
    "text,expected", MODEL_CARD_EXAMPLES, ids=["figure1", "digit_words", "hf_card"]
)
def test_model_card_examples(
    mlx_bfloat16: PrivacyFilter, text: str, expected: list[tuple[str, int, int]]
) -> None:
    """The worked examples published with the model must reproduce exactly."""
    assert _spans(mlx_bfloat16.predict(text)) == expected


def test_spelled_out_phone_number_is_redacted(mlx_bfloat16: PrivacyFilter) -> None:
    """An English phone number written out in words must still be detected.

    This is the ``digit_words`` adversarial axis of the model card. The model is
    Coverage is uneven even in English, and the model misses the same construction
    in French entirely; that is a property of the weights, not of this port, and
    the sample sweep above checks that the port reproduces the reference's
    behaviour on those strings exactly.
    """
    for text in (
        "Call me at oh six one two three four five six seven eight.",
        "Ring me on double oh four four seven seven zero zero nine zero zero one two three.",
        "If you need to reach me directly, my personal cell number is "
        "two six eight-seven two two-one zero four nine.",
    ):
        prediction = mlx_bfloat16.predict(text)
        assert any(span.label == "private_phone" for span in prediction.spans), text
        assert "<PRIVATE_PHONE>" in prediction.redacted_text


def test_batched_and_single_paths_agree(mlx_bfloat16: PrivacyFilter) -> None:
    """Padded batched inference must produce the same spans as one-at-a-time inference."""
    texts = [sample["text"] for sample in SAMPLES[:16]]
    for text, batched in zip(texts, mlx_bfloat16.predict_batch(texts), strict=True):
        assert _spans(batched) == _spans(mlx_bfloat16.predict(text)), text


def test_byte_offsets_match_character_offsets(mlx_bfloat16: PrivacyFilter) -> None:
    """Reported byte offsets must index the same substring as the character offsets."""
    for sample in SAMPLES:
        prediction = mlx_bfloat16.predict(sample["text"])
        encoded = prediction.text.encode("utf-8")
        for span in prediction.spans:
            assert encoded[span.byte_start : span.byte_end].decode("utf-8") == span.text
