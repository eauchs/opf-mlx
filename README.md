# opf-mlx

A native [MLX](https://github.com/ml-explore/mlx) implementation of [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter), the 1.5B-parameter (50M active) bidirectional token classifier that detects and redacts personally identifiable information. It runs entirely on Apple silicon, with no PyTorch at inference time.

It is **not the first MLX port** of this model: [`mlx-embeddings`](https://github.com/Blaizzy/mlx-embeddings) has carried the architecture since April 2026 and `mlx-community` publishes converted weights. What this repository adds is the decoding half that neither covers — constrained Viterbi, BIOES span reconstruction, character and byte offsets, redaction identical to the official CLI — and an attention path that holds up at long context. See [Relation to mlx-embeddings](#relation-to-mlx-embeddings).

The model tags every token of the input in a single bidirectional forward pass — there is no autoregressive loop — and a constrained Viterbi decoder turns those per-token logits into coherent BIOES spans over eight privacy categories: `account_number`, `private_address`, `private_date`, `private_email`, `private_person`, `private_phone`, `private_url` and `secret`.

## Install

```bash
uv venv --python 3.12 && uv pip install -e .
```

The checkpoint is downloaded from the Hub on first use into `~/.opf/privacy_filter`, the same location the official `opf` CLI uses, so the two share one copy of the weights.

## Usage

```python
from opf_mlx import redact

redacted, spans = redact(
    "Contact Sarah Lindqvist at sarah.lindqvist@example.org or on 020 7946 0958."
)
# 'Contact <PRIVATE_PERSON> at <PRIVATE_EMAIL> or on <PRIVATE_PHONE>.'
# spans[0] -> Span(label='private_person', start=8, end=23, byte_start=8, byte_end=23, ...)
```

Reuse a loaded model across calls, and batch short inputs into one forward pass:

```python
from opf_mlx import PrivacyFilter

pf = PrivacyFilter()
for prediction in pf.predict_batch(messages):
    print(prediction.redacted_text)
```

From the command line:

```bash
opf-mlx redact report.txt              # redacted text on stdout, summary on stderr
opf-mlx redact - --json < report.txt   # full structured result
opf-mlx spans report.txt --color       # detected spans as JSON
opf-mlx bench --compare-torch          # throughput and peak memory
opf-mlx convert --bits 8 -o mlx-8bit   # write quantized MLX weights
```

## Benchmark

Measured on an Apple M3 Max with 128 GB of unified memory (macOS 25.4, Python 3.12, MLX 0.32.2), under `caffeinate`, median of 5 timed runs after one warm-up, **one variant per process** so that co-resident models do not distort the figures. The document workload is one 32,768-token text in a single window; the batch workload is 64 SMS-length messages (160 characters) padded into one forward pass. Timings cover the forward pass only, not tokenization or Viterbi decoding.

| Variant | doc tok/s | doc peak GB | batch tok/s | batch msg/s | batch peak GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLX bf16 | 56,856 | 4.46 | 51,523 | 1,137 | 3.48 |
| MLX 4-bit | 50,750 | 2.74 | 44,544 | 983 | 1.75 |
| MLX 8-bit | 47,188 | 3.39 | 40,883 | 902 | 2.29 |
| MLX bf16, float32 experts | 39,054 | 8.12 | 27,919 | 616 | 6.24 |
| PyTorch reference, MPS | 203 | 12.16* | 160 | 4 | 12.16* |

\* Driver-allocated memory after the run; `torch.mps` exposes no peak counter. MLX figures come from `mx.get_peak_memory()`.

Against the reference that is **268x on the document and 297x on the batch**, taking both sides from the same process so the comparison stays apples to apples. Most of that gap is not MLX being fast, it is the reference having no fast path on this hardware: its Triton grouped-matmul MoE kernels are CUDA-only, so on Apple silicon it falls back to a path that gathers per-expert weights into float32 tensors in chunks of 32 tokens — for a 32k-token document that is 1,024 gathers of roughly 400 MB per layer. The number is the honest answer to "what do I get on a Mac today", not a claim about the two frameworks in general.

Quantization here buys memory, not speed: 4-bit more than halves peak memory but runs slower than bf16, because at this size the model is compute-bound and dequantization is pure overhead. Checkpoint sizes on disk: 2.6 GB in bfloat16, 1.4 GB at 8 bits, 754 MB at 4 bits.

## Parity

Both implementations are run over the fifty French and English strings in `tests/samples/samples.json`, which cover spelled-out phone numbers, official public addresses, API keys and credentials, dates, IBANs and negatives. Reproduce with `scripts/parity.py`.

**In float32 the port is exact.** Every one of the 50 label sequences and every one of the 50 span sets, offsets included, matches the reference:

| Baseline: reference in float32 | labels | spans | max abs logit diff |
| --- | ---: | ---: | ---: |
| MLX float32 | 50/50 | 50/50 | 3.3e-05 |

**In bfloat16 nothing is exact, including the reference against itself.** This checkpoint's residual stream reaches magnitudes around 2e4, where one bfloat16 ulp is 128, so a single ulp of difference in any matmul propagates. Running the reference on CPU and on MPS already disagrees on one sample out of fifty, with logits up to 6.4 apart:

| Baseline: reference on CPU, bfloat16 | labels | spans | max abs logit diff |
| --- | ---: | ---: | ---: |
| Reference on MPS (the noise floor) | 49/50 | 49/50 | 6.4 |
| MLX bf16 | 48/50 | 50/50 | 8.4 |
| MLX bf16, float32 experts | 48/50 | 50/50 | 8.0 |
| MLX 8-bit | 47/50 | 49/50 | 9.5 |
| MLX 4-bit | 39/50 | 42/50 | 11.6 |

Per-token argmax labels drift on two sequences out of fifty, but every decoded span set matches: the constrained decoder absorbs the noise that an unconstrained argmax would let through. That is one span set better than the reference manages against its own MPS backend.

**8 bits is nearly free, 4 bits is not.** Quantizing the experts, the attention projections and the embedding to 8 bits costs one span set out of fifty while cutting the checkpoint from 2.6 GB to 1.4 GB. At 4 bits, eight break: two spelled-out phone numbers, an IBAN, a social security number, three addresses and a date. The MoE router and the classification head are left unquantized in both cases.

The three worked examples published with the model reproduce exactly, including Figure 1 of the model card and the `digit_words` adversarial example from Table 10, a phone number spelled out in words.

## Relation to mlx-embeddings

[`mlx-embeddings`](https://github.com/Blaizzy/mlx-embeddings) added `openai_privacy_filter` in April 2026, and `mlx-community` publishes converted weights in bf16, 4/5/6/8-bit, mxfp4, mxfp8 and nvfp4. This repository is a second, independent implementation; it did not come first, and the weights it loads are theirs and OpenAI's.

Head to head against `mlx-embeddings` 0.1.1, same machine, alternating runs so neither side gets a cold or a thermally throttled slot: three rounds, median of 5 timed passes each, medians across rounds reported.

| Workload | mlx-embeddings 0.1.1 | opf-mlx |
| --- | ---: | ---: |
| 2,048 tokens | 44,153 tok/s | **52,693 tok/s** |
| 8,192 tokens | 27,567 tok/s | **56,832 tok/s** |
| 32,768 tokens | 8,165 tok/s, 10.62 GB | **41,574 tok/s, 4.46 GB** |
| 64 SMS-length messages | 48,543 tok/s, 1,071 msg/s | **52,153 tok/s, 1,151 msg/s** |
| Span sets matching the PyTorch reference | not applicable, see below | 50/50 |

The gap comes almost entirely from two places. `mlx-embeddings` builds the local attention band as a dense `[L, L]` mask, which is O(L²) and costs 10.6 GB at 32k; tiling the band keeps throughput flat in sequence length instead of collapsing. It also lets `mx.gather_mm` do the routed-token row indexing through `lhs_indices`, which on this shape is 4x slower than gathering the rows first. Both are ordinary engineering, not a difference in the model.

The tiling half has been offered back upstream as [Blaizzy/mlx-embeddings#76](https://github.com/Blaizzy/mlx-embeddings/pull/76); if it lands, their long-context numbers become comparable to these.

The substantive difference is elsewhere. `mlx-embeddings` stops at logits — its documented usage is `argmax` plus a `groupby` over token ids, which does not enforce the BIOES grammar, does not reconstruct character offsets, and does not reproduce the official CLI's spans. Everything in `decode.py` and `tokenizer.py` here has no counterpart upstream.

## Design notes

- **Attention.** Banded bidirectional attention over a 257-token window (128 left, 128 right, plus self), with a per-head attention sink and YaRN-scaled rotary embeddings. Rather than materializing the full `[T, T]` mask or a per-token unfolded key tensor, queries are processed in tiles of 256 positions against the 512 keys that tile can reach, each tile going through the fused `mx.fast.scaled_dot_product_attention` with its own small mask. Peak memory then depends on the tile, not the document: 4.5 GB at 32k instead of 10.6 GB. `banded_attention_reference` keeps the unfused float32-softmax version as an executable specification, and a test holds the two to 1e-4 of each other.
- **Mixture of experts.** 128 experts, top-4 per token, routed by a float32 gate. Routed tokens are sorted by expert id before `mx.gather_mm`, so each expert runs as one grouped matmul over its whole caseload instead of one matmul per routed token. Sorting is not novel — `mlx_lm.models.switch_layers.SwitchGLU` does it internally — but *how* the sorted rows reach the matmul matters: passing `lhs_indices` and letting `gather_mm` gather them costs 24.2 ms per layer on an 8k-token document, against 5.9 ms for an explicit `t[rows]` gather followed by a plain `gather_mm`. That one line is worth more than twice the end-to-end throughput.
- **Tokenizer.** The reference tokenizes with `tiktoken` using the encoding named in the checkpoint config (`o200k_base`), not with a Hugging Face tokenizer, and reports character offsets. Both are reproduced here so span boundaries are directly comparable with the official CLI. Byte offsets are exposed alongside as `byte_start` / `byte_end`.
- **Decoder.** The Viterbi transition table encodes the BIOES grammar: an open span (`B` or `I`) may only continue with `I` or `E` of the same class and can never fall back to background without an explicit close. The six transition biases are read from the checkpoint's `viterbi_calibration.json`, whose published operating point is all zeros.
- **Conventions.** Plain `nn.Module` subclasses, an args dataclass built from `config.json`, and a `sanitize()` that maps the reference tensor names onto the module tree, mirroring how models are written in `mlx-lm`. Nothing in `mlx-lm` is monkey-patched, and one `mx.eval()` per forward pass keeps the graph free of per-layer synchronisation.

## Development

```bash
uv pip install -e '.[dev]'   # pulls in torch and the reference opf package
pytest                       # unit tests plus parity against the reference
ruff check . && ruff format --check .

OPF_MOE_TRITON=0 python scripts/parity.py --quantized mlx-8bit --quantized mlx-4bit
caffeinate -dimsu python scripts/bench.py --compare-torch --repeats 5
```

`OPF_MOE_TRITON=0` selects the reference's pure-PyTorch expert path; its default kernels require Triton, which is unavailable on Apple silicon.

## License and attribution

Apache License 2.0, see [LICENSE](LICENSE).

The model weights are not distributed here. They are published by OpenAI at [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) under Apache 2.0 and downloaded at runtime. The decoding grammar, label taxonomy, span reconstruction rules and calibration format follow the reference implementation at [github.com/openai/privacy-filter](https://github.com/openai/privacy-filter), also Apache 2.0. Copyright for the weights and the reference implementation remains with OpenAI.
