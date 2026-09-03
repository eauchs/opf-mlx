# opf-mlx

A native [MLX](https://github.com/ml-explore/mlx) implementation of [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter), the 1.5B-parameter (50M active) bidirectional token classifier that detects and redacts personally identifiable information. It runs entirely on Apple silicon, with no PyTorch at inference time.

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

Measured on an Apple M3 Max with 128 GB of unified memory (macOS 25.4, Python 3.12, MLX 0.32.2), under `caffeinate`, median of 5 timed runs after one warm-up. The document workload is one 32,768-token text in a single window; the batch workload is 64 SMS-length messages (160 characters) padded into one forward pass. Timings cover the forward pass only, not tokenization or Viterbi decoding.

| Variant | doc tok/s | doc peak GB | batch tok/s | batch msg/s | batch peak GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLX bf16 | 14,638 | 4.41 | 14,718 | 325 | 3.52 |
| MLX bf16, float32 experts | 9,905 | 10.71 | 9,458 | 209 | 8.91 |
| MLX 8-bit | 11,724 | 5.89 | 11,223 | 248 | 4.98 |
| MLX 4-bit | 12,540 | 5.24 | 8,569 | 189 | 4.45 |
| PyTorch reference, MPS | 167 | 13.09* | 154 | 3 | 13.09* |

\* Driver-allocated memory after the run; `torch.mps` exposes no peak counter. MLX figures come from `mx.get_peak_memory()`.

That is **87x on the document and 95x on the batch** against the reference. Most of that gap is not MLX being fast, it is the reference having no fast path on this hardware: its Triton grouped-matmul MoE kernels are CUDA-only, so on Apple silicon it falls back to a path that gathers per-expert weights into float32 tensors in chunks of 32 tokens — for a 32k-token document that is 1,024 gathers of roughly 400 MB per layer. The number is the honest answer to "what do I get on a Mac today", not a claim about the two frameworks in general.

Checkpoint sizes on disk: 2.6 GB in bfloat16, 1.4 GB at 8 bits, 754 MB at 4 bits.

## Parity

Both implementations are run over the fifty French and English strings in `tests/samples/samples.json`, which cover spelled-out phone numbers, official public addresses, API keys and credentials, dates, IBANs and negatives. Reproduce with `scripts/parity.py`.

**In float32 the port is exact.** Every one of the 50 label sequences and every one of the 50 span sets, byte offsets included, matches the reference:

| Baseline: reference in float32 | labels | spans | max abs logit diff |
| --- | ---: | ---: | ---: |
| MLX float32 | 50/50 | 50/50 | 2.9e-05 |

**In bfloat16 nothing is exact, including the reference against itself.** This checkpoint's residual stream reaches magnitudes around 2e4, where one bfloat16 ulp is 128, so a single ulp of difference in any matmul propagates. Running the reference on CPU and on MPS already disagrees on one sample out of fifty, with logits up to 6.4 apart. The port sits at that same noise floor:

| Baseline: reference on CPU, bfloat16 | labels | spans | max abs logit diff |
| --- | ---: | ---: | ---: |
| Reference on MPS (the noise floor) | 49/50 | 49/50 | 6.4 |
| MLX bf16 | 48/50 | 49/50 | 10.1 |
| MLX bf16, float32 experts | 47/50 | 49/50 | 10.1 |
| MLX 8-bit | 46/50 | 49/50 | 9.8 |
| MLX 4-bit | 40/50 | 41/50 | 11.8 |

The one bfloat16 sample where MLX and the CPU reference disagree is the same one the reference disagrees with itself on: a French phone number spelled out in words, where the span boundaries are identical and only the category differs in a three-way near-tie between `private_email`, `private_phone` and `secret`.

**8 bits is free, 4 bits is not.** Quantizing the experts, the attention projections and the embedding to 8 bits leaves span parity untouched at 49/50 while cutting the checkpoint from 2.6 GB to 1.4 GB. At 4 bits, 9 of the 50 samples break down as: two false positives where the reference detects nothing, one of them over an official public address; two dropped spans, a person name and an address; one account number relabelled as a date; three spans whose boundaries shift by a token or two; and the near-tie above. The MoE router and the classification head are left unquantized in both cases.

The three worked examples published with the model reproduce exactly, including Figure 1 of the model card and the `digit_words` adversarial example from Table 10, a phone number spelled out in words.

## Design notes

- **Attention.** Banded bidirectional attention over a 257-token window (128 left, 128 right, plus self), with a per-head attention sink and YaRN-scaled rotary embeddings. Rather than materializing the full `[T, T]` mask or a per-token unfolded key tensor, queries are processed in tiles of 256 positions against the 512 keys that tile can reach; the band mask is built per tile. Scores are accumulated in bfloat16 and the softmax runs in float32, matching the reference.
- **Mixture of experts.** 128 experts, top-4 per token, routed by a float32 gate. Routed tokens are sorted by expert id before `mx.gather_mm`, so each expert runs as one grouped matmul over its whole caseload instead of one matmul per routed token. That single change is worth 3.1x on a 4k-token document (3,490 to 10,752 tokens/s).
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
