# TinyLlama-1.1B / Llama-2 status — what's shipped, what's left

Snapshot of how much of the Llama-family decoder is wired up in
aricode-pack today.  The Llama-2 architecture has five
distinguishing pieces vs the GPT-2 family (which already ships):
RMSNorm, SwiGLU FFN, RoPE, grouped-query attention, and an
autoregressive decode loop driven by sampling.

**Status: all five are implemented and locked in by regression
tests.**  The remaining work to deploy a real fine-tuned
Llama-2-7B is checkpoint wrangling, not codegen or stdlib.

## What's done

### KV-cache attention

`aricode-ml/attention_kv_f32.ari` (single head) and
`attention_mh_kv_f32.ari` (multi-head) provide one-token-at-a-time
SDPA against an append-only cache.  The pack-side arch entry is
`["multi_head_attention_kv", max_seq, d_model, n_heads]` (handled
in `aricode_pack.py` near the `multi_head_attention_kv` branches
of `tensor_specs` / `gen_forward` / `gen_scratch`).  Cached K and
V live in `[max_seq, d_head]` slabs per head.

This is the backbone of GPT-2-small (`examples/gpt2_small/`)
and shows up uniformly in any decoder that doesn't use GQA.

### Sampling

`aricode-ml/sampling_f32.ari` ships argmax (greedy) and
temperature softmax sampling over a logit row.  The decoder
`gen_decoder_main` in pack.py wires the LM head's output through
this kernel each step.  `--max-new-tokens` controls how many
steps the prefill-then-sample loop runs.

### RMSNorm

`aricode-ml/rmsnorm_f32.ari` (Llama / Mistral style — γ scale
only, no β).  Pack arch entry `["rmsnorm", dim]`.  HF state-dict
keys: `model.layers.{i}.input_layernorm.weight`,
`model.layers.{i}.post_attention_layernorm.weight`,
`model.norm.weight`.

A native AVX2 builtin `arr_f32_rmsnorm` is **not** shipped — the
implementation uses the existing `arr_f32_dot` (for the sum-of-
squares pass) plus a scalar normalize loop.  At Llama-2-7B's
d_model = 4096 this is dwarfed by FFN and attention; a SIMD
builtin would be a future micro-optimization, not a correctness
gap.

### SwiGLU FFN

Pack arch entry `["swiglu_ffn", d_model, d_ffn]` bundles three
Linears (gate / up / down) and the elementwise `silu(gate(x)) *
up(x)` step into one op.  Bundling keeps the two intermediate
`[d_ffn]` scratches local to the op (Llama-2-7B's d_ffn is
11008; exposing them as separate Linears would force every
activation buffer in the model to grow to that size).

State-dict keys: `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
(no biases, Llama convention).  SiLU = `x * sigmoid(x)` is
implemented as a stdlib helper using scalar `math_exp`; a
vectorized builtin is deferrable.

### RoPE

`aricode-ml/rope_f32.ari` allocates a `[max_seq × d_head]` table
of interleaved (cos, sin) pairs at startup using `math_sin` /
`math_cos`, then `rope_apply_f32(table, vec, pos, d_head)`
rotates an in-place vector by the cached pair for that position.
The interleaved-pair form matches the
`transformers.LlamaRotaryEmbedding` reference (cached K is
already-rotated; Q rotates per step).

### GQA + RoPE attention

`aricode-ml/attention_gqa_kv_f32.ari` is the merged version: per
step, Q is built per head (`n_heads` of them), K/V are built per
KV head (`n_kv_heads` of them), RoPE is applied to Q for the
current head and K for the current KV head before the K-write,
then the SDPA proceeds with the broadcast-against-cache pattern.

Pack arch entry `["multi_head_attention_gqa_kv", max_seq,
d_model, n_heads, n_kv_heads, theta]`.  When `n_kv_heads ==
n_heads` (Llama-2-7B has 32/32) the path collapses to standard
MHA with RoPE; the dispatch is unchanged so theta plumbing stays
uniform.

### pack.py decoder mode

`gen_decoder_main` (in `aricode_pack.py`) replaces the
classifier-style argmax tail with a prefill + autoregressive
sample loop.  Tied-LM-head fallback handles the Llama / GPT-2
convention of aliasing `lm_head.weight` from
`model.embed_tokens.weight` (or `wte.weight`) when
`lm_head.weight` is absent in the state_dict.

### Regression coverage

- `examples/tiny_llama_min/` — synthetic 1-block Llama-shaped
  checkpoint; embedding → RMSNorm → GQA-KV (with RoPE) → SwiGLU
  FFN → RMSNorm → LM head.  Token-for-token greedy match against
  a PyTorch reference.
- Per-primitive: `examples/rmsnorm_min/`, `examples/swiglu_min/`,
  `examples/rope_min/`, `examples/gqa_min/`,
  `examples/mha_causal_min/`, `examples/rmsnorm_swiglu_decoder/`,
  `examples/tiny_decoder_2block/`.
- Unit tests in `tests/`: `test_attention_kv`, `test_attention_mh_kv`,
  `test_sampling`, `test_kv_reset`, `test_kv_overflow`,
  `test_rope_edges`, `test_sampling_edges`, `test_codegen_quirks`.

## What's left for a real Llama-2-7B deploy

### Checkpoint wrangling

The TinyLlama-1.1B and Llama-2-7B HF checkpoints distribute
weights as a **SentencePiece BPE tokenizer + safetensors shards**.
aricode-pack can already read `.safetensors`, but:

1. **Tokenizer pre-step** — the pack tool doesn't run a
   SentencePiece tokenizer.  The user feeds token IDs as the
   prompt (`prompt.bin`).  For a real demo this means a small
   Python script that loads `tokenizer.model` and emits the
   prompt-as-IDs — same pattern as the existing GPT-2 demo's
   `prepare.py`.  Llama-3 ships a 128k vocab; the existing
   `_embedding_token_bytes` already returns 2 bytes for
   vocab ≤ 65536 and 4 bytes above, so the Llama-3 vocab would
   simply route through 4-byte tokens.
2. **Multi-shard safetensors** — Llama-2-7B is split across
   ~10 shards.  The `--checkpoint` argument today takes one
   path; it would either need a sharded variant or a
   one-time merge step.
3. **arch.json** for full Llama-2-7B (32 blocks, 32 heads of
   d_head=128, n_kv_heads=32) is a straightforward expansion
   of the `tiny_llama_min` arch — drop in the v0.32 op vocab,
   bump dimensions.

### Quantization for the 7B

Llama-2-7B in f32 is ~28 GB of weights — too large for the
current `.text` payload reservation (1 GiB after the v0.32
codegen bump).  Two paths to fit:

- **int8 weights** (4× smaller): the existing
  `--quantize int8` path would shrink the model to ~7 GB.  This
  works for the FFN linears today (per-Linear scratch landed in
  v0.27), but the QKV / O linears in the MHA-KV / GQA-KV ops
  haven't been quantized yet.  Adding int8 to those layer kinds
  is a follow-up parallel to v0.27 — same arr_i8_matvec_f32
  kernel, same per-Linear scratch pattern, just plumbed through
  the GQA forward.
- **Int4 weights** (8× smaller, 7B → ~3.5 GB): no plumbing today.
  Would need a new builtin, new dequant path, and a new packer
  branch.  Out of scope for "just deploy Llama-2-7B"; revisit if
  there's actual demand for sub-8GB binaries.

### Optional kernel optimizations

These are speed wins, not correctness blockers:

- **`arr_f32_silu` builtin** — the SiLU step inside SwiGLU is
  scalar today.  A vectorized builtin (template:
  `arr_f32_exp` + an extra `vmulps`) would close ~20-30 % of
  FFN wall.  ~80 lines in `codegen_builtins.c`.
- **`arr_f32_rmsnorm` builtin** — the same pattern as
  `arr_f32_layernorm` minus the mean pass.  Would close a
  small constant cost at every block.
- **`arr_f32_rope_apply` builtin** — RoPE today rotates a
  pair of floats at a time using two `vmulps` and a
  `vaddsubps`-equivalent in the stdlib.  Lifting the entire
  rotation to a builtin (operating on the full d_head
  vector against the cached cos/sin row) would be ~50-100
  cycles per token saved.

None of these block deployment; the Llama-shaped tests pass
without them.

## Recommended order of attack (when picking this back up)

1. Build the SentencePiece + arch.json + multi-shard merge for
   TinyLlama-1.1B.  Verify the existing v0.32 pipeline produces
   token-by-token agreement with `transformers.AutoModelForCausalLM`
   on a few prompts.
2. Stretch to Llama-2-7B with int8 QKV/O — the only new code
   is plumbing the existing int8 path through the GQA-KV
   layer kind.
3. If wall-clock matters, ship the three optional builtins
   (silu, rmsnorm, rope) — each is a self-contained
   `codegen_builtins.c` patch + bench delta.

## Pointers

- Pack arch entries: `aricode_pack.py`, around the
  `multi_head_attention_gqa_kv` / `swiglu_ffn` / `rmsnorm`
  branches in `tensor_specs`, `tensor_names`, `weight_tensors`,
  `gen_act_decls`, `residual_slots`, `gen_forward`,
  `gen_scratch`, `_load_pytorch_state_dict`.
- Decoder loop: `gen_decoder_main` in `aricode_pack.py`.
- Stdlib: `aricode-ml/{rope_f32,sampling_f32,attention_kv_f32,
  attention_mh_kv_f32,attention_gqa_kv_f32}.ari`.
- Trophy demo (encoder-of-decoder family validated): `examples/
  gpt2_small/run_test.sh` — same pipeline, no GQA / RoPE, one
  HF download + prepare + pack + greedy match.
