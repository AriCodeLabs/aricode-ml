# aricode-ml

Neural network kernels and a model-to-binary packer for the
[aricode](https://github.com/Lynx-Boss/aricode) compiler.

Two halves:

- **Training-side primitives** in pure `.ari` — dense, conv2d (3×3 pad
  1, single- or multi-channel input), maxpool, single-head attention,
  layernorm, AdamW + SGD, plus the AVX2 builtins behind them.  Used by
  the in-tree demos that train MNIST end-to-end (98.65 % CNN with
  parallel workers, 23 s on a 4-core CPU).

- **`aricode-pack`** — a tool that takes a PyTorch `state_dict` plus a
  short JSON architecture spec and emits a single self-contained ELF
  binary with the model weights baked into `.text`.  No Python, no
  CUDA libs, no glibc, no runtime linker.  With `--quantize int8` a
  2-conv MNIST CNN ships in 218 KB total.

aricode is **not** a way to train large models faster than CUDA — it
won't be, by 1-2 orders of magnitude.  It's a deployment niche: the
slot where PyTorch's 1 GB stack is too much (edge devices, FaaS cold
starts, regulated environments, offline machines, archival).

## Quick start: GPU train → CPU deploy

```sh
# 1. Train any small model in PyTorch on your GPU
#    — or drop in a .safetensors from HuggingFace Hub.
cd examples/mnist_infer
python3 -m venv .venv
.venv/bin/pip install torch torchvision safetensors
.venv/bin/python train_cnn2_and_export.py
# → cnn2_mnist.pt              (state_dict, 27 s on RTX 3060, 98.77 % acc)
#    aricode-pack also reads .safetensors directly, no torch.load round-trip.

# 2. Declare the architecture (one-time, JSON).
#    Or get a starter from the checkpoint itself:
#      python ../../tools/aricode_pack.py --checkpoint cnn2_mnist.pt --infer-arch
cat arch_cnn2.json
# [
#     ["conv2d_3x3_p1", 1, 8],
#     ["relu"],
#     ["conv2d_3x3_p1", 8, 16],     ← multi-channel input (v0.5+)
#     ["relu"],
#     ["maxpool_2x2", 16],
#     ["flatten"],
#     ["linear", 3136, 64],
#     ["relu"],
#     ["linear", 64, 10]
# ]

# 3. Pack the trained model — int8 weights, single-binary deploy.
.venv/bin/python ../../tools/aricode_pack.py \
    --checkpoint cnn2_mnist.pt \
    --arch arch_cnn2.json \
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format stdin \
    --quantize int8 \
    --out cnn2_cli
# → cnn2_cli.ari + per-tensor staging .i8 / .f32 files

# 4. Compile to one self-contained binary; staging files no longer needed.
aric cnn2_cli.ari -o cnn2_cli
rm cnn2_cli_W*.i8 cnn2_cli_b*.f32 cnn2_cli_*c*.i8 2>/dev/null

# 5. Serve.  No Python, no .pt file, no external weights.
cat /tmp/img_0.bin | ./cnn2_cli
# → 7

ls -la cnn2_cli
# -rwxr-xr-x  218K  cnn2_cli                       (bare ELF, fully static)
file cnn2_cli
# ELF 64-bit LSB executable, x86-64, statically linked, no section header
```

End-to-end measurement on a Ryzen 7 5800X + RTX 3060 (2-conv CNN, MNIST):

| Stage                                    | Stack          | Wall            |
|------------------------------------------|----------------|-----------------|
| Train CNN2, 8 epochs                     | PyTorch + GPU  | 27 s            |
| Pack + compile                           | aricode-pack   | <1 s            |
| **Inference, 10 K test images (batch)**  | aricode binary | **1.45 s**      |
| **Single-shot CLI cold-start**           | aricode binary | **0.64 ms**     |
| **Test accuracy**                        | bit-exact match| **98.79 %**     |
| **Binary size (single-shot CLI)**        | aricode binary | **218 KB**      |

The 0.64 ms cold-start figure is ~10 000× faster than spinning up a
Python interpreter with PyTorch loaded — that's the gap this tool
exists to exploit.  With `--quantize int8` you also get a 4× shrink on
both the binary and the runtime weight RAM, at no measurable cost in
either accuracy or wall-clock (a single REX-byte typo in the i8
matvec kernel made an early version 75× slower; edge test #52 catches
that regression now).

## Quantisation

`--quantize int8` switches the deploy format to per-tensor symmetric
int8 (scale = max|x| / 127, zero point = 0) for Linear weights, and
keeps biases as f32 (a 64-element bias is 256 bytes either way; the
constant offset hurts more than the bytes save).  Conv weights also
go to int8 in the binary but get dequantised back to f32 once at
startup since the AVX2 conv kernel is f32-only — those tensors are
small (~5 KB total in the 2-conv demo) so the dequant pass is free.

Linear matmul on int8 uses the native `arr_i8_matvec_f32` builtin —
loads 8 sign-bytes via `vpmovsxbd`, promotes to f32 with `vcvtdq2ps`,
runs the standard `vfmadd231ps` chain.  Same throughput as the f32
matvec, no warm-up dequant pass, no f32 weight buffers in RAM.

|                          | f32      | int8     | Δ              |
|--------------------------|---------:|---------:|----------------|
| Binary (CLI single-shot) | 824 KB   | 218 KB   | **3.78×** smaller |
| Cold-start               | 0.64 ms  | 0.64 ms  | identical       |
| Batch 10 K               | 1.45 s   | 1.45 s   | identical       |
| Test accuracy            | 98.77 %  | 98.79 %  | within noise    |
| Weight RAM               | ~810 KB  | ~205 KB  | **4×** smaller    |

## What's in the box

```
aricode-ml/
├── tools/
│   ├── aricode_pack.py         model → .ari + .f32/.i8 (or single binary)
│   └── README.md               packer-specific docs
├── attention_f32.ari           single-head scaled dot-product attention
├── conv2d.ari / conv2d_f32.ari conv forward + im2col + maxpool
├── dense.ari                   dense_forward + relu + mse_loss
├── loss.ari                    xent_backward (softmax-CE fused)
├── optimizer.ari               adam_update_moments, adam_apply, SGD, clip
├── math_ops.ari                math_pow_int, scalar wrappers
├── examples/
│   ├── mnist/                  in-tree training demos (f32 + f64, SGD/AdamW,
│   │                           sequential and 4-thread parallel)
│   ├── mnist_infer/            train-on-GPU → deploy-on-CPU end to end
│   └── threading/              parallel matvec micro-benchmarks
└── tests/                      numerical sanity tests against analytical
                                solutions (test_attention is a hand-checked
                                2-token causal forward)
```

### Layer vocabulary supported by `aricode-pack`

| Layer kind       | Args                       | Notes                                 |
|------------------|----------------------------|---------------------------------------|
| `linear`         | `in_features, out_features`| `nn.Linear` weight `(out, in)`         |
| `conv2d_3x3_p1`  | `C_in, C_out`              | 28×28 spatial; C_in > 1 uses native    |
|                  |                            | multi-channel kernel.                  |
| `maxpool_2x2`    | `C`                        | 28×28 → 14×14                          |
| `attention`      | `seq, d_in, d_head, causal`| Single-head scaled dot-product.  Loads |
|                  |                            | `q_proj.weight` / `k_proj.weight` /    |
|                  |                            | `v_proj.weight` (HF naming) from the   |
|                  |                            | state_dict.                            |
| `multi_head_attention` | `seq, d_model, n_heads, causal` | Multi-head SDPA built on the |
|                  |                            | single-head kernel.  d_head =          |
|                  |                            | d_model/n_heads.  Loads q_proj /       |
|                  |                            | k_proj / v_proj / out_proj (all        |
|                  |                            | shaped d_model×d_model).               |
| `embedding`      | `vocab_size, d_model, seq` | Token-ID lookup table.  Must be the    |
|                  |                            | first arch entry; consumes a 1/2/4-    |
|                  |                            | byte-per-token raw stream from the     |
|                  |                            | input loader (auto-sized from          |
|                  |                            | vocab_size).  Loads HF                 |
|                  |                            | `embeddings.word_embeddings.weight` /  |
|                  |                            | `wte.weight` etc.  Real HF vocabs      |
|                  |                            | (BERT/distilbert: 30522) work.         |
| `positional_embedding` | `max_pos, d_model, seq` | Learned positional embedding —      |
|                  |                            | adds `W_pos[0:seq]` in-place to the    |
|                  |                            | current activation.  Loads HF          |
|                  |                            | `embeddings.position_embeddings.weight`|
|                  |                            | / `wpe.weight`.                        |
| `layernorm`      | `dim`                      | Affine LayerNorm over the last `dim`   |
|                  |                            | elements (γ, β learnable; loads as     |
|                  |                            | `LayerNorm.weight` / `LayerNorm.bias`).|
| `flatten`        | —                          | reshape, no code emitted               |
| `relu` / `sigmoid` / `tanh` / `softmax` / `gelu` | —      | in-place activations.  `gelu` uses the |
|                  |                            | tanh approximation matching            |
|                  |                            | `torch.nn.GELU(approximate='tanh')`.   |
| `save_residual` / `add_residual` | —          | LIFO-paired snapshot/accumulate for    |
|                  |                            | residual connections.  Save after one  |
|                  |                            | layer; add after a sub-block to fold   |
|                  |                            | the snapshot back in.                  |

Restrictions today: spatial size is 28×28 (the AVX2 conv builtins are
hardcoded for MNIST); CIFAR-style 32×32 RGB needs a generic conv
builtin which is on the roadmap.  Multi-channel conv with C_in > 1
goes through native AVX2 kernels (`arr_f32_conv2d_3x3_p1_multi` and
`arr_i8_conv2d_3x3_p1_multi`) — same kernel call shape whether
C_in = 1 or C_in = 64.

## When this is the right tool

✓ You trained a small classifier and want to deploy it where Python
or CUDA can't go: edge devices, FaaS cold starts, secure enclaves,
embedded x86 boards, hardened appliances.

✓ Your inference workload is sporadic and cold-start dominates.  A
PyTorch invocation needs 2-5 s to boot before it can do any work; an
aricode binary serves at sub-millisecond.

✓ You need an artefact a regulator, a hospital, or your security team
can audit end to end.  The whole inference path is human-readable
`.ari` plus open-source AVX2 kernels — no opaque CUDA, no
multi-gigabyte runtime.

✓ You want a model that still works in 20 years.  `.pt` from 2015
won't load today (Python deps drifted, CUDA versions gone); a static
ELF from 2026 will run on any 2046 x86_64 Linux that boots, full
stop.

## When it isn't

✗ You're training large models.  Stay on PyTorch + GPU.  aricode-ml
isn't competing for that workload.

✗ You need llama.cpp / vLLM-style throughput on big LLMs.  Those
ecosystems have years of quantisation and batched-attention work
this repo doesn't approximate.

✗ Your model has layers we don't pack yet (RNN, sinusoidal
positional encoding, arbitrary-spatial conv).  Through v0.21, every
layer of a learned-positional Pre-LN transformer encoder ships and
is regression-tested: token embedding (1/2/4-byte vocab), learned
positional embedding, single- and multi-head SDPA, LayerNorm, GELU,
residuals, batched Linear (FFN).  Stacking these into a multi-block
encoder is example-wiring only.

## Roadmap

Shipped:
- v0.8: native int8 conv (`arr_i8_conv2d_3x3_p1`) — single-channel
  conv weights stay int8 in RAM, no startup dequant pass.
- v0.9: HuggingFace `.safetensors` checkpoint reading.  Pack accepts
  `--checkpoint *.safetensors` directly; no torch.load round-trip
  needed.  See `tools/convert_to_safetensors.py` for migrating
  existing `.pt` artefacts.
- v0.10: `--infer-arch` walks a state_dict and emits a starter
  `arch.json` for sequential MLP / CNN architectures.  Removes the
  manual layer declaration step for the common cases (verified bit-
  exact against the hand-written archs for both demos).
- v0.11: native multi-channel f32 conv (`arr_f32_conv2d_3x3_p1_multi`)
  replaces the per-input-channel user-fn loop.  cnn2 demo: 1.45 s →
  1.05 s for 10K samples (~38 % wall, layer-level ~3×).
- v0.12: native multi-channel int8 conv
  (`arr_i8_conv2d_3x3_p1_multi`).  Multi-channel int8 weights stay
  int8 in RAM for `--input-format stdin` builds (single-shot CLI
  cold-start halves: 2 ms → 1 ms).  Batch loaders still dequant
  once at startup since amortising across N samples wins on
  steady-state throughput; the packer picks per `--input-format`.
- v0.13: transformer attention pack-side.  `["attention", seq, d_in,
  d_head, causal]` is now a first-class arch entry; the packer
  resolves `q_proj` / `k_proj` / `v_proj` from the state_dict
  (HuggingFace convention), inlines the attention library into the
  prologue, allocates the descriptor + scratch, and emits the
  forward.  `--infer-arch` detects the q/k/v triple and emits a
  starter attention entry (with placeholder seq).  End-to-end
  regression: `examples/attention_min/run_test.sh` packs a synthetic
  4-token attention layer and matches the PyTorch reference within
  1.3e-6 — pure f32 quantisation noise.  Also new in v0.13:
  `--input-format embedded` for baking a fixed input file into
  `.text` (test rigs, deterministic single-shot demos).
- v0.14: affine LayerNorm.  `["layernorm", dim]` arch entry; loads
  γ / β from `LayerNorm.weight` / `LayerNorm.bias` (HF convention).
  arr_f32_layernorm normalises in-place; a small scalar helper then
  applies the affine pass.  Regression: `examples/layernorm_min/`
  matches PyTorch's `torch.nn.LayerNorm + Linear` within 1.4e-6.
- v0.15: GELU activation.  `["gelu"]` arch entry; tanh approximation
  matching `torch.nn.GELU(approximate='tanh')` — the form used by
  HF / OpenAI / BERT lineages.  Regression: `examples/gelu_min/`
  packs a Linear → GELU → Linear FFN block and matches the PyTorch
  reference within 1.9e-6.
- v0.16: residual connections via paired `["save_residual"]` /
  `["add_residual"]` arch entries.  LIFO-paired so transformer
  blocks (with attention residual + FFN residual nested or
  sequential) compose naturally.  Stack-validates at pack time —
  unbalanced save/add or size-mismatched pairs error out clearly.
  Regression: `examples/residual_min/` packs a Linear → save →
  Linear → GELU → Linear → add residual-FFN block and matches the
  PyTorch reference within 2.9e-6.
- v0.17: batched Linear + full Pre-LN transformer encoder block.
  When a Linear's input has more elements than `in_f` (typical
  after an attention block — input is `[seq, d_in]` flat),
  emit_linear now emits a per-row loop using `_row_in` / `_row_out`
  scratch.  Activation sizing in `gen_act_decls` is batch-aware,
  the residual-slot validator agrees on shapes, and `n_out` always
  reads `sizes[-1]` so multi-row outputs print correctly under
  `--no-argmax`.  End-to-end regression:
  `examples/transformer_block_min/` packs the standard
      save → LN → Attention → add → save → LN → FFN(Linear-GELU-Linear) → add
  Pre-LN block (seq=4, d_model=16, d_ff=32) and matches PyTorch
  within 7.6e-6 across all 64 output elements.
- v0.18: multi-head attention.  `["multi_head_attention", seq,
  d_model, n_heads, causal]` arch entry; loads `q_proj` / `k_proj` /
  `v_proj` / `out_proj` (all `d_model × d_model`) from the state_dict
  and dispatches into the single-head kernel n_heads times via
  per-head weight slicing.  Concatenates head outputs and applies
  the output projection.  Regression: `examples/multi_head_min/`
  packs `[multi_head_attention, seq=4, d_model=16, n_heads=4]` and
  matches PyTorch's `torch.nn.functional.scaled_dot_product_attention`
  composed-with-out-projection within 4e-5 — sufficient for f32
  classifier-class accuracy.
- v0.19: embedding-table front-end.  `["embedding", vocab_size,
  d_model, seq]` arch entry; loads HuggingFace canonical names
  (`embeddings.word_embeddings.weight`, `wte.weight`, etc.) and
  emits a per-token copy_slice lookup.  Regression:
  `examples/embedding_min/` packs an embedding-only arch with token
  IDs `[3, 1, 4, 2]` and matches `nn.Embedding` within 1e-6.
- v0.20: multi-byte token loader for real-vocab transformers.  Auto-
  sizes the input byte stream by `vocab_size` — 1 byte for vocab
  ≤ 256, 2 bytes for ≤ 65 536 (BERT, distilbert, GPT-2 small,
  Llama-3 fits), 4 bytes for the long tail.  The embedding forward
  decodes little-endian.  Regression:
  `examples/embedding_2byte_min/` packs the distilbert vocab
  (30522), uses tokens spanning the full byte range, and matches
  PyTorch within 1e-6.
- v0.21: learned positional embedding.  `["positional_embedding",
  max_pos, d_model, seq]` arch entry.  Slices `W_pos[0:seq*d_model]`
  into a per-layer scratch and folds it back into the current
  activation via `arr_f32_add_scaled`.  Loads HF
  `embeddings.position_embeddings.weight` / `wpe.weight`.
  Regression: `examples/positional_min/` chains
  `[embedding(100,16,4), positional_embedding(32,16,4)]` and
  matches `tok_emb[ids] + pos_emb[:seq]` within 1e-6.
- v0.22: full Pre-LN transformer encoder integration.  No new
  layer kinds — just the compose-it-all-together test that proves
  every primitive shipped through v0.21 cooperates correctly under
  the standard encoder stack:
      embedding → +positional_embedding
        → save → LN → MHA → add
        → save → LN → FFN(Linear-GELU-Linear) → add
        → LN → classifier
  Regression: `examples/encoder_full_min/` packs this stack with
  a synthetic checkpoint and matches PyTorch within 3.3e-6 on the
  classifier logits.  Stacking N blocks against a real fine-tuned
  distilbert is now purely a matter of state_dict keys + tokenizer
  glue.

Pending:
- Sinusoidal positional encoding (no params, computed from position).
  Today's learned variant covers BERT/distilbert/GPT-2; the
  Attention-Is-All-You-Need / Llama-2 sinusoidal flavour needs a
  small per-position helper.  Lower priority since most practical
  HF encoder checkpoints use the learned form.
- Real distilbert pack: drop a HuggingFace fine-tuned checkpoint in
  and run sentence classification end-to-end.  All the building
  blocks ship through v0.22.  Outstanding work: tokenizer pre-step
  (the packer doesn't run a wordpiece tokenizer; users feed token
  IDs directly), and an arch.json template for the standard 6-block
  / 12-block distilbert / BERT layouts.
- generic-spatial conv2d (arbitrary H × W) — unlocks CIFAR-10 and
  any architecture with maxpool between conv layers.
- ARM / RISC-V back-end (today: x86_64 + AVX2 only).

## Running the in-tree training demos

These don't involve aricode-pack — they're aricode itself doing the
training, no PyTorch.  Useful as smoke tests for the AVX2 kernels
and as a reference for the kernel call shapes the packer emits.

```sh
cd examples/mnist
./get_data.sh                            # one-time MNIST fetch
aric mnist_cnn_par2_f32.ari -o cnn_par2  # parallel f32 CNN, AdamW
./cnn_par2                               # 23 s, 98.65 % acc, 4 cores
```

The MNIST demo set covers the shape-progression of the project:
single-thread MLP (`mnist.ari`, 33 s, 97.15 %), single-thread CNN
(`mnist_cnn.ari`, 97 s, 98.66 %), 4-thread CNN with f32 + AdamW
(`mnist_cnn_par2_f32.ari`, 23 s, 98.65 %).

## License

Copyright © 2026 Edwin F. Veliz Jaramillo.  All rights reserved.
