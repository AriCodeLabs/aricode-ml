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
| `conv2d_3x3_p1`  | `C_in, C_out`              | 28×28 spatial.  C_in > 1 supported via |
|                  |                            | a per-input-channel loop.              |
| `maxpool_2x2`    | `C`                        | 28×28 → 14×14                          |
| `flatten`        | —                          | reshape, no code emitted               |
| `relu` / `sigmoid` / `tanh` / `softmax` | —              | in-place activations                   |

Restrictions today: spatial size is 28×28 (the AVX2 conv builtin is
hardcoded for MNIST); CIFAR-style 32×32 RGB needs a generic conv
builtin which is on the roadmap.  Multi-channel conv with C_in > 1 is
implemented as a user-fn loop over input channels — works, runs ~3-4×
slower than a true multi-channel kernel would.

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

✗ Your model has layers we don't pack yet (RNN, transformer block,
arbitrary-spatial conv).  These are roadmap; for now, simple
feed-forward nets and the MNIST CNN architecture family are the
sweet spot.

## Roadmap

Shipped:
- v0.8: native int8 conv (`arr_i8_conv2d_3x3_p1`) — single-channel
  conv weights stay int8 in RAM, no startup dequant pass.
- v0.9: HuggingFace `.safetensors` checkpoint reading (this release).
  Pack accepts `--checkpoint *.safetensors` directly; no torch.load
  round-trip needed.  See `tools/convert_to_safetensors.py` for
  migrating existing `.pt` artefacts.

Pending:
- multi-channel int8 conv builtin (closes the last dequant pass for
  deep CNNs; today multi-channel conv with `--quantize int8` falls
  back to f32 dequant-at-startup).
- multi-channel f32 conv builtin (`arr_f32_conv2d_3x3_p1_multi`) —
  port the existing f64 implementation, skip the user-fn loop.
- stand-alone transformer block packer (attention layer is already
  shipped in `attention_f32.ari`; needs the pack-side wiring).
- HF auto-arch detection — derive `arch.json` from common HF model
  shapes (sentence-transformers, distilbert) without manual
  declaration.
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
