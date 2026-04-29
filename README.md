# aricode-ml

Neural network kernels and a model-to-binary packer for the
[aricode](https://github.com/Lynx-Boss/aricode) compiler.

Two halves:

- **Training-side primitives** in pure `.ari` — dense, conv2d (3×3 pad
  1), maxpool, single-head attention, AdamW + SGD, plus the AVX2
  builtins behind them.  Used by the in-tree examples that train MNIST
  end-to-end (98.65 % CNN with parallel workers, 23 s on a 4-core CPU).

- **`aricode-pack`** — a tool that takes a PyTorch `state_dict` plus a
  short JSON architecture spec and emits a single self-contained ELF
  binary with the model weights baked into `.text`.  No Python, no
  CUDA libs, no glibc, no runtime linker; ~200–400 KB total for small
  models.

aricode is **not** a way to train large models faster than CUDA — it
won't be, by 1-2 orders of magnitude.  It's a deployment niche: the
slot where PyTorch's 1 GB stack is too much (edge devices, FaaS cold
starts, regulated environments, offline machines, archival).

## Quick start: GPU train → CPU deploy

```sh
# 1. Train any small model in PyTorch on your GPU.
cd examples/mnist_infer
python3 -m venv .venv
.venv/bin/pip install torch torchvision
.venv/bin/python train_cnn_and_export.py
# → cnn_mnist.pt              (state_dict, 33 s on RTX 3060, 98.32 % acc)

# 2. Declare the architecture (one-time, JSON).
cat arch_cnn.json
# [
#     ["conv2d_3x3_p1", 1, 8],
#     ["relu"],
#     ["maxpool_2x2", 8],
#     ["flatten"],
#     ["linear", 1568, 64],
#     ["relu"],
#     ["linear", 64, 10]
# ]

# 3. Pack the trained model into an aricode source tree.
.venv/bin/python ../../tools/aricode_pack.py \
    --checkpoint cnn_mnist.pt \
    --arch arch_cnn.json \
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format stdin --embed \
    --out cnn_cli
# → cnn_cli.ari  + 6 staging .f32 files

# 4. Compile to one self-contained binary; staging files no longer needed.
aric cnn_cli.ari -o cnn_cli
rm cnn_cli_W*.f32 cnn_cli_b*.f32

# 5. Serve.  No Python, no .pt file, no external weights.
cat /tmp/img_0.bin | ./cnn_cli
# → 7

ls -la cnn_cli
# -rwxr-xr-x  414K  cnn_cli                       (bare ELF, fully static)
file cnn_cli
# ELF 64-bit LSB executable, x86-64, statically linked, no section header
```

End-to-end measurement on a Ryzen 7 5800X + RTX 3060:

| Stage                                    | Stack          | Wall            |
|------------------------------------------|----------------|-----------------|
| Train CNN, 10 epochs, MNIST              | PyTorch + GPU  | 33 s            |
| Pack + compile                           | aricode-pack   | <1 s            |
| **Inference, 10 K test images (batch)**  | aricode binary | **393 ms**      |
| **Single-shot CLI cold-start**           | aricode binary | **0.41 ms**     |
| **Test accuracy**                        | bit-exact match| **98.32 %**     |

The cold-start figure is ~10 000× faster than spinning up a Python
interpreter with PyTorch loaded — that's the gap this tool exists to
exploit.

## What's in the box

```
aricode-ml/
├── tools/
│   ├── aricode_pack.py         model → .ari + .f32 (or single binary)
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

| Layer kind       | Args                       | Notes                              |
|------------------|----------------------------|------------------------------------|
| `linear`         | `in_features, out_features`| `nn.Linear` weight `(out, in)`     |
| `conv2d_3x3_p1`  | `C_in, C_out`              | C_in = 1, fixed 28×28 spatial      |
| `maxpool_2x2`    | `C`                        | 28×28 → 14×14                      |
| `flatten`        | —                          | reshape, no code emitted           |
| `relu` / `sigmoid` / `tanh` / `softmax` | —              | in-place activations               |

Restrictions in v0.4: conv is single-channel input only (the existing
AVX2 builtin `arr_f32_conv2d_3x3_p1` is fixed at C_in=1, 28×28).
Multi-channel conv and arbitrary spatial sizes land when there's a
demo that needs them.

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
multi-channel conv).  These are roadmap; for now, simple feed-forward
nets and the MNIST CNN architecture are the sweet spot.

## Roadmap

- v0.5: int8 quantisation (4× memory shrink, real speedup on the
  small-model path)
- v0.5: multi-channel conv2d (`arr_f32_conv2d_3x3_p1_multi`)
- v0.6: HuggingFace bridge — read `.safetensors`, auto-derive arch
  for known shapes (sentence-transformers, distilbert)
- v0.6: stand-alone transformer block packer (attention layer is
  already shipped in `.ari`; needs the pack-side wiring)
- v0.7: ARM / RISC-V back-end (today: x86_64 + AVX2 only)

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
