# mnist_infer — train on GPU (PyTorch), serve on CPU (aricode)

This example demonstrates the deployment niche aricode actually beats
the PyTorch stack at: a **fully static, ~5 KB binary with no runtime
dependencies** that loads model weights from a raw f32 blob and serves
inference through AVX2 inner loops.

| Stage           | Stack          | Wall clock       | Footprint     |
|-----------------|----------------|------------------|---------------|
| Train 10 epochs | PyTorch on GPU | 33 s             | ~1 GB libs    |
| Inference 10 K  | aricode binary | **121 ms**       | **204 KB total** |
| Test accuracy   | bit-exact match | 97.54 %         | —             |

(Numbers from a Ryzen 7 5800X CPU + RTX 3060.  Your hardware will
vary, but the *ratio* — train heavy on GPU, deploy tiny on CPU —
holds.)

## Architecture

A 784 → 64 → 10 ReLU MLP.  PyTorch's `nn.Linear.weight` shape
`(out_features, in_features)` matches aricode's `arr_f32_matvec`
row-major `(m, n)` convention exactly, so weights transfer with
no transpose.

## Reproduce

```sh
cd aricode-stdlib/aricode-ml/examples/mnist_infer

# 1. Train + export weights (uses GPU if available)
python3 -m venv .venv && .venv/bin/pip install torch torchvision
.venv/bin/python train_and_export.py
# → mlp_784_64_10.f32 (~200 KB) + mnist_data/MNIST/raw/*

# 2. Symlink the test data the inference binary expects
ln -sf mnist_data/MNIST/raw/t10k-images-idx3-ubyte .
ln -sf mnist_data/MNIST/raw/t10k-labels-idx1-ubyte .

# 3. Build + run aricode inference
aric mnist_infer.ari -o mnist_infer
./mnist_infer
# → "Test accuracy: 0.9754"
```

## Blob layout

The exporter writes four tensors back-to-back as little-endian f32:

```
W1  (64 × 784)  ← 50176 floats
b1  (64,)       ← 64
W2  (10 × 64)   ← 640
b2  (10,)       ← 10
                ─────
                  50890 floats  =  203 560 bytes
```

aricode reads them with four `file_read` calls in the same order.
No header, no version byte, no struct — the schema lives in both
files' source code, full stop.  When the model architecture changes,
both ends move together by hand.

## Why this is interesting

The argument for aricode isn't "faster than CUDA" — it's not, by
1-2 orders of magnitude on training.  It's **deployment**:

- **No Python.**  No interpreter to ship, no `pip install` at the
  destination, no virtualenv.
- **No CUDA / cuDNN.**  Inference runs everywhere x86_64 + AVX2 does.
- **No dynamic linker.**  The binary is `statically linked` per
  `file(1)` — moves between machines as one file.
- **Tiny.**  204 KB for the full deployment beats the *first round
  trip* of pulling a docker image with libtorch.

Niches: edge devices, tight container images, FaaS cold starts,
secure enclaves, anywhere a 1 GB Python+CUDA stack is overkill or
unavailable.
