# aricode-pack — model → static binary compiler

Take a PyTorch checkpoint, declare the architecture in JSON, get back
a `.ari` source plus per-tensor weight files.  `aric` then compiles
that into a static x86_64+AVX2 binary with the weights baked into
`.text` — one self-contained ELF, no runtime linker, no Python, no
CUDA libs.

## Why

The PyTorch / CUDA stack is ~1 GB of runtime and a Python interpreter,
which is fine in a research workstation and useless on:

- containers where every megabyte of cold-start matters,
- edge devices and IoT gateways with no GPU,
- regulated environments where the deployment pipeline must be
  auditable end to end,
- machines where you can't `pip install` (offline boxes, hardened
  appliances, secure enclaves),
- archival deploys (a 2026 ELF still runs in 2046; a 2015 `.pt`
  doesn't load today).

aricode-pack turns "the model" into a deliverable a sysadmin can
treat like any other binary: copy one file (~200-800 KB depending on
quantisation), run.  No runtime linker, no Python, no virtualenv, no
`requirements.txt` to drift.

It does **not** try to compete with CUDA on training, and it doesn't
try to outperform `ggml` on heavy LLM serving — those niches have
different shapes.  This is the "small model, simple deployment" slot.

## Supported checkpoint formats

aricode-pack reads two formats from disk via `--checkpoint`:

| Extension                | Loader                         | Use case |
|--------------------------|--------------------------------|----------|
| `.pt`, `.pth`            | `torch.load(weights_only=True)` | Standard PyTorch state_dict — fastest path when you trained the model yourself. |
| `.safetensors`           | `safetensors.torch.load_file`  | HuggingFace Hub native format.  Skip the torch.load pickle round-trip; safer to load from untrusted sources. |

Both formats produce the same flat `{tensor_name → tensor}` dict that
the rest of the pack pipeline consumes.  The conversion helper
`tools/convert_to_safetensors.py` migrates a `.pt` to `.safetensors`
in one shot when you want to switch:

```sh
python convert_to_safetensors.py mymodel.pt mymodel.safetensors
```

## Usage

```sh
# Train however you like; export the state_dict.
python train.py            # → mymodel.pt

# Declare the architecture (explicit, no introspection magic).
cat > arch.json <<EOF
[
    ["linear", 784, 64],
    ["relu"],
    ["linear", 64, 10]
]
EOF

# Pack — with int8 quantisation for a 4× smaller binary.
python aricode_pack.py \
    --checkpoint mymodel.pt \
    --arch arch.json \
    --input-format mnist \
    --input-images t10k-images-idx3-ubyte \
    --input-labels t10k-labels-idx1-ubyte \
    --n-test 10000 \
    --quantize int8 \
    --out mymodel
# → mymodel.ari + per-tensor staging files

aric mymodel.ari -o mymodel
rm mymodel_*.i8 mymodel_*.f32 2>/dev/null   # staging no longer needed
./mymodel                                    # self-contained inference
```

### Input modes

`--input-format mnist`: read a raw idx-ubyte test set from disk and
report accuracy.  Used to evaluate a packed model end to end.

`--input-format stdin`: read `n_in` raw bytes from fd 0, run one
forward pass, print the argmax.  Use this for the deploy build —
plug into pipelines with `cat image.bin | ./model` or wrap in a
shell loop.

### Quantisation

Default is `--quantize none` (f32 weights, smaller-but-not-tiny
binaries).  `--quantize int8` switches to per-tensor symmetric
int8 (scale = max|x|/127):

| Layer kind   | Storage                   | Runtime                          |
|--------------|---------------------------|----------------------------------|
| `linear`     | int8 weights, f32 bias    | `arr_i8_matvec_f32` (native int8 matvec, on-the-fly dequant per 8-lane chunk) |
| `conv2d_3x3_p1` | int8 weights, f32 bias | dequant once at startup, then f32 conv |
| activations  | always f32                | unchanged                        |

Net effect on a 2-conv MNIST CNN: ~4× smaller binary, ~4× smaller
runtime weight RAM, no measurable wall-clock or accuracy regression.

## Supported layers

| kind            | extra args            | semantics                    | in-place? |
|-----------------|-----------------------|------------------------------|:---------:|
| `linear`        | `in_features, out_features` | `y = W·x + b`          | no        |
| `conv2d_3x3_p1` | `C_in, C_out`         | 3×3 conv pad 1, 28×28 spatial | no       |
| `maxpool_2x2`   | `C`                   | 28×28 → 14×14                | no        |
| `flatten`       | —                     | reshape, no code emitted     | yes (alias) |
| `relu`          | —                     | `max(0, x)`                  | yes       |
| `sigmoid`       | —                     | `1 / (1 + e^-x)`             | yes       |
| `tanh`          | —                     | `tanh(x)`                    | yes       |
| `softmax`       | —                     | normalised exp               | yes       |

Linear weights match PyTorch's `nn.Linear.weight` shape
`(out_features, in_features)` row-major exactly — no transpose at
pack time, no transpose at runtime.

Conv weights match `nn.Conv2d.weight` shape `(C_out, C_in, kH, kW)`,
flattened on the spatial dims to `(C_out, C_in × 9)` row-major, which
is what `arr_f32_conv2d_3x3_p1` expects.  C_in > 1 works via a
per-input-channel loop in the generated source — slower than a true
multi-channel kernel would be, but works today.

## State_dict key convention

By default aricode-pack expects `fc1.weight / fc1.bias / fc2.weight /
…` (PyTorch's natural naming when you write `self.fc1 = nn.Linear(…)`).
For convs it tries `conv.weight`, `conv1.weight`, `conv2.weight`, …
in order — covers single- and multi-conv subclasses without
configuration.  Override with `--keys`:

```sh
# nn.Sequential default naming: 0.weight, 0.bias, 2.weight, ...
--keys "{idx_plus_1_times_2_minus_2}.{kind}"
# Custom subclass:
--keys "model.layers.{idx}.{kind}"
```

`--keys` is a Python `str.format` template with `{idx}` (0-based),
`{idx_plus_1}` (1-based), and `{kind}` (`weight` | `bias`).

## Verifying parity

The `.f32` (or `.i8`) staging files are byte-identical to what a
manual `tensor.numpy().tofile(f)` (or quantise + tofile) over the
same state_dict would produce.  See `examples/mnist_infer/` for the
flow side by side; `cmp` between manual and packed files comes back
identical.

After deploy: run the packed binary against the original test set
and confirm accuracy matches what PyTorch reported on the same
weights.  All current demos are bit-exact.
