# aricode-pack — model → static binary compiler

Take a PyTorch checkpoint, declare the architecture, get back a
`.ari` source + `.f32` weight blob.  `aric` then compiles that into
a static x86_64+AVX2 binary that loads the weights at startup and
serves inference.

## Why

The PyTorch / CUDA stack is ~1 GB of runtime and a Python interpreter,
which is fine in a research workstation and useless on:

- containers where every megabyte of cold-start matters,
- edge devices and IoT gateways with no GPU,
- regulated environments where the deployment pipeline must be
  auditable end to end,
- machines where you can't `pip install` (offline boxes, hardened
  appliances, secure enclaves).

aricode-pack turns "the model" into a deliverable a sysadmin can
treat like any other binary: copy two files (the `.f32` blob and the
compiled binary, ~200 KB total for a small MLP), run.  No runtime
linker, no Python, no virtualenv, no `requirements.txt` to drift.

It does **not** try to compete with CUDA on training, and it doesn't
try to outperform `ggml` on heavy LLM serving — those niches have
different shapes.  This is the "small model, simple deployment"
slot.

## Usage

```sh
# Train however you like; export the state_dict.
python train.py            # → mymodel.pt

# Declare the architecture.
cat > arch.json <<EOF
[
    ["linear", 784, 64],
    ["relu"],
    ["linear", 64, 10]
]
EOF

# Pack.
python aricode_pack.py \
    --checkpoint mymodel.pt \
    --arch arch.json \
    --input-format mnist \
    --input-images t10k-images-idx3-ubyte \
    --input-labels t10k-labels-idx1-ubyte \
    --n-test 10000 \
    --out mymodel
# → mymodel.ari + mymodel.f32

aric mymodel.ari -o mymodel
./mymodel        # runs inference, prints accuracy
```

## Supported layers (v0.1)

| kind       | extra args            | semantics                   | in-place? |
|------------|-----------------------|-----------------------------|:---------:|
| `linear`   | `in_features, out_features` | `y = W·x + b`         | no        |
| `relu`     | —                     | `max(0, x)`                 | yes       |
| `sigmoid`  | —                     | `1 / (1 + e^-x)`            | yes       |
| `tanh`     | —                     | `tanh(x)`                   | yes       |
| `softmax`  | —                     | normalised exp              | yes       |

Linear weights match PyTorch's `nn.Linear.weight` shape
`(out_features, in_features)` row-major exactly — no transpose at
pack time, no transpose at runtime.

CNN layers (`conv2d_3x3_p1`, `maxpool_2x2`) and attention land in
v0.2 once the layer template stabilises.  We're keeping the
vocabulary deliberately small until the design pressure for each
layer is concrete.

## State_dict key convention

By default aricode-pack expects `fc1.weight / fc1.bias / fc2.weight
/ ...` (PyTorch's natural naming when you write `self.fc1 =
nn.Linear(...)`).  Override with `--keys`:

```sh
# nn.Sequential default naming: 0.weight, 0.bias, 2.weight, ...
--keys "{idx_plus_1_times_2_minus_2}.{kind}"      # or just compute it
# Custom subclass:
--keys "model.layers.{idx}.{kind}"
```

`--keys` is a Python `str.format` template with `{idx}` (0-based),
`{idx_plus_1}` (1-based), and `{kind}` (`weight` | `bias`).

## Verifying parity

The `.f32` file aricode-pack emits is byte-identical to what a
manual `torch.tensor.numpy().tofile(f)` over the same state_dict in
declaration order would produce.  See `examples/mnist_infer/` for
both flows side by side — `cmp` between them comes back identical.

## Limits we'll fix

- **Weights are loaded from a separate file at runtime.** v0.2 will
  embed them in `.rodata` and emit a single self-contained binary,
  removing the second deploy artefact.
- **No quantisation.**  Everything is f32.  int8 packing is the
  natural follow-up — gives ~4× memory shrink and likely a real
  speedup on the small-model path.
- **Manual architecture spec.**  v0.2 will optionally introspect
  `nn.Module` graphs for the simple cases (Sequential, plain
  subclasses) and emit the spec automatically.
