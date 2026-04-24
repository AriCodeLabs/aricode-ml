# MNIST digit classifier

End-to-end MLP training in aricode. Reads raw IDX files, trains a
784 → 128 → 10 network with ReLU hidden / softmax output, reports
test accuracy per epoch.

Two variants ship side-by-side.  Same architecture, different
optimisers + preprocessing — pick whichever makes the point you
need.

| Variant            | Architecture                                     | Preprocessing                       | Optimiser                  | Epochs | Final acc |
|--------------------|--------------------------------------------------|-------------------------------------|----------------------------|-------:|-----------|
| `mnist.ari`        | MLP 784 → 128 → 10                               | `px / 255` ([0, 1])                 | SGD + lr decay 0.85        |    10  | 97.15 %   |
| `mnist_adam.ari`   | MLP 784 → 256 → 10                               | `(px / 255 − 0.1307) / 0.3081`      | AdamW + smoothing + cosine |    20  | 98.61 %   |
| `mnist_cnn.ari`    | Conv8 → Pool → FC 1568→64 → 10                   | `(px / 255 − 0.1307) / 0.3081`      | AdamW + smoothing + cosine |    10  | **98.66 %** |
| `mnist_lenet.ari`  | Conv8 → Conv16 → Pool → FC 3136→64 → 10          | `(px / 255 − 0.1307) / 0.3081`      | AdamW + smoothing + cosine |    10  | 98.60 %   |

## Architecture (all variants)

- **Input**: 28×28 grayscale
- **Hidden**: ReLU activation, He init (size varies — see table above)
- **Output**: 10-way softmax
- **Loss**: cross-entropy (fused with softmax in the backward pass,
  so the gradient reduces to `probs − onehot`)
- **Training**: full 60 000 train / 10 000 test, batch 64

## `mnist.ari` — SGD baseline

- **Optimiser**: mini-batch SGD, `lr = 0.1`, exponential decay 0.85 per
  epoch (mean-gradient scaling: effective per-sample lr = lr / batch).
- **Preprocessing**: raw pixel intensities scaled to `[0, 1]`.

## `mnist_adam.ari` — AdamW + full recipe

Layers the standard deep-learning hygiene on top of the baseline:

- **Input standardisation**: `(px / 255 − 0.1307) / 0.3081` — MNIST's
  per-channel mean/std, the torchvision convention.
- **Hidden width**: 256 units (vs 128 in the SGD baseline).
- **Label smoothing** (α = 0.05): target label gets `1 − α`, the nine
  off labels share `α / 9`.  Keeps the softmax from over-saturating
  on clean training data.
- **AdamW**: `lr_max = 1e-3`, β₁ = 0.9, β₂ = 0.999, ε = 1e-8,
  decoupled weight decay `wd = 1e-3` on weights only (not biases).
  Uses the AVX2 `arr_f64_adam_apply` fused kernel for the hot step.
- **Cosine lr schedule**: glides from `lr_max = 1e-3` down to
  `lr_min = 1e-5` over 20 epochs — lets the optimiser exploit a
  large step size early and settle into a flat minimum at the end.

## Usage

```sh
./get_data.sh                     # one-time: fetch the four IDX files

# SGD baseline (97.15 %, ~33 s):
aric mnist.ari -o mnist
./mnist

# AdamW MLP variant (98.61 %, ~137 s):
aric mnist_adam.ari -o mnist_adam
./mnist_adam

# CNN variant (98.66 %, ~124 s):
aric mnist_cnn.ari -o mnist_cnn
./mnist_cnn
```

`get_data.sh` pulls from an S3 mirror since the canonical
`yann.lecun.com` host is unreliable.

## AdamW trajectory (MLP)

| Epoch | train NLL | test acc |
|-------|-----------|----------|
|   1   | 0.2658    | 96.75 %  |
|   5   | 0.0910    | 98.32 %  |
|  10   | 0.0697    | 98.41 %  |
|  15   | 0.0637    | 98.53 %  |
|  20   | 0.0619    | **98.61 %** |

137 s wall-clock, ~50 KB binary.

## `mnist_cnn.ari` — one-conv CNN

Adds spatial structure on top of the Adam recipe:

- **Conv**: 1 → 8 channels, 3 × 3 kernel, pad 1, stride 1.
  Preserves the 28 × 28 spatial resolution.
- **ReLU** → **MaxPool 2 × 2, stride 2** → feature map 8 × 14 × 14.
- **FC1**: 1568 → 64, ReLU.
- **FC2**: 64 → 10, softmax.

Same AdamW + label-smoothing + cosine schedule as the MLP variant.
Convolution and pooling primitives are in `../../conv2d.ari` — a
pure-`.ari` reference using AVX2 `arr_f64_add_scaled` / `dot` / `sum`
for the hot arithmetic (im2col construction is still scalar).  A
compiler-level `arr_f64_conv2d` builtin is the next speed-up.

Trajectory (hyperparameters identical to `mnist_adam.ari`):

| Epoch | train NLL | test acc |
|-------|-----------|----------|
|   1   | 0.2998    | 96.73 %  |
|   5   | 0.1122    | 98.18 %  |
|  10   | 0.0925    | **98.66 %** |

124 s wall-clock, ~65 KB binary.  Beats the MLP at half the epoch
budget and with ~2× fewer parameters (101 K vs 203 K).  Still a
gentle upward slope at epoch 10 — more epochs or a wider conv (16
channels, or two stacked conv layers) would push toward 99 %.

## `mnist_lenet.ari` — stacked 2-conv CNN

First demo to exercise the full multi-channel convolution path:

- **Conv1**: 1 → 8 ch, 3 × 3, pad 1  → 8 × 28 × 28
- **ReLU**
- **Conv2**: 8 → 16 ch, 3 × 3, pad 1  → 16 × 28 × 28   (multi-channel builtin)
- **ReLU**
- **MaxPool 2 × 2** → 16 × 14 × 14  (= 3136 flat)
- **FC1**: 3136 → 64 + ReLU
- **FC2**: 64 → 10

Forward uses `arr_f64_conv2d_3x3_p1` for the first layer and
`arr_f64_conv2d_3x3_p1_multi` for the second.  Backward runs the
standard single-channel weight gradient on conv1, the multi-channel
weight gradient on conv2, and — critically — the transpose-conv
input gradient from conv2 back into conv1 so conv1 trains too.

| Epoch | train NLL | test acc |
|-------|-----------|----------|
|   1   | 0.2695    | 96.97 %  |
|   5   | 0.1100    | 98.27 %  |
|  10   | 0.0935    | **98.60 %** |

440 s wall-clock, ~202 K parameters.  Comparable accuracy to
`mnist_cnn.ari` at the same 10-epoch budget; the point of this
demo is to exercise the full multi-channel forward / backward stack
end-to-end, not to set a MNIST record.  Data augmentation or more
epochs with a retuned lr schedule are the natural next steps toward
99 %+.

## Notes

- Paths are resolved relative to the binary's CWD, so run it from
  this directory.
- `byte_at(ptr, offset)` is unchecked — it's a raw `movzx` with no
  bounds check, so the caller owns validity. Meant for file-read
  buffers where the underlying `arr_new` length (in i64 slots) is
  8× smaller than the byte count being indexed.
