# MNIST digit classifier

End-to-end MLP training in aricode. Reads raw IDX files, trains a
784 → 128 → 10 network with ReLU hidden / softmax output, reports
test accuracy per epoch.

Two variants ship side-by-side.  Same architecture, different
optimisers + preprocessing — pick whichever makes the point you
need.

| Variant            | Hidden | Preprocessing                       | Optimiser                  | Epochs | Final acc |
|--------------------|-------:|-------------------------------------|----------------------------|-------:|-----------|
| `mnist.ari`        | 128    | `px / 255` ([0, 1])                 | SGD + lr decay 0.85        |    10  | 97.15 %   |
| `mnist_adam.ari`   | 256    | `(px / 255 − 0.1307) / 0.3081`      | AdamW + smoothing + cosine |    20  | **98.61 %** |

## Architecture (both variants)

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

# AdamW variant (98.14 %, ~36 s):
aric mnist_adam.ari -o mnist_adam
./mnist_adam
```

`get_data.sh` pulls from an S3 mirror since the canonical
`yann.lecun.com` host is unreliable.

## AdamW trajectory

| Epoch | train NLL | test acc |
|-------|-----------|----------|
|   1   | 0.2658    | 96.75 %  |
|   5   | 0.0910    | 98.32 %  |
|  10   | 0.0697    | 98.41 %  |
|  15   | 0.0637    | 98.53 %  |
|  20   | 0.0619    | **98.61 %** |

137 s wall-clock, ~50 KB binary.  Still a slight climb at epoch 20,
so more epochs + light augmentation (shifts, rotations) could keep
pushing.  For a bigger jump the next step is CNN builtins
(`arr_f64_conv2d` + `arr_f64_max_pool`), neither of which ship yet —
those unlock ~99 % on MNIST.

## Notes

- Paths are resolved relative to the binary's CWD, so run it from
  this directory.
- `byte_at(ptr, offset)` is unchecked — it's a raw `movzx` with no
  bounds check, so the caller owns validity. Meant for file-read
  buffers where the underlying `arr_new` length (in i64 slots) is
  8× smaller than the byte count being indexed.
