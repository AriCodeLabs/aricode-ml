# MNIST digit classifier

End-to-end MLP training in aricode. Reads raw IDX files, trains a
784 → 128 → 10 network with ReLU hidden / softmax output, reports
test accuracy per epoch.

Two variants ship side-by-side.  Same architecture, different
optimisers + preprocessing — pick whichever makes the point you
need.

| Variant            | Preprocessing                       | Optimiser           | Final acc |
|--------------------|-------------------------------------|---------------------|-----------|
| `mnist.ari`        | `px / 255` ([0, 1])                 | SGD + lr decay 0.85 | 97.15 %   |
| `mnist_adam.ari`   | `(px / 255 − 0.1307) / 0.3081`      | AdamW + smoothing   | **98.14 %** |

## Architecture (both variants)

- **Input**: 28×28 grayscale
- **Hidden**: 128 units, ReLU activation, He init
- **Output**: 10-way softmax
- **Loss**: cross-entropy (fused with softmax in the backward pass,
  so the gradient reduces to `probs − onehot`)
- **Training**: full 60 000 train / 10 000 test, 10 epochs, batch 64

## `mnist.ari` — SGD baseline

- **Optimiser**: mini-batch SGD, `lr = 0.1`, exponential decay 0.85 per
  epoch (mean-gradient scaling: effective per-sample lr = lr / batch).
- **Preprocessing**: raw pixel intensities scaled to `[0, 1]`.

## `mnist_adam.ari` — AdamW + full recipe

Layers three standard tricks on top of the baseline:

- **Input standardisation**: `(px / 255 − 0.1307) / 0.3081` — MNIST's
  per-channel mean/std, the torchvision convention.
- **Label smoothing** (α = 0.05): target label gets `1 − α`, the nine
  off labels share `α / 9`.  Keeps the softmax from over-saturating
  on clean training data.
- **AdamW**: `lr = 1e-3`, β₁ = 0.9, β₂ = 0.999, ε = 1e-8,
  decoupled weight decay `wd = 1e-3` on weights only (not biases).
  Uses the AVX2 `arr_f64_adam_apply` fused kernel for the hot step.

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
|   1   | 0.2926    | 95.98 %  |
|   3   | 0.1335    | 97.50 %  |
|   5   | 0.1070    | 97.99 %  |
|   7   | 0.0944    | 98.06 %  |
|  10   | 0.0846    | **98.14 %** |

Test accuracy is still climbing at epoch 10 — adding epochs or a lr
schedule would likely push past 98.3 %.  For higher ceilings the
remaining knobs are a larger hidden layer or CNN builtins (conv2d
+ max-pool), neither of which ship yet.

## Notes

- Paths are resolved relative to the binary's CWD, so run it from
  this directory.
- `byte_at(ptr, offset)` is unchecked — it's a raw `movzx` with no
  bounds check, so the caller owns validity. Meant for file-read
  buffers where the underlying `arr_new` length (in i64 slots) is
  8× smaller than the byte count being indexed.
