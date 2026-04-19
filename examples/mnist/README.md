# MNIST digit classifier

End-to-end MLP training in aricode. Reads raw IDX files, trains a
784 → 128 → 10 network with ReLU hidden / softmax output, reports
test accuracy per epoch.

## Architecture

- **Input**: 28×28 grayscale, normalised to `[0, 1]`
- **Hidden**: 128 units, ReLU activation, He init
- **Output**: 10-way softmax
- **Loss**: cross-entropy (fused with softmax in the backward pass,
  so the gradient reduces to `probs - onehot`)
- **Optimizer**: mini-batch SGD, batch size 64, `lr = 0.1`
  (mean-gradient scaling: effective per-sample lr = lr / batch_size),
  exponential decay of 0.85 per epoch
- **Training**: full 60 000 MNIST train / 10 000 test, 10 epochs

## Usage

```sh
./get_data.sh                     # one-time: fetch the four IDX files
aric mnist.ari -o mnist
./mnist                           # run from this directory
```

`get_data.sh` pulls from an S3 mirror since the canonical
`yann.lecun.com` host is unreliable.

## Expected output

```
Loaded MNIST subset.
epoch done, avg train NLL:    (lr = 0.100)
0.4139
test accuracy:
0.9223
...
epoch done, avg train NLL:    (lr = 0.020, after 9× decay)
0.0885
test accuracy:
0.9715
```

~97.15 % test accuracy after 10 epochs, in about 33 s on a Zen 3
laptop.  Binary is ~21 KB.  Pushing past 98 % typically needs a
larger hidden layer (256 or 512) or Adam with a warmup schedule —
both wire in cleanly on top of this template.

## Notes

- Paths are resolved relative to the binary's CWD, so run it from
  this directory.
- `byte_at(ptr, offset)` is unchecked — it's a raw `movzx` with no
  bounds check, so the caller owns validity. Meant for file-read
  buffers where the underlying `arr_new` length (in i64 slots) is
  8× smaller than the byte count being indexed.
