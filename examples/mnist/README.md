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
  (mean-gradient scaling: effective per-sample lr = lr / batch_size)
- **Training**: full 60 000 MNIST train / 10 000 test

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
epoch done, avg train NLL:
0.4139
test accuracy:
0.9223
epoch done, avg train NLL:
0.2142
test accuracy:
0.9453
epoch done, avg train NLL:
0.1598
test accuracy:
0.9573
epoch done, avg train NLL:
0.1284
test accuracy:
0.9639
epoch done, avg train NLL:
0.1077
test accuracy:
0.9688
```

~97 % test accuracy after 5 epochs, in about 17 s on a Zen 3
laptop.  Binary is ~21 KB.  Dialling `N_EPOCH` up to 10-15 with
a learning rate decay typically gets to ~98 %.

## Notes

- Paths are resolved relative to the binary's CWD, so run it from
  this directory.
- `byte_at(ptr, offset)` is unchecked — it's a raw `movzx` with no
  bounds check, so the caller owns validity. Meant for file-read
  buffers where the underlying `arr_new` length (in i64 slots) is
  8× smaller than the byte count being indexed.
