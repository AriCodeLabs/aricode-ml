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
- **Optimizer**: SGD, batch size 1, `lr = 0.05`
- **Sample sizes**: 5 000 train / 1 000 test (from the full 60k/10k set)

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
0.5810
test accuracy:
0.8810
...
epoch done, avg train NLL:
0.1486
test accuracy:
0.9030
```

~90% test accuracy after 5 epochs, in about 4.5 s on a Zen 3
laptop. Binary is ~21 KB.

## Notes

- Paths are resolved relative to the binary's CWD, so run it from
  this directory.
- This demo uses the scalar `math_exp` + a hand-rolled `softmax_n`
  instead of the AVX2 `arr_f64_softmax` builtin, because the
  builtin requires `N % 4 == 0` and the label space here is 10.
  A builtin with a scalar tail would let us drop the helper.
- `byte_at(ptr, offset)` is unchecked — it's a raw `movzx` with no
  bounds check, so the caller owns validity. Meant for file-read
  buffers where the underlying `arr_new` length (in i64 slots) is
  8× smaller than the byte count being indexed.
