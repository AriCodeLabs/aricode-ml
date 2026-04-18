#!/bin/bash
# Download MNIST from an unpacked IDX mirror (the canonical lecun.com
# host is flaky; this mirror serves the same bytes over plain HTTP).
set -euo pipefail
cd "$(dirname "$0")"

BASE="https://ossci-datasets.s3.amazonaws.com/mnist"
FILES=(
    "train-images-idx3-ubyte"
    "train-labels-idx1-ubyte"
    "t10k-images-idx3-ubyte"
    "t10k-labels-idx1-ubyte"
)

for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "  [skip] $f already present"
        continue
    fi
    echo "  [fetch] $f.gz"
    curl -sS -o "$f.gz" "$BASE/$f.gz"
    gunzip "$f.gz"
done

echo "Done.  Files in $(pwd):"
ls -la *ubyte
