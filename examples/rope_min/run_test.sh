#!/bin/bash
# rope_min — RoPE alloc/apply hand-validation.
# Locks in: rope_alloc_f32 table layout (interleaved cos/sin per pair)
# and rope_apply_f32 in-place rotation against hand-computed expected
# values for d_head=4, theta=10000, pos=1.

set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"

echo "[1/2] compiling..."
"$ARIC" test_rope.ari -o test_rope > /dev/null

echo "[2/2] running..."
./test_rope > /tmp/rope_min_out.txt
cat /tmp/rope_min_out.txt
if grep -q "ROPE_OK" /tmp/rope_min_out.txt; then
    echo "ROPE_MIN_OK — RoPE table + apply match hand-computed values"
else
    echo "FAIL: ROPE_OK marker not found"
    exit 1
fi
