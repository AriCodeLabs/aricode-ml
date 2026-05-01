#!/bin/bash
# gelu_min — packed-binary GELU regression.
#
# Exercises the gelu_f32 user-fn helper (tanh approximation) by packing
# a Linear → GELU → Linear FFN block and comparing to torch.nn.GELU.

set -euo pipefail

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PACK="$SCRIPT_DIR/../../tools/aricode_pack.py"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

echo "[1/4] generating synthetic checkpoint + reference..."
"$PY" make_synth.py

echo "[2/4] packing through aricode-pack..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format embedded --input-file "$(pwd)/X_input.f32" \
    --embed --no-argmax --out gelu_min > /dev/null

echo "[3/4] compiling .ari → static ELF..."
"$ARIC" gelu_min.ari -o gelu_min > /dev/null

echo "[4/4] running + comparing to PyTorch reference..."
./gelu_min > /tmp/gelu_min_out.txt
"$PY" -c "
import numpy as np
expected = np.fromfile('expected.f32', dtype=np.float32)
got = np.array([float(l) for l in open('/tmp/gelu_min_out.txt') if l.strip()],
               dtype=np.float32)
if got.shape != expected.shape:
    raise SystemExit(f'shape mismatch: expected {expected.shape}, got {got.shape}')
diff = np.abs(got - expected)
print(f'  max abs diff:  {diff.max():.3e}')
print(f'  mean abs diff: {diff.mean():.3e}')
if diff.max() >= 1e-3:
    raise SystemExit(f'FAIL: max diff {diff.max():.3e} exceeds 1e-3 tolerance')
print('  GELU_OK — packed binary matches torch.nn.GELU(tanh) reference')
"
