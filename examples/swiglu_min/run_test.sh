#!/bin/bash
# swiglu_min — SwiGLU FFN pack.py regression.
# Locks in: silu_mul_f32 helper, `["swiglu_ffn", d_model, d_ffn]`
# arch dispatch, three-projection (gate/up/down) tensor naming,
# and the swiglu state_dict resolver fallback keys.

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
    --embed --no-argmax --out swiglu_min > /dev/null

echo "[3/4] compiling..."
"$ARIC" swiglu_min.ari -o swiglu_min > /dev/null

echo "[4/4] comparing to PyTorch..."
./swiglu_min > /tmp/swiglu_min_out.txt
"$PY" -c "
import numpy as np
expected = np.fromfile('expected.f32', dtype=np.float32)
got = np.array([float(l) for l in open('/tmp/swiglu_min_out.txt') if l.strip()],
               dtype=np.float32)
if got.shape != expected.shape:
    raise SystemExit(f'shape mismatch: expected {expected.shape}, got {got.shape}')
diff = np.abs(got - expected)
print(f'  max abs diff:  {diff.max():.3e}')
print(f'  mean abs diff: {diff.mean():.3e}')
if diff.max() >= 1e-3:
    raise SystemExit(f'FAIL: max diff {diff.max():.3e} exceeds 1e-3 tolerance')
print('  SWIGLU_OK — packed SwiGLU FFN matches PyTorch reference')
"
