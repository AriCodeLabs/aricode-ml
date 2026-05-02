#!/bin/bash
# mha_causal_min — packed-binary multi-head SDPA with triangular
# (causal) mask.  Exercises the slot-19 mask code path in
# attention_forward_f32 against PyTorch's is_causal reference.
#
# This is the first regression test for the causal mask path; it has
# to pass before the KV-cache decoder loop can stand on it.
set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PACK="$SCRIPT_DIR/../../tools/aricode_pack.py"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

echo "[1/4] generating synthetic checkpoint + reference..."
"$PY" make_synth.py

echo "[2/4] packing (causal=1)..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --input-format embedded --input-file "$(pwd)/X_input.f32" \
    --embed --no-argmax --out mha_causal_min > /dev/null

echo "[3/4] compiling..."
"$ARIC" mha_causal_min.ari -o mha_causal_min > /dev/null

echo "[4/4] comparing to PyTorch (is_causal=True)..."
./mha_causal_min > /tmp/mha_causal_min_out.txt
"$PY" -c "
import numpy as np
expected = np.fromfile('expected.f32', dtype=np.float32)
got = np.array([float(l) for l in open('/tmp/mha_causal_min_out.txt') if l.strip()],
               dtype=np.float32)
if got.shape != expected.shape:
    raise SystemExit(f'shape mismatch: expected {expected.shape}, got {got.shape}')
diff = np.abs(got - expected)
print(f'  max abs diff:  {diff.max():.3e}')
print(f'  mean abs diff: {diff.mean():.3e}')
if diff.max() >= 1e-3:
    raise SystemExit(f'FAIL: max diff {diff.max():.3e} exceeds 1e-3 tolerance')
print('  MHA_CAUSAL_OK — packed causal multi-head attention matches PyTorch')
"
