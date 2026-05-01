#!/bin/bash
# transformer_block_min — full Pre-LN transformer encoder block end-to-end.
#
# Exercises every transformer-related arch entry shipped through v0.16
# in a single binary: save_residual, layernorm, attention, add_residual,
# linear, gelu.  This is the standard modern (Pre-LN) encoder block:
#
#   y = x + Attention(LN(x))
#   z = y + FFN(LN(y))     where FFN = Linear → GELU → Linear
#
# Single-head SDPA (multi-head wrapping is the next roadmap piece).

set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PACK="$SCRIPT_DIR/../../tools/aricode_pack.py"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

echo "[1/4] generating synthetic transformer-block checkpoint + reference..."
"$PY" make_synth.py

echo "[2/4] packing through aricode-pack..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format embedded --input-file "$(pwd)/X_input.f32" \
    --embed --no-argmax --out tb_min > /dev/null

echo "[3/4] compiling..."
"$ARIC" tb_min.ari -o tb_min > /dev/null

echo "[4/4] comparing to PyTorch..."
./tb_min > /tmp/tb_min_out.txt
"$PY" -c "
import numpy as np
expected = np.fromfile('expected.f32', dtype=np.float32)
got = np.array([float(l) for l in open('/tmp/tb_min_out.txt') if l.strip()],
               dtype=np.float32)
if got.shape != expected.shape:
    raise SystemExit(f'shape mismatch: expected {expected.shape}, got {got.shape}')
diff = np.abs(got - expected)
print(f'  max abs diff:  {diff.max():.3e}')
print(f'  mean abs diff: {diff.mean():.3e}')
if diff.max() >= 1e-3:
    raise SystemExit(f'FAIL: max diff {diff.max():.3e} exceeds 1e-3 tolerance')
print('  TBLOCK_OK — packed transformer block matches PyTorch reference')
"
