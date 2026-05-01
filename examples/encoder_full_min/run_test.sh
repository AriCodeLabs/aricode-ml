#!/bin/bash
# encoder_full_min — full-stack Pre-LN transformer encoder demo.
#
# Exercises every layer the packer supports (v0.13–v0.21):
#   embedding + positional_embedding + (LN + MHA + residual +
#   LN + FFN-with-GELU + residual) + LN + classifier head
#
# This is the "do all the pieces compose end-to-end?" integration
# check.  If it passes, packing a fine-tuned distilbert-class model
# is example/test wiring only.

set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PACK="$SCRIPT_DIR/../../tools/aricode_pack.py"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

echo "[1/4] generating synthetic encoder checkpoint + reference..."
"$PY" make_synth.py

echo "[2/4] packing through aricode-pack..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format embedded --input-file "$(pwd)/tokens.bin" \
    --embed --no-argmax --out enc_min > /dev/null

echo "[3/4] compiling..."
"$ARIC" enc_min.ari -o enc_min > /dev/null

echo "[4/4] comparing to PyTorch..."
./enc_min > /tmp/enc_min_out.txt
"$PY" -c "
import numpy as np
expected = np.fromfile('expected.f32', dtype=np.float32)
got = np.array([float(l) for l in open('/tmp/enc_min_out.txt') if l.strip()],
               dtype=np.float32)
if got.shape != expected.shape:
    raise SystemExit(f'shape mismatch: expected {expected.shape}, got {got.shape}')
diff = np.abs(got - expected)
print(f'  max abs diff:  {diff.max():.3e}')
print(f'  mean abs diff: {diff.mean():.3e}')
# Multi-block transformer accumulates float noise — slightly looser
# tolerance than the per-layer regressions.  1e-3 is well below
# classifier-class accuracy in any practical setting.
if diff.max() >= 1e-3:
    raise SystemExit(f'FAIL: max diff {diff.max():.3e} exceeds 1e-3 tolerance')
print('  ENC_OK — full Pre-LN transformer encoder matches PyTorch')
"
