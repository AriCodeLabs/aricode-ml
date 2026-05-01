#!/bin/bash
# distilbert_2block_min — synthetic distilbert-style encoder regression
# (1 block, Post-LN, real HF distilbert key naming).
#
# Why it's worth the per-test directory:
#   - Exercises HF's nested key conventions
#     (transformer.layer.{i}.attention.{q,k,v,out}_lin,
#      transformer.layer.{i}.{sa,output}_layer_norm,
#      embeddings.{word,position}_embeddings,
#      embeddings.LayerNorm) end-to-end through the packer's resolvers.
#   - Validates Post-LN block ordering (LN → save → MHA → add → LN →
#     save → FFN → add → LN), which differs from encoder_full_min's
#     Pre-LN setup.
#   - Catches the GELU saturation bug fixed in v0.24: without the
#     clamp on `2*inner` before math_exp, this test fails by ~6 max abs.

set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PACK="$SCRIPT_DIR/../../tools/aricode_pack.py"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

echo "[1/4] generating synthetic distilbert checkpoint..."
"$PY" make_synth.py

echo "[2/4] packing through aricode-pack..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format embedded --input-file "$(pwd)/tokens.bin" \
    --embed --no-argmax --out db_min > /dev/null

echo "[3/4] compiling..."
"$ARIC" db_min.ari -o db_min > /dev/null

echo "[4/4] comparing to PyTorch..."
./db_min > /tmp/db_min_out.txt
"$PY" -c "
import numpy as np
expected = np.fromfile('expected.f32', dtype=np.float32)
got = np.array([float(l) for l in open('/tmp/db_min_out.txt') if l.strip()],
               dtype=np.float32)
if got.shape != expected.shape:
    raise SystemExit(f'shape mismatch: expected {expected.shape}, got {got.shape}')
diff = np.abs(got - expected)
print(f'  max abs diff:  {diff.max():.3e}')
print(f'  mean abs diff: {diff.mean():.3e}')
if diff.max() >= 1e-3:
    raise SystemExit(f'FAIL: max diff {diff.max():.3e} exceeds 1e-3 tolerance')
print('  DBERT_OK — synthetic distilbert encoder matches PyTorch')
"
