#!/bin/bash
# embedding_2byte_min — packed-binary embedding lookup with a
# real-HF-class vocab (distilbert: 30522).  Forces the 2-byte token
# loader path and checks correctness across the full u16 range.

set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PACK="$SCRIPT_DIR/../../tools/aricode_pack.py"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

echo "[1/4] generating distilbert-class embedding checkpoint..."
"$PY" make_synth.py

echo "[2/4] packing..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --input-format embedded --input-file "$(pwd)/tokens.bin" \
    --embed --no-argmax --out emb2_min > /dev/null

echo "[3/4] compiling..."
"$ARIC" emb2_min.ari -o emb2_min > /dev/null

echo "[4/4] comparing to PyTorch..."
./emb2_min > /tmp/emb2_min_out.txt
"$PY" -c "
import numpy as np
expected = np.fromfile('expected.f32', dtype=np.float32)
got = np.array([float(l) for l in open('/tmp/emb2_min_out.txt') if l.strip()],
               dtype=np.float32)
if got.shape != expected.shape:
    raise SystemExit(f'shape mismatch: expected {expected.shape}, got {got.shape}')
diff = np.abs(got - expected)
print(f'  max abs diff:  {diff.max():.3e}')
print(f'  mean abs diff: {diff.mean():.3e}')
if diff.max() >= 1e-3:
    raise SystemExit(f'FAIL: max diff {diff.max():.3e} exceeds 1e-3 tolerance')
print('  EMB2_OK — packed embedding (vocab=30522, 2-byte loader) matches PyTorch')
"
