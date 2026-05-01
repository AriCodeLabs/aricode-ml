#!/bin/bash
# attention_min — end-to-end transformer pack-side regression test.
#
# 1. Build a synthetic 4-token, d_in=8, d_head=16 attention layer in
#    PyTorch with deterministic random weights.
# 2. Pack it through aricode-pack with --input-format embedded.
# 3. Compile to a static ELF, run it.
# 4. Compare every output element to the PyTorch reference; pass if
#    max abs diff < 1e-3 (well above f32 noise floor, well below any
#    real algebraic error).
#
# This locks in: q_proj/k_proj/v_proj key resolution, the multi-tensor
# weight pipeline (3 W + 3 b per attention layer), the descriptor
# allocator emission, the embedded ATTENTION_LIB inlining, and the
# attention_forward_f32 invocation.

set -euo pipefail

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PACK="$SCRIPT_DIR/../../tools/aricode_pack.py"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

echo "[1/4] generating synthetic checkpoint + reference..."
"$PY" make_synth.py

echo "[2/4] packing through aricode-pack (--input-format embedded)..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --input-format embedded --input-file "$(pwd)/X_input.f32" \
    --embed --no-argmax --out attn_min > /dev/null

echo "[3/4] compiling .ari → static ELF..."
"$ARIC" attn_min.ari -o attn_min > /dev/null

echo "[4/4] running + comparing to PyTorch reference..."
./attn_min > /tmp/attn_min_out.txt
"$PY" -c "
import numpy as np
expected = np.fromfile('expected.f32', dtype=np.float32)
got = np.array([float(l) for l in open('/tmp/attn_min_out.txt') if l.strip()],
               dtype=np.float32)
if got.shape != expected.shape:
    raise SystemExit(f'shape mismatch: expected {expected.shape}, got {got.shape}')
diff = np.abs(got - expected)
print(f'  max abs diff:  {diff.max():.3e}')
print(f'  mean abs diff: {diff.mean():.3e}')
if diff.max() >= 1e-3:
    raise SystemExit(f'FAIL: max diff {diff.max():.3e} exceeds 1e-3 tolerance')
print('  ATTN_OK — packed binary matches PyTorch attention reference')
"
