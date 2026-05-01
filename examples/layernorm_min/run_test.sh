#!/bin/bash
# layernorm_min — packed-binary LayerNorm regression.
#
# Builds a synthetic LayerNorm(dim=16) → Linear(16→8) checkpoint with
# deterministic random γ / β / W / b, packs it through aricode-pack
# with --input-format embedded, runs the static ELF, asserts the
# output matches PyTorch's torch.nn.LayerNorm + Linear within 1e-3.
#
# Locks in: LayerNorm.weight / LayerNorm.bias key resolution, the
# layernorm_affine_f32 user-fn pipeline, in-place buffer reuse
# through gen_act_decls, and the LayerNorm → Linear shape contract.

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
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format embedded --input-file "$(pwd)/X_input.f32" \
    --embed --no-argmax --out ln_min > /dev/null

echo "[3/4] compiling .ari → static ELF..."
"$ARIC" ln_min.ari -o ln_min > /dev/null

echo "[4/4] running + comparing to PyTorch reference..."
./ln_min > /tmp/ln_min_out.txt
"$PY" -c "
import numpy as np
expected = np.fromfile('expected.f32', dtype=np.float32)
got = np.array([float(l) for l in open('/tmp/ln_min_out.txt') if l.strip()],
               dtype=np.float32)
if got.shape != expected.shape:
    raise SystemExit(f'shape mismatch: expected {expected.shape}, got {got.shape}')
diff = np.abs(got - expected)
print(f'  max abs diff:  {diff.max():.3e}')
print(f'  mean abs diff: {diff.mean():.3e}')
if diff.max() >= 1e-3:
    raise SystemExit(f'FAIL: max diff {diff.max():.3e} exceeds 1e-3 tolerance')
print('  LN_OK — packed binary matches PyTorch LayerNorm + Linear reference')
"
