#!/bin/bash
# gqa_min — sanity check: GQA-KV kernel with n_kv_heads == n_heads
# (group_size = 1) reduces to standard MHA + RoPE.

set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PACK="$SCRIPT_DIR/../../tools/aricode_pack.py"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

echo "[1/4] generating synthetic checkpoint + reference..."
"$PY" prepare.py | tail -4

echo "[2/4] packing through pack.py decoder mode..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format embedded --input-file "$(pwd)/prompt.bin" \
    --embed --no-argmax \
    --max-new-tokens 4 \
    --out gqa_min > /dev/null

echo "[3/4] compiling .ari → static ELF..."
"$ARIC" gqa_min.ari -o gqa_min > /dev/null
ls -la gqa_min | awk '{printf "  binary size: %s bytes\n", $5}'

echo "[4/4] running + comparing token IDs to PyTorch..."
./gqa_min > /tmp/gqa_min_out.txt
"$PY" -c "
import struct
expected = []
with open('expected_tokens.bin', 'rb') as f:
    while True:
        b = f.read(4)
        if not b: break
        expected.append(struct.unpack('<i', b)[0])
got = [int(line) for line in open('/tmp/gqa_min_out.txt') if line.strip()]
print(f'  PyTorch tokens: {expected}')
print(f'  aricode tokens: {got}')
if got != expected:
    raise SystemExit(f'FAIL: token mismatch')
print('  GQA_MIN_OK — GQA-KV with n_kv_heads == n_heads matches PyTorch')
"
