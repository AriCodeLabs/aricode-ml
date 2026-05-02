#!/bin/bash
# tiny_decoder_packed — pack.py-generated decoder regression.
#
# Same architecture as tiny_decoder_min, but every line of the .ari is
# emitted automatically by pack.py from arch.json + the saved state
# dict.  Validates that the new --decoder-loop / multi_head_attention_kv
# wiring in pack.py produces a binary that token-for-token matches
# PyTorch's greedy autoregressive reference.

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
    --max-new-tokens 3 \
    --out tiny_decoder_packed > /dev/null

echo "[3/4] compiling .ari → static ELF..."
"$ARIC" tiny_decoder_packed.ari -o tiny_decoder_packed > /dev/null
ls -la tiny_decoder_packed | awk '{printf "  binary size: %s bytes\n", $5}'

echo "[4/4] running + comparing token IDs to PyTorch..."
./tiny_decoder_packed > /tmp/tiny_decoder_packed_out.txt
"$PY" -c "
import struct
expected = []
with open('expected_tokens.bin', 'rb') as f:
    while True:
        b = f.read(4)
        if not b: break
        expected.append(struct.unpack('<i', b)[0])
got = [int(line) for line in open('/tmp/tiny_decoder_packed_out.txt') if line.strip()]
print(f'  PyTorch tokens: {expected}')
print(f'  aricode tokens: {got}')
if got != expected:
    raise SystemExit(f'FAIL: token mismatch')
print('  TINY_DECODER_PACKED_OK — pack.py decoder mode matches PyTorch')
"
