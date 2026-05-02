#!/bin/bash
# tiny_decoder_2block (int8) — 2-block decoder with per-block KV cache,
# packed with --quantize int8.  Validates that the decoder loop still
# routes correctly when each block's FFN Linears use the int8 matvec
# kernel while the MHA-KV path stays f32.

set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PACK="$SCRIPT_DIR/../../tools/aricode_pack.py"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

echo "[1/4] generating 2-block synth checkpoint + reference..."
"$PY" prepare.py | tail -3

echo "[2/4] packing through pack.py decoder mode (int8)..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format embedded --input-file "$(pwd)/prompt.bin" \
    --embed --no-argmax \
    --quantize int8 \
    --max-new-tokens 4 \
    --out tiny_decoder_2block_int8 > /dev/null

echo "[3/4] compiling .ari → static ELF..."
"$ARIC" tiny_decoder_2block_int8.ari -o tiny_decoder_2block_int8 > /dev/null
ls -la tiny_decoder_2block_int8 | awk '{printf "  binary size: %s bytes\n", $5}'

echo "[4/4] running + comparing token IDs to PyTorch..."
./tiny_decoder_2block_int8 > /tmp/tiny_decoder_2block_int8_out.txt
"$PY" -c "
import struct
expected = []
with open('expected_tokens.bin', 'rb') as f:
    while True:
        b = f.read(4)
        if not b: break
        expected.append(struct.unpack('<i', b)[0])
got = [int(line) for line in open('/tmp/tiny_decoder_2block_int8_out.txt') if line.strip()]
print(f'  PyTorch tokens: {expected}')
print(f'  aricode tokens: {got}')
if got != expected:
    raise SystemExit(f'FAIL: token mismatch')
print('  TINY_2BLOCK_OK (int8) — 2-block decoder + int8 matches PyTorch')
"
