#!/bin/bash
# tiny_decoder_packed (int8) — pack.py-generated decoder regression with
# --quantize int8.  Same as run_test.sh but adds the int8 flag so the
# Linear layers (FFN + LM-head) get per-tensor symmetric int8 weights;
# the embedding / MHA-KV / layernorm tensors stay f32 (QUANTISABLE in
# main() guards them).  Validates token-for-token match against PyTorch.

set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PACK="$SCRIPT_DIR/../../tools/aricode_pack.py"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

echo "[1/4] generating synthetic checkpoint + reference..."
"$PY" prepare.py | tail -4

echo "[2/4] packing through pack.py decoder mode (int8)..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format embedded --input-file "$(pwd)/prompt.bin" \
    --embed --no-argmax \
    --quantize int8 \
    --max-new-tokens 3 \
    --out tiny_decoder_packed_int8 > /dev/null

echo "[3/4] compiling .ari → static ELF..."
"$ARIC" tiny_decoder_packed_int8.ari -o tiny_decoder_packed_int8 > /dev/null
ls -la tiny_decoder_packed_int8 | awk '{printf "  binary size: %s bytes\n", $5}'

echo "[4/4] running + comparing token IDs to PyTorch..."
./tiny_decoder_packed_int8 > /tmp/tiny_decoder_packed_int8_out.txt
"$PY" -c "
import struct
expected = []
with open('expected_tokens.bin', 'rb') as f:
    while True:
        b = f.read(4)
        if not b: break
        expected.append(struct.unpack('<i', b)[0])
got = [int(line) for line in open('/tmp/tiny_decoder_packed_int8_out.txt') if line.strip()]
print(f'  PyTorch tokens: {expected}')
print(f'  aricode tokens: {got}')
if got != expected:
    raise SystemExit(f'FAIL: token mismatch')
print('  TINY_DECODER_PACKED_OK (int8) — pack.py decoder mode + int8 matches PyTorch')
"
