#!/bin/bash
# tiny_decoder_2block — 2-block decoder with per-block KV cache.
# Validates that pack.py's decoder mode allocates and routes one
# independent KV state per multi_head_attention_kv layer (the
# pattern used by every real LLM with N stacked transformer blocks).

set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PACK="$SCRIPT_DIR/../../tools/aricode_pack.py"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

echo "[1/4] generating 2-block synth checkpoint + reference..."
"$PY" prepare.py | tail -3

echo "[2/4] packing through pack.py decoder mode..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format embedded --input-file "$(pwd)/prompt.bin" \
    --embed --no-argmax \
    --max-new-tokens 4 \
    --out tiny_decoder_2block > /dev/null

echo "[3/4] compiling .ari → static ELF..."
"$ARIC" tiny_decoder_2block.ari -o tiny_decoder_2block > /dev/null
ls -la tiny_decoder_2block | awk '{printf "  binary size: %s bytes\n", $5}'

echo "[4/4] running + comparing token IDs to PyTorch..."
./tiny_decoder_2block > /tmp/tiny_decoder_2block_out.txt
"$PY" -c "
import struct
expected = []
with open('expected_tokens.bin', 'rb') as f:
    while True:
        b = f.read(4)
        if not b: break
        expected.append(struct.unpack('<i', b)[0])
got = [int(line) for line in open('/tmp/tiny_decoder_2block_out.txt') if line.strip()]
print(f'  PyTorch tokens: {expected}')
print(f'  aricode tokens: {got}')
if got != expected:
    raise SystemExit(f'FAIL: token mismatch')
print('  TINY_2BLOCK_OK — 2-block decoder with per-block KV state matches PyTorch')
"
