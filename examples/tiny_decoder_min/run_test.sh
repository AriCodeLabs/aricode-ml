#!/bin/bash
# tiny_decoder_min — end-to-end autoregressive decoder regression.
#
# Validates the full Track A stack (KV-cache MH attention +
# greedy sampling + decoder loop) against PyTorch token-for-token.
#
# Architecture: 1-block GPT-style transformer with vocab=16, d_model=8,
# n_heads=2, ff=16.  Greedy decode 3 tokens from a 2-token prompt.
# Pass condition: emitted token IDs equal PyTorch's argmax sequence.

set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

echo "[1/3] generating synthetic checkpoint + reference greedy decode..."
"$PY" prepare.py | tail -3

echo "[2/3] compiling tiny_decoder.ari → static ELF..."
"$ARIC" tiny_decoder.ari -o tiny_decoder > /dev/null
ls -la tiny_decoder | awk '{printf "  binary size: %s bytes\n", $5}'

echo "[3/3] running decoder + comparing token IDs to PyTorch..."
./tiny_decoder > /tmp/tiny_decoder_out.txt
"$PY" -c "
import struct
expected = []
with open('expected_tokens.bin', 'rb') as f:
    while True:
        b = f.read(4)
        if not b: break
        expected.append(struct.unpack('<i', b)[0])
got = [int(line) for line in open('/tmp/tiny_decoder_out.txt') if line.strip()]
print(f'  PyTorch tokens: {expected}')
print(f'  aricode tokens: {got}')
if got != expected:
    raise SystemExit(f'FAIL: token mismatch')
print('  TINY_DECODER_OK — KV-cache decoder matches PyTorch token-for-token')
"
