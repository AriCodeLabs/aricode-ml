#!/bin/bash
# rmsnorm_swiglu_decoder — composition test.
# 1-block decoder using the new RMSNorm + SwiGLU arch ops alongside
# the existing multi_head_attention_kv decoder pipeline.  Validates
# that the new pieces compose: per-token decoder loop + KV cache +
# embedding lookup + sampling, plus RMSNorm + SwiGLU.
#
# Once RoPE+GQA land, swapping multi_head_attention_kv for
# multi_head_attention_gqa_kv (and adding rope_theta) is the only
# structural change needed to reach actual TinyLlama.

set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PACK="$SCRIPT_DIR/../../tools/aricode_pack.py"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

echo "[1/4] generating synthetic checkpoint + reference..."
"$PY" prepare.py | tail -3

echo "[2/4] packing through pack.py decoder mode..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format embedded --input-file "$(pwd)/prompt.bin" \
    --embed --no-argmax \
    --max-new-tokens 3 \
    --out rmsnorm_swiglu_decoder > /dev/null

echo "[3/4] compiling .ari → static ELF..."
"$ARIC" rmsnorm_swiglu_decoder.ari -o rmsnorm_swiglu_decoder > /dev/null
ls -la rmsnorm_swiglu_decoder | awk '{printf "  binary size: %s bytes\n", $5}'

echo "[4/4] running + comparing token IDs to PyTorch..."
./rmsnorm_swiglu_decoder > /tmp/rmsnorm_swiglu_decoder_out.txt
"$PY" -c "
import struct
expected = []
with open('expected_tokens.bin', 'rb') as f:
    while True:
        b = f.read(4)
        if not b: break
        expected.append(struct.unpack('<i', b)[0])
got = [int(line) for line in open('/tmp/rmsnorm_swiglu_decoder_out.txt') if line.strip()]
print(f'  PyTorch tokens: {expected}')
print(f'  aricode tokens: {got}')
if got != expected:
    raise SystemExit(f'FAIL: token mismatch')
print('  RMSNORM_SWIGLU_DECODER_OK — RMSNorm + SwiGLU + MHA-KV decoder matches PyTorch')
"
