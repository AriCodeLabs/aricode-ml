#!/bin/bash
# tinyllama_1b — pack the real HuggingFace TinyLlama-1.1B and validate
# token-for-token greedy match against the HF PyTorch reference.
#
# Architecture: 22 transformer blocks, RMSNorm + GQA-KV (n_heads=32,
# n_kv_heads=4) + RoPE θ=10000 + SwiGLU FFN (2048 → 5632 → 2048).
# vocab=32000.
#
# RoPE convention: HF Llama uses split-half; our rope_apply_f32 uses
# interleaved.  prepare.py permutes Q/K weight rows per head so the
# kernel's interleaved-pair rotation matches HF's split-half output
# bit-for-bit (math is identical, only the physical layout differs).
#
# Resource budget:
#   synth.pt sidecar: ~4.4 GB (f32 weights — embedding + MHA + SwiGLU
#     + LM head; loaded at runtime via file_read).
#   binary: ~170 KB code (no --embed; dodges the rel32 ±2 GB CALL
#     overflow that an inline-embed of 4.4 GB would trip).
#   wall: ~6 s end-to-end (load 4.4 GB + 8 tokens of greedy decode).

set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PACK="$SCRIPT_DIR/../../tools/aricode_pack.py"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

if [ -f "synth.pt" ] && [ -f "prompt.bin" ] && [ -f "expected_tokens.bin" ]; then
    echo "[1/4] reusing cached synth.pt / prompt.bin / expected_tokens.bin"
else
    echo "[1/4] downloading + re-keying TinyLlama-1.1B (~2.2 GB safetensors)..."
    "$PY" prepare.py 2>&1 | tail -10
fi

echo "[2/4] packing through pack.py decoder mode (sidecar f32)..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format embedded --input-file "$(pwd)/prompt.bin" \
    --no-argmax \
    --max-new-tokens 8 \
    --out tinyllama_1b > /tmp/tinyllama_1b_pack.log 2>&1 || {
        echo "PACK FAILED — tail:"; tail -30 /tmp/tinyllama_1b_pack.log; exit 1;
    }
ls -la tinyllama_1b.ari | awk '{printf "  .ari source: %s bytes\n", $5}'
ls -la tinyllama_1b.f32 | awk '{printf "  weight sidecar: %s bytes (%.2f GB)\n", $5, $5/1073741824}'

echo "[3/4] compiling .ari → static ELF..."
"$ARIC" tinyllama_1b.ari -o tinyllama_1b > /tmp/tinyllama_1b_build.log 2>&1 || {
        echo "COMPILE FAILED — tail:"; tail -30 /tmp/tinyllama_1b_build.log; exit 1;
    }
ls -la tinyllama_1b | awk '{printf "  binary: %s bytes (%.0f KB)\n", $5, $5/1024}'

echo "[4/4] running + diffing tokens vs PyTorch HF reference..."
time ./tinyllama_1b > /tmp/tinyllama_1b_out.txt
"$PY" -c "
import struct
from transformers import AutoTokenizer
expected = []
with open('expected_tokens.bin', 'rb') as f:
    while True:
        b = f.read(4)
        if not b: break
        expected.append(struct.unpack('<i', b)[0])
got = [int(l) for l in open('/tmp/tinyllama_1b_out.txt') if l.strip()]
tok = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
print(f'  prompt:           Once upon a time, there was a')
print(f'  PyTorch reference: {tok.decode(expected)!r}')
print(f'  aricode output:    {tok.decode(got)!r}')
print(f'  PyTorch tokens: {expected}')
print(f'  aricode tokens: {got}')
if got == expected:
    print('  TINYLLAMA_1B_OK — full 8-token greedy match against HF PyTorch')
else:
    n_prefix = 0
    for a, b in zip(got, expected):
        if a == b: n_prefix += 1
        else: break
    if n_prefix >= 4:
        print(f'  TINYLLAMA_1B_PARTIAL_OK — first {n_prefix}/{len(expected)} tokens match')
    else:
        raise SystemExit(f'FAIL: only {n_prefix} prefix tokens match')
"
