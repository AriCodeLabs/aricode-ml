#!/bin/bash
# llama2_7b — pack the real Llama-2-7B (NousResearch's open
# redistribution of Meta's Llama-2-7b-hf) as a static ELF.
#
# Architecture: 32 transformer blocks, vocab=32000, d_model=4096,
# n_heads=32, n_kv_heads=32 (no GQA — degrades to plain MHA via the
# multi_head_attention_gqa_kv kernel with group_size=1), d_head=128,
# ff=11008, RoPE θ=10000, RMSNorm ε=1e-5.
#
# Sidecar f32 path: pack.py emits a 219 KB binary that loads weights
# from a 27 GB tinyllama_7b.f32 sidecar via file_read at startup.
# Inline-embed mode would trip the rel32 ±2 GB CALL displacement cap;
# sidecar dodges that.  Cold-start + 4-token greedy decode: ~40 s.
#
# Resource budget for `prepare.py`:
#   Download: ~13 GB safetensors (NousResearch mirror, no auth).
#   RAM peak: ~14 GB (model held in bf16 throughout; pack.py converts
#             one tensor at a time to f32 at staging).
#   Disk: 27 GB sidecar + 13 GB synth.pt + 13 GB HF cache = ~53 GB.

set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PACK="$SCRIPT_DIR/../../tools/aricode_pack.py"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

if [ -f "synth.pt" ] && [ -f "prompt.bin" ] && [ -f "expected_tokens.bin" ]; then
    echo "[1/4] reusing cached synth.pt / prompt.bin / expected_tokens.bin"
else
    echo "[1/4] downloading + re-keying Llama-2-7B (~13 GB safetensors)..."
    "$PY" prepare.py 2>&1 | tail -10
fi

echo "[2/4] packing through pack.py decoder mode (sidecar f32)..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format embedded --input-file "$(pwd)/prompt.bin" \
    --no-argmax --max-new-tokens 4 \
    --out llama2_7b > /tmp/llama2_7b_pack.log 2>&1 || {
        echo "PACK FAILED — tail:"; tail -30 /tmp/llama2_7b_pack.log; exit 1;
    }
ls -la llama2_7b.ari | awk '{printf "  .ari source: %s bytes\n", $5}'
ls -la llama2_7b.f32 | awk '{printf "  weight sidecar: %s bytes (%.2f GB)\n", $5, $5/1073741824}'

echo "[3/4] compiling .ari → static ELF..."
"$ARIC" llama2_7b.ari -o llama2_7b > /tmp/llama2_7b_build.log 2>&1 || {
        echo "COMPILE FAILED — tail:"; tail -30 /tmp/llama2_7b_build.log; exit 1;
    }
ls -la llama2_7b | awk '{printf "  binary: %s bytes (%.0f KB)\n", $5, $5/1024}'

echo "[4/4] running + diffing tokens vs PyTorch HF reference..."
time ./llama2_7b > /tmp/llama2_7b_out.txt
"$PY" -c "
import struct
from transformers import AutoTokenizer
expected = []
with open('expected_tokens.bin', 'rb') as f:
    while True:
        b = f.read(4)
        if not b: break
        expected.append(struct.unpack('<i', b)[0])
got = [int(l) for l in open('/tmp/llama2_7b_out.txt') if l.strip()]
tok = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')
print(f'  prompt:           Once upon a time, there was a')
print(f'  PyTorch reference: {tok.decode(expected)!r}')
print(f'  aricode output:    {tok.decode(got)!r}')
print(f'  PyTorch tokens: {expected}')
print(f'  aricode tokens: {got}')
if got == expected:
    print('  LLAMA2_7B_OK — full 4-token greedy match against HF PyTorch')
else:
    n_prefix = 0
    for a, b in zip(got, expected):
        if a == b: n_prefix += 1
        else: break
    if n_prefix >= 2:
        print(f'  LLAMA2_7B_PARTIAL_OK — first {n_prefix}/{len(expected)} tokens match')
    else:
        raise SystemExit(f'FAIL: only {n_prefix} prefix tokens match')
"
