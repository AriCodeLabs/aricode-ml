#!/bin/bash
# gpt2_small — pack HuggingFace GPT-2 (124M) as a static ELF and
# greedy-decode 16 tokens, comparing against the in-memory PyTorch
# reference.  The trophy demo: a real LLM packed end-to-end.
#
# Resource budget: ~500 MB of weights + ~75 MB KV cache (12 layers
# × max_seq=128 × 768 × 2).  Build is single-file aric → ELF;
# expect ~30 s for the C compile + a few seconds for pack.

set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PACK="$SCRIPT_DIR/../../tools/aricode_pack.py"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

echo "[1/4] downloading + re-keying GPT-2 124M..."
"$PY" prepare.py | tail -20

echo "[2/4] packing through pack.py decoder mode..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format embedded --input-file "$(pwd)/prompt.bin" \
    --embed --no-argmax \
    --max-new-tokens 16 \
    --out gpt2_small > /tmp/gpt2_small_pack.log 2>&1 || {
        echo "PACK FAILED — tail of log:"; tail -30 /tmp/gpt2_small_pack.log; exit 1;
    }
ls -la gpt2_small.ari | awk '{printf "  .ari source size: %s bytes\n", $5}'

echo "[3/4] compiling .ari → static ELF (this may take ~30 s)..."
"$ARIC" gpt2_small.ari -o gpt2_small > /tmp/gpt2_small_build.log 2>&1 || {
        echo "COMPILE FAILED — tail of log:"; tail -30 /tmp/gpt2_small_build.log; exit 1;
    }
ls -la gpt2_small | awk '{printf "  binary size: %s bytes (%.1f MB)\n", $5, $5/1048576}'

echo "[4/4] running + comparing token IDs to PyTorch..."
./gpt2_small > /tmp/gpt2_small_out.txt
"$PY" -c "
import struct
expected = []
with open('expected_tokens.bin', 'rb') as f:
    while True:
        b = f.read(4)
        if not b: break
        expected.append(struct.unpack('<i', b)[0])
got = [int(line) for line in open('/tmp/gpt2_small_out.txt') if line.strip()]
print(f'  PyTorch tokens: {expected}')
print(f'  aricode tokens: {got}')
n_match = 0
for a, b in zip(got, expected):
    if a == b: n_match += 1
    else: break
print(f'  prefix match  : {n_match} / {len(expected)}')
if got == expected:
    print('  GPT2_SMALL_OK — full 16-token greedy match against PyTorch reference')
elif n_match >= 5:
    print(f'  GPT2_SMALL_PARTIAL_OK — first {n_match} tokens match (>=5 acceptable)')
else:
    raise SystemExit(f'FAIL: only {n_match} tokens match')
"
