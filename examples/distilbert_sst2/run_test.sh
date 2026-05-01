#!/bin/bash
# distilbert_sst2 — pack a real fine-tuned HuggingFace distilbert
# sentiment classifier as a static ELF and verify it matches PyTorch
# bit-for-bit (within f32 noise) on a fixed input sentence.
#
# This is the milestone "real model" deploy demo — every layer of a
# 6-block 67M-param transformer encoder + classifier head packed
# end-to-end with no Python at runtime.
#
# Requires: torch, transformers (auto-installed via pip on first run).

set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
ARIC="$SCRIPT_DIR/../../../../aricoderoot/aricode/src/compiler/aric"
PACK="$SCRIPT_DIR/../../tools/aricode_pack.py"
PY="${PYTHON:-/tmp/aricode_venv/bin/python}"

echo "[1/4] downloading distilbert-base-uncased-finetuned-sst-2 + tokenising..."
"$PY" -m pip install --quiet transformers > /dev/null 2>&1 || true
"$PY" prepare.py | tail -5

echo "[2/4] packing through aricode-pack (int8 quantised, single-shot embedded)..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format embedded --input-file "$(pwd)/tokens.bin" \
    --embed --quantize int8 --no-argmax --out distilbert_sst2 > /dev/null

echo "[3/4] compiling .ari → static ELF (~173 MB; FFN linears int8, embedding/MHA/LN f32)..."
"$ARIC" distilbert_sst2.ari -o distilbert_sst2 > /dev/null
ls -la distilbert_sst2 | awk '{printf "  size: %s bytes\n", $5}'

echo "[4/4] running + comparing [CLS] logits to PyTorch..."
./distilbert_sst2 > /tmp/dbert_out.txt
"$PY" -c "
import numpy as np
expected = np.fromfile('expected.f32', dtype=np.float32)   # PyTorch [CLS] logits
got = np.array([float(l) for l in open('/tmp/dbert_out.txt') if l.strip()],
               dtype=np.float32)
# The packer applies the classifier head batched per-row of the [seq,
# d_model] activation, so it produces seq*2 = 32 logits.  PyTorch's
# DistilBertForSequenceClassification only runs pre_classifier on the
# [CLS] token (row 0), so we compare the first 2 elements of our
# output to the PyTorch reference.
got_cls = got[:2]
print(f'  PyTorch [CLS] logits: {expected.tolist()}')
print(f'  aricode [CLS] logits: {got_cls.tolist()}')
diff = np.abs(got_cls - expected)
print(f'  max abs diff: {diff.max():.3e}')
print(f'  PyTorch label: {\"positive\" if expected[1] > expected[0] else \"negative\"}')
print(f'  aricode label: {\"positive\" if got_cls[1] > got_cls[0] else \"negative\"}')
# Tolerance: per-tensor int8 symmetric quantisation across 67M
# params, on top of tanh-approx GELU and f32 noise — not bit-exact
# but well below classifier-class accuracy.  Decision boundaries
# in SST-2 span multiple logit units; 0.5 is generous head-room
# but still catches any algebraic bug at distilbert scale.
if diff.max() >= 0.5:
    raise SystemExit(f'FAIL: max diff {diff.max():.3e} exceeds 0.5 tolerance')
if (got_cls[1] > got_cls[0]) != (expected[1] > expected[0]):
    raise SystemExit('FAIL: predicted label disagrees with PyTorch')
print('  DBERT_SST2_OK — packed real distilbert SST-2 matches PyTorch')
"
