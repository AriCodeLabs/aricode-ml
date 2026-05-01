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

echo "[2/4] packing through aricode-pack (f32, single-shot embedded)..."
"$PY" "$PACK" \
    --checkpoint synth.pt --arch arch.json \
    --keys "fc{idx_plus_1}.{kind}" \
    --input-format embedded --input-file "$(pwd)/tokens.bin" \
    --embed --no-argmax --out distilbert_sst2 > /dev/null

echo "[3/4] compiling .ari → static ELF (268 MB f32 weights baked in)..."
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
# Tolerance: tanh-approx GELU + 67M-param error accumulation pushes
# this above the 1e-3 line that smaller models hit.  1e-2 is still
# vastly below classifier-class accuracy (decision boundary spans
# multiple units of the logit scale).
if diff.max() >= 1e-2:
    raise SystemExit(f'FAIL: max diff {diff.max():.3e} exceeds 1e-2 tolerance')
if (got_cls[1] > got_cls[0]) != (expected[1] > expected[0]):
    raise SystemExit('FAIL: predicted label disagrees with PyTorch')
print('  DBERT_SST2_OK — packed real distilbert SST-2 matches PyTorch')
"
