"""Download distilbert-base-uncased-finetuned-sst-2 from HuggingFace,
re-key the state_dict into aricode-pack's naming convention, run the
PyTorch reference forward pass on a fixed input, and dump everything
the .ari side needs:

  synth.pt        — re-keyed state_dict (HF distilbert key naming for
                    LayerNorm + MHA + embedding, fc{N} for FFN linears
                    + pre_classifier + classifier — matches the
                    packer's --keys "fc{idx_plus_1}.{kind}" template).
  tokens.bin      — input token IDs as little-endian uint16, padded to
                    seq=16 with [PAD]=0.
  expected.f32    — PyTorch logits for [CLS] (row 0): 2 floats.

We hardcode a fixed test sentence so the whole pipeline is
deterministic.  Pick "I love this movie, it was fantastic" — strong
positive sentiment so the [POS] logit dominates.

Run before pack.sh.
"""

import sys
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
SENTENCE = "I love this movie, it was fantastic"
SEQ = 16   # pad/truncate input to this length

print(f"[1/3] loading {MODEL} ...", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
tok = AutoTokenizer.from_pretrained(MODEL)

print(f"[2/3] tokenizing: {SENTENCE!r}", flush=True)
ids = tok.encode(SENTENCE, add_special_tokens=True)
if len(ids) > SEQ:
    ids = ids[:SEQ]
else:
    ids = ids + [tok.pad_token_id] * (SEQ - len(ids))
print(f"  token ids (len {SEQ}): {ids}")

# uint16 little-endian: aricode's 2-byte token loader path.
np.array(ids, dtype=np.uint16).tofile("tokens.bin")

print(f"[3/3] running PyTorch forward + re-keying state_dict ...", flush=True)
with torch.no_grad():
    input_ids = torch.tensor([ids], dtype=torch.long)
    out = model(input_ids).logits[0]   # [2]
expected = out.detach().numpy().astype(np.float32)
expected.tofile("expected.f32")
print(f"  PyTorch logits: {expected.tolist()}")
print(f"  predicted label: {'positive' if expected[1] > expected[0] else 'negative'}")

# Re-key the state_dict.  We accept HF distilbert's key conventions
# directly for LayerNorm + MHA + embedding (the packer's resolver
# handles those).  But for the 14 Linears (12 FFN + 2 classifier head)
# we need to flatten the nested transformer.layer.{i}.ffn.lin{1,2}
# names into the fc{N} form the packer's --keys template expects.

sd = model.state_dict()

# Strip the "distilbert." prefix from non-classifier keys.  The
# packer's resolver patterns are written without that prefix.
def strip_prefix(k):
    if k.startswith("distilbert."):
        return k[len("distilbert."):]
    return k

remapped = {}
for k, v in sd.items():
    if k.startswith("distilbert.transformer.layer."):
        # Examples:
        #   distilbert.transformer.layer.0.ffn.lin1.weight  →  fc1.weight
        #   distilbert.transformer.layer.0.ffn.lin2.weight  →  fc2.weight
        #   distilbert.transformer.layer.5.ffn.lin1.weight  →  fc11.weight
        #   distilbert.transformer.layer.5.ffn.lin2.weight  →  fc12.weight
        # Other transformer.layer.{i}.* keys stay (LayerNorm + MHA
        # use the resolver patterns we already have).
        parts = k.split(".")
        # parts: ['distilbert', 'transformer', 'layer', '<i>', 'ffn'?, 'lin{1,2}'?, '<wb>']
        if len(parts) >= 7 and parts[4] == "ffn":
            block = int(parts[3])
            lin = parts[5]   # 'lin1' or 'lin2'
            wb = parts[6]    # 'weight' or 'bias'
            li = block * 2 + (1 if lin == "lin1" else 2)
            remapped[f"fc{li}.{wb}"] = v
        else:
            # MHA (q_lin/k_lin/v_lin/out_lin) and LayerNorm (sa, output)
            # — strip the distilbert. prefix so the packer's existing
            # patterns (transformer.layer.{i}.attention.q_lin etc.)
            # match.
            remapped[strip_prefix(k)] = v
    elif k.startswith("distilbert.embeddings."):
        # embeddings.word_embeddings.weight  →  same (after strip)
        # embeddings.position_embeddings.weight  →  same
        # embeddings.LayerNorm.{weight,bias}  →  same
        remapped[strip_prefix(k)] = v
    elif k == "pre_classifier.weight":
        remapped["fc13.weight"] = v
    elif k == "pre_classifier.bias":
        remapped["fc13.bias"] = v
    elif k == "classifier.weight":
        remapped["fc14.weight"] = v
    elif k == "classifier.bias":
        remapped["fc14.bias"] = v
    else:
        remapped[k] = v

torch.save(remapped, "synth.pt")
print(f"  re-keyed state_dict: {len(remapped)} tensors → synth.pt")
print(f"  fc1..fc14 are FFN block 0..5 + pre_classifier + classifier")

print()
print(f"Next: bash pack.sh   # to invoke aricode-pack and build the binary")
