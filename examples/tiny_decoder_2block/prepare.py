"""tiny_decoder_2block — 2-block decoder validating per-layer KV state.

Same architectural pattern as tiny_decoder_packed but with TWO transformer
blocks back-to-back, larger config:
  vocab=32, d_model=16, n_heads=4 (d_head=4), max_seq=8, ff=32

State_dict keys use a per-block layered convention so pack.py's MHA
resolver finds them by layer index:
  embed.weight                                   → embedding (ei=0)
  ln_0.weight / ln_0.bias                        → block 0 pre-attn LN (ni=0)
  layers.0.q_proj.weight / .bias                 → block 0 MH-KV Q (mi=0)
  layers.0.k_proj.weight / .bias                 → block 0 MH-KV K
  layers.0.v_proj.weight / .bias                 → block 0 MH-KV V
  layers.0.out_proj.weight / .bias               → block 0 MH-KV O
  ln_1.weight / ln_1.bias                        → block 0 pre-FFN LN  (ni=1)
  fc1.weight  / fc1.bias                         → block 0 FFN1        (li=0)
  fc2.weight  / fc2.bias                         → block 0 FFN2        (li=1)
  ln_2.weight / ln_2.bias                        → block 1 pre-attn LN (ni=2)
  layers.1.q_proj.* / k_proj.* / v_proj.* / out_proj.*   → block 1 MH-KV
  ln_3.weight / ln_3.bias                        → block 1 pre-FFN LN
  fc3.weight / fc3.bias                          → block 1 FFN1
  fc4.weight / fc4.bias                          → block 1 FFN2
  ln_4.weight / ln_4.bias                        → final LN
  fc5.weight / fc5.bias                          → LM head
"""

import math
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(11)

VOCAB    = 32
D_MODEL  = 16
N_HEADS  = 4
D_HEAD   = D_MODEL // N_HEADS
MAX_SEQ  = 8
FF       = 32
N_BLOCKS = 2
PROMPT   = [3, 7, 1]
MAX_NEW  = 4


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj   = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj   = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj   = nn.Linear(D_MODEL, D_MODEL)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL)


class TinyDecoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.layers = nn.ModuleList([Block() for _ in range(N_BLOCKS)])
        self.lns = nn.ModuleList([
            nn.LayerNorm(D_MODEL, eps=1e-5)
            for _ in range(2 * N_BLOCKS + 1)
        ])
        self.ffs = nn.ModuleList([
            nn.Linear(D_MODEL, FF) if i % 2 == 0 else nn.Linear(FF, D_MODEL)
            for _ in range(N_BLOCKS) for i in range(2)
        ])
        self.lm = nn.Linear(D_MODEL, VOCAB)

    def forward(self, ids):
        x = self.embed(ids)
        seq = ids.shape[0]
        mask = torch.full((seq, seq), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        for b in range(N_BLOCKS):
            ln_attn = self.lns[2 * b]
            ln_ff = self.lns[2 * b + 1]
            blk = self.layers[b]
            ff1 = self.ffs[2 * b]
            ff2 = self.ffs[2 * b + 1]
            h = ln_attn(x)
            Q = blk.q_proj(h).view(-1, N_HEADS, D_HEAD)
            K = blk.k_proj(h).view(-1, N_HEADS, D_HEAD)
            V = blk.v_proj(h).view(-1, N_HEADS, D_HEAD)
            outs = []
            for h_idx in range(N_HEADS):
                scores = Q[:, h_idx] @ K[:, h_idx].t() / math.sqrt(D_HEAD)
                scores = scores + mask
                soft = F.softmax(scores, dim=-1)
                outs.append(soft @ V[:, h_idx])
            concat = torch.cat(outs, dim=-1)
            x = x + blk.out_proj(concat)
            h2 = ln_ff(x)
            h2 = ff1(h2)
            h2 = F.gelu(h2, approximate="tanh")
            h2 = ff2(h2)
            x = x + h2
        x = self.lns[-1](x)
        return self.lm(x)


model = TinyDecoder2()
model.eval()

# Re-key into the layout pack.py expects.  HuggingFace MHA-KV resolver
# scans `q_proj`-style keys with optional `layers.{mi}.` prefix; LN
# resolver scans `ln_{ni}` fallbacks; Linear takes the --keys template
# we'll pass on the pack.py command line (`fc{idx_plus_1}.{kind}`).
sd = {}
sd["embed.weight"] = model.embed.weight
for b in range(N_BLOCKS):
    blk = model.layers[b]
    sd[f"layers.{b}.q_proj.weight"]   = blk.q_proj.weight
    sd[f"layers.{b}.q_proj.bias"]     = blk.q_proj.bias
    sd[f"layers.{b}.k_proj.weight"]   = blk.k_proj.weight
    sd[f"layers.{b}.k_proj.bias"]     = blk.k_proj.bias
    sd[f"layers.{b}.v_proj.weight"]   = blk.v_proj.weight
    sd[f"layers.{b}.v_proj.bias"]     = blk.v_proj.bias
    sd[f"layers.{b}.out_proj.weight"] = blk.out_proj.weight
    sd[f"layers.{b}.out_proj.bias"]   = blk.out_proj.bias
for i, ln in enumerate(model.lns):
    sd[f"ln_{i}.weight"] = ln.weight
    sd[f"ln_{i}.bias"]   = ln.bias
# Linear keys: the --keys template `fc{idx_plus_1}.{kind}` resolves
# li=0 → fc1, li=1 → fc2, ..., li=4 → fc5.
li = 0
for ff in model.ffs:
    sd[f"fc{li + 1}.weight"] = ff.weight
    sd[f"fc{li + 1}.bias"]   = ff.bias
    li += 1
sd[f"fc{li + 1}.weight"] = model.lm.weight
sd[f"fc{li + 1}.bias"]   = model.lm.bias
torch.save(sd, "synth.pt")

with open("prompt.bin", "wb") as f:
    for tok in PROMPT:
        f.write(struct.pack("<B", tok))

ids = list(PROMPT)
generated = []
with torch.no_grad():
    for _ in range(MAX_NEW):
        logits = model(torch.tensor(ids, dtype=torch.long))
        next_tok = int(logits[-1].argmax().item())
        generated.append(next_tok)
        ids.append(next_tok)

print(f"prompt:    {PROMPT}")
print(f"generated: {generated}")
print(f"full ids:  {ids}")

with open("expected_tokens.bin", "wb") as f:
    for tok in generated:
        f.write(struct.pack("<i", tok))
