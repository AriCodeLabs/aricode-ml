"""tiny_decoder_packed — same architecture as tiny_decoder_min, but
all weights packed via aricode_pack.py instead of hand-emitted.

Saves a single state_dict (synth.pt) using HuggingFace-friendly key
names so pack.py's resolvers find every tensor:
  embed.weight                              → embedding
  ln_0.weight / ln_0.bias                   → pre-attn LayerNorm
  q_proj.weight  / q_proj.bias              → MH-attn Q projection
  k_proj.weight  / k_proj.bias              → MH-attn K projection
  v_proj.weight  / v_proj.bias              → MH-attn V projection
  out_proj.weight / out_proj.bias           → MH-attn output projection
  ln_1.weight / ln_1.bias                   → pre-FFN LayerNorm
  fc1.weight / fc1.bias                     → FFN1 (d_model → ff)
  fc2.weight / fc2.bias                     → FFN2 (ff → d_model)
  ln_2.weight / ln_2.bias                   → final LayerNorm
  fc3.weight / fc3.bias                     → LM head (d_model → vocab)

prompt.bin: 1 byte per token (vocab=16 ≤ 256 → 1 byte/token).
expected_tokens.bin: greedy-decode reference for compare in run_test.sh.
"""

import math
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7)

VOCAB    = 16
D_MODEL  = 8
N_HEADS  = 2
D_HEAD   = D_MODEL // N_HEADS
MAX_SEQ  = 8
FF       = 16
PROMPT   = [1, 5]
MAX_NEW  = 3


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.ln_0  = nn.LayerNorm(D_MODEL, eps=1e-5)
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL)
        self.ln_1  = nn.LayerNorm(D_MODEL, eps=1e-5)
        self.fc1   = nn.Linear(D_MODEL, FF)
        self.fc2   = nn.Linear(FF, D_MODEL)
        self.ln_2  = nn.LayerNorm(D_MODEL, eps=1e-5)
        self.fc3   = nn.Linear(D_MODEL, VOCAB)

    def forward(self, ids):
        x = self.embed(ids)
        h = self.ln_0(x)
        Q = self.q_proj(h).view(-1, N_HEADS, D_HEAD)
        K = self.k_proj(h).view(-1, N_HEADS, D_HEAD)
        V = self.v_proj(h).view(-1, N_HEADS, D_HEAD)
        seq = ids.shape[0]
        mask = torch.full((seq, seq), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        outs = []
        for h_idx in range(N_HEADS):
            scores = Q[:, h_idx] @ K[:, h_idx].t() / math.sqrt(D_HEAD)
            scores = scores + mask
            soft = F.softmax(scores, dim=-1)
            outs.append(soft @ V[:, h_idx])
        concat = torch.cat(outs, dim=-1)
        x = x + self.out_proj(concat)
        h2 = self.ln_1(x)
        h2 = self.fc1(h2)
        h2 = F.gelu(h2, approximate="tanh")
        h2 = self.fc2(h2)
        x = x + h2
        x = self.ln_2(x)
        return self.fc3(x)


model = TinyDecoder()
model.eval()
torch.save(model.state_dict(), "synth.pt")

with open("prompt.bin", "wb") as f:
    for tok in PROMPT:
        f.write(struct.pack("<B", tok))   # vocab=16 → 1 byte/token

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
print(f"sd keys:   {sorted(model.state_dict().keys())}")

with open("expected_tokens.bin", "wb") as f:
    for tok in generated:
        f.write(struct.pack("<i", tok))
