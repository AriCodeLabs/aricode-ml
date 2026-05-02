"""tiny_decoder_min — synthetic 1-block GPT-style decoder.

Architecture (all f32, all PyTorch reference):
  vocab=16, d_model=8, n_heads=2 (d_head=4), max_seq=8, ff=16
  embedding(16, 8)
  block:
    pre-LN(8) → MH-attn(8, 2 heads, causal) → +residual
    pre-LN(8) → FFN(8 → 16 → 8, GELU)        → +residual
  final LN(8)
  LM head: Linear(8 → 16)

Greedy autoregressive decode:
  prompt = [1, 5]   (seq_p=2)
  max_new = 3
  produces 3 token IDs by feeding the previous step's argmax back in.

Saves every weight tensor as a raw f32 file (one .f32 per name) so
the .ari side can embed_file each by name.  Saves expected_tokens.bin
(3 i32 LE) for the validator.
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

# ---------- model definition (matches the .ari decoder layout) ----------

class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.ln1   = nn.LayerNorm(D_MODEL, eps=1e-5)
        self.q     = nn.Linear(D_MODEL, D_MODEL)
        self.k     = nn.Linear(D_MODEL, D_MODEL)
        self.v     = nn.Linear(D_MODEL, D_MODEL)
        self.o     = nn.Linear(D_MODEL, D_MODEL)
        self.ln2   = nn.LayerNorm(D_MODEL, eps=1e-5)
        self.ff1   = nn.Linear(D_MODEL, FF)
        self.ff2   = nn.Linear(FF, D_MODEL)
        self.ln_f  = nn.LayerNorm(D_MODEL, eps=1e-5)
        self.lm    = nn.Linear(D_MODEL, VOCAB)

    def forward(self, ids):
        x = self.embed(ids)                           # [seq, d_model]
        # Pre-LN attention block
        h = self.ln1(x)
        Q = self.q(h).view(-1, N_HEADS, D_HEAD)       # [seq, n_heads, d_head]
        K = self.k(h).view(-1, N_HEADS, D_HEAD)
        V = self.v(h).view(-1, N_HEADS, D_HEAD)
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
        attn_out = self.o(concat)
        x = x + attn_out
        # Pre-LN FFN block (GELU tanh-approx to match aricode-ml's GELU)
        h2 = self.ln2(x)
        h2 = self.ff1(h2)
        h2 = F.gelu(h2, approximate="tanh")
        h2 = self.ff2(h2)
        x = x + h2
        # Final LN + LM head
        x = self.ln_f(x)
        logits = self.lm(x)                           # [seq, vocab]
        return logits


model = TinyDecoder()
model.eval()

# ---------- save tensors ----------
# Naming convention: `<name>.f32` raw float32 little-endian.

def save(name, t):
    arr = t.detach().contiguous().to(torch.float32).numpy()
    arr.tofile(f"{name}.f32")

save("embed_W", model.embed.weight)
save("ln1_gamma", model.ln1.weight); save("ln1_beta", model.ln1.bias)
save("q_W", model.q.weight); save("q_b", model.q.bias)
save("k_W", model.k.weight); save("k_b", model.k.bias)
save("v_W", model.v.weight); save("v_b", model.v.bias)
save("o_W", model.o.weight); save("o_b", model.o.bias)
save("ln2_gamma", model.ln2.weight); save("ln2_beta", model.ln2.bias)
save("ff1_W", model.ff1.weight); save("ff1_b", model.ff1.bias)
save("ff2_W", model.ff2.weight); save("ff2_b", model.ff2.bias)
save("lnf_gamma", model.ln_f.weight); save("lnf_beta", model.ln_f.bias)
save("lm_W", model.lm.weight); save("lm_b", model.lm.bias)

# ---------- prompt as raw uint16 LE token ids (room for vocab > 255) ----------

with open("prompt.bin", "wb") as f:
    for tok in PROMPT:
        f.write(struct.pack("<H", tok))

# ---------- greedy autoregressive decode reference ----------

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
