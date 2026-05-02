"""rmsnorm_swiglu_decoder — 1-block decoder using the new pieces:
  - RMSNorm (instead of LayerNorm) for pre-attn / pre-FFN / final norm
  - SwiGLU (instead of GELU + 2× Linear) for the FFN block
  - Existing multi_head_attention_kv (no RoPE/GQA — those land in
    the next session as multi_head_attention_gqa_kv).

Validates that the new RMSNorm + SwiGLU arch ops compose cleanly
with the existing decoder pipeline (decoder loop main, KV cache,
embedding lookup, sampling).  When RoPE+GQA arrive, swapping
multi_head_attention_kv for multi_head_attention_gqa_kv (and
adding the rope theta) is the only structural change to reach
TinyLlama / Llama.

Architecture:
  vocab=16, d_model=8, n_heads=2, max_seq=8, ff=16
  embed → save_resid → rmsnorm → MHA-KV → add_resid
        → save_resid → rmsnorm → swiglu_ffn → add_resid
        → rmsnorm → LM head

State_dict keys use the same Llama HF convention pack.py already
resolves (model.layers.{i}.input_layernorm.weight,
model.layers.{i}.mlp.{gate,up,down}_proj.weight, etc.).

torch.manual_seed(23) for reproducibility.
"""

import math
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(23)

VOCAB    = 16
D_MODEL  = 8
N_HEADS  = 2
D_HEAD   = D_MODEL // N_HEADS
MAX_SEQ  = 8
FF       = 16
PROMPT   = [2, 5]
MAX_NEW  = 3


def rmsnorm(x, gamma, eps=1e-5):
    ms = (x * x).mean(dim=-1, keepdim=True)
    return x * gamma / torch.sqrt(ms + eps)


class TinyLlamaNoRope(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.in_norm   = nn.Parameter(torch.randn(D_MODEL))   # γ pre-attn
        self.q_proj    = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj    = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj    = nn.Linear(D_MODEL, D_MODEL)
        self.out_proj  = nn.Linear(D_MODEL, D_MODEL)
        self.post_norm = nn.Parameter(torch.randn(D_MODEL))   # γ pre-FFN
        self.gate_proj = nn.Linear(D_MODEL, FF, bias=False)
        self.up_proj   = nn.Linear(D_MODEL, FF, bias=False)
        self.down_proj = nn.Linear(FF, D_MODEL, bias=False)
        self.final_norm = nn.Parameter(torch.randn(D_MODEL))  # γ final
        self.lm_head   = nn.Linear(D_MODEL, VOCAB)

    def forward(self, ids):
        x = self.embed(ids)                                   # [seq, D]
        seq = ids.shape[0]
        mask = torch.full((seq, seq), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        # Pre-attn RMSNorm + MHA + residual
        h = rmsnorm(x, self.in_norm)
        Q = self.q_proj(h).view(-1, N_HEADS, D_HEAD)
        K = self.k_proj(h).view(-1, N_HEADS, D_HEAD)
        V = self.v_proj(h).view(-1, N_HEADS, D_HEAD)
        outs = []
        for hh in range(N_HEADS):
            scores = Q[:, hh] @ K[:, hh].t() / math.sqrt(D_HEAD)
            scores = scores + mask
            soft = F.softmax(scores, dim=-1)
            outs.append(soft @ V[:, hh])
        concat = torch.cat(outs, dim=-1)
        x = x + self.out_proj(concat)
        # Pre-FFN RMSNorm + SwiGLU + residual
        h2 = rmsnorm(x, self.post_norm)
        gate = self.gate_proj(h2)
        up   = self.up_proj(h2)
        h2 = F.silu(gate) * up
        h2 = self.down_proj(h2)
        x = x + h2
        # Final RMSNorm + LM head
        x = rmsnorm(x, self.final_norm)
        return self.lm_head(x)


model = TinyLlamaNoRope()
model.eval()

# Re-key into pack.py's expected names.  RMSNorm: ri counts top-down,
# so ri=0 is in_norm, ri=1 is post_norm, ri=2 is final_norm.  pack.py's
# resolver treats ri=0 as block 0 input_layernorm, ri=1 as block 0
# post_attention_layernorm, ri=2 as model.norm (final).
sd = {
    "embed.weight": model.embed.weight,
    "model.layers.0.input_layernorm.weight":          model.in_norm,
    "model.layers.0.post_attention_layernorm.weight": model.post_norm,
    "model.norm.weight":                              model.final_norm,
    "model.layers.0.mlp.gate_proj.weight":            model.gate_proj.weight,
    "model.layers.0.mlp.up_proj.weight":              model.up_proj.weight,
    "model.layers.0.mlp.down_proj.weight":            model.down_proj.weight,
    # MHA-KV uses the same q_proj/k_proj/v_proj/out_proj convention:
    "q_proj.weight": model.q_proj.weight, "q_proj.bias": model.q_proj.bias,
    "k_proj.weight": model.k_proj.weight, "k_proj.bias": model.k_proj.bias,
    "v_proj.weight": model.v_proj.weight, "v_proj.bias": model.v_proj.bias,
    "out_proj.weight": model.out_proj.weight, "out_proj.bias": model.out_proj.bias,
    # LM head — pack.py uses --keys "fc{idx_plus_1}.{kind}" for Linears.
    # Our only Linear is the LM head at li=0 → fc1.
    "fc1.weight": model.lm_head.weight,
    "fc1.bias":   model.lm_head.bias,
}
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
