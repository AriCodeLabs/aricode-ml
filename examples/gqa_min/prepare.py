"""gqa_min — sanity check that the GQA-KV kernel degrades to standard
MHA + RoPE when n_kv_heads == n_heads.

Same rig as tiny_llama_min but with no SwiGLU FFN, no post-attn
RMSNorm, and n_kv_heads = n_heads = 4.  Validates:
  - attn_gqa_kv_step_f32 produces correct outputs in the GQA = MHA
    boundary case (group_size = 1).
  - RoPE applies identically across all heads.
  - Embedding + RMSNorm + GQA + RMSNorm + LM head wiring.
"""

import math
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(53)

VOCAB        = 32
D_MODEL      = 16
N_HEADS      = 4
N_KV_HEADS   = 4
D_HEAD       = D_MODEL // N_HEADS
GROUP_SIZE   = N_HEADS // N_KV_HEADS    # 1
KV_DIM       = N_KV_HEADS * D_HEAD       # = D_MODEL
MAX_SEQ      = 8
ROPE_THETA   = 10000.0
PROMPT       = [2, 3, 11]
MAX_NEW      = 4
EPS          = 1e-5


def build_rope_table(max_seq, d_head, theta):
    table = torch.zeros(max_seq, d_head, dtype=torch.float32)
    log_theta = math.log(theta)
    half = d_head // 2
    for pos in range(max_seq):
        for i in range(half):
            exponent = -2.0 * i / d_head
            freq = math.exp(exponent * log_theta)
            angle = pos * freq
            table[pos, 2 * i]     = math.cos(angle)
            table[pos, 2 * i + 1] = math.sin(angle)
    return table


ROPE_TABLE = build_rope_table(MAX_SEQ, D_HEAD, ROPE_THETA)


def apply_rope_torch(vec, pos, table):
    d_head = vec.shape[-1]
    half = d_head // 2
    out = vec.clone()
    for i in range(half):
        c = table[pos, 2 * i].item()
        s = table[pos, 2 * i + 1].item()
        x0 = vec[2 * i].item()
        x1 = vec[2 * i + 1].item()
        out[2 * i]     = x0 * c - x1 * s
        out[2 * i + 1] = x0 * s + x1 * c
    return out


def rmsnorm(x, gamma, eps=EPS):
    ms = (x * x).mean(dim=-1, keepdim=True)
    return x * gamma / torch.sqrt(ms + eps)


class GqaMin(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.gamma_in = nn.Parameter(torch.randn(D_MODEL))
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_proj = nn.Linear(D_MODEL, KV_DIM,  bias=False)
        self.v_proj = nn.Linear(D_MODEL, KV_DIM,  bias=False)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.gamma_final = nn.Parameter(torch.randn(D_MODEL))
        self.lm_head = nn.Linear(D_MODEL, VOCAB)

    def forward(self, ids):
        seq = ids.shape[0]
        x = self.embed(ids)
        Kc = torch.zeros(MAX_SEQ, N_KV_HEADS, D_HEAD)
        Vc = torch.zeros(MAX_SEQ, N_KV_HEADS, D_HEAD)
        outs = []
        for t in range(seq):
            x_t = x[t]
            h = rmsnorm(x_t, self.gamma_in)
            Q = self.q_proj(h).view(N_HEADS, D_HEAD)
            K = self.k_proj(h).view(N_KV_HEADS, D_HEAD)
            V = self.v_proj(h).view(N_KV_HEADS, D_HEAD)
            for hh in range(N_HEADS):
                Q[hh] = apply_rope_torch(Q[hh], t, ROPE_TABLE)
            for kh in range(N_KV_HEADS):
                K[kh] = apply_rope_torch(K[kh], t, ROPE_TABLE)
            Kc[t] = K
            Vc[t] = V
            head_outs = []
            inv_sqrt = 1.0 / math.sqrt(D_HEAD)
            for hh in range(N_HEADS):
                kv_h = hh // GROUP_SIZE
                scores = (Kc[: t + 1, kv_h] @ Q[hh]) * inv_sqrt
                soft = F.softmax(scores, dim=-1)
                head_outs.append(soft @ Vc[: t + 1, kv_h])
            concat = torch.cat(head_outs, dim=-1)
            attn_out = self.o_proj(concat)
            x_t = x_t + attn_out
            outs.append(x_t)
        h_final = torch.stack(outs, dim=0)
        h_final = rmsnorm(h_final, self.gamma_final)
        return self.lm_head(h_final)


model = GqaMin()
model.eval()

sd_native = model.state_dict()
sd = {
    "embed.weight":                                   sd_native["embed.weight"],
    "model.layers.0.input_layernorm.weight":          sd_native["gamma_in"],
    "model.layers.0.self_attn.q_proj.weight":         sd_native["q_proj.weight"],
    "model.layers.0.self_attn.k_proj.weight":         sd_native["k_proj.weight"],
    "model.layers.0.self_attn.v_proj.weight":         sd_native["v_proj.weight"],
    "model.layers.0.self_attn.o_proj.weight":         sd_native["o_proj.weight"],
    "model.norm.weight":                              sd_native["gamma_final"],
    "fc1.weight":                                     sd_native["lm_head.weight"],
    "fc1.bias":                                       sd_native["lm_head.bias"],
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
print(f"n params:  {sum(p.numel() for p in model.parameters())}")

with open("expected_tokens.bin", "wb") as f:
    for tok in generated:
        f.write(struct.pack("<i", tok))
