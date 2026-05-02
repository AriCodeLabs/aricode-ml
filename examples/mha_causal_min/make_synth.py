"""Synthetic causal multi-head attention checkpoint.

seq=4, d_model=16, n_heads=4 (so d_head=4 per head), causal=1.
HF-style key naming (q_proj/k_proj/v_proj/out_proj).

This is the Track A v0.28 milestone test: validate that the existing
causal-mask code path (slot 19 of attn descriptor → triangular -INF
fill in attention_forward_f32) matches PyTorch's
F.scaled_dot_product_attention(..., is_causal=True) within f32 noise.

torch.manual_seed(31).
"""

import math
import torch

torch.manual_seed(31)

seq, d_model, n_heads = 4, 16, 4
d_head = d_model // n_heads

W_Q = torch.randn(d_model, d_model)
b_Q = torch.randn(d_model)
W_K = torch.randn(d_model, d_model)
b_K = torch.randn(d_model)
W_V = torch.randn(d_model, d_model)
b_V = torch.randn(d_model)
W_O = torch.randn(d_model, d_model)
b_O = torch.randn(d_model)

X = torch.randn(seq, d_model)

# Reference forward — match the algorithm we implement: per-head SDPA
# with a causal triangular mask, concat, output project.
Q = X @ W_Q.t() + b_Q   # [seq, d_model]
K = X @ W_K.t() + b_K
V = X @ W_V.t() + b_V

Q_h = Q.view(seq, n_heads, d_head)
K_h = K.view(seq, n_heads, d_head)
V_h = V.view(seq, n_heads, d_head)

# Causal mask: scores[i, j] = -inf when j > i. Same convention as
# attention_f32.ari's slot-19 triangular mask.
mask = torch.full((seq, seq), float("-inf"))
mask = torch.triu(mask, diagonal=1)

head_outs = []
for h in range(n_heads):
    Qh = Q_h[:, h, :]
    Kh = K_h[:, h, :]
    Vh = V_h[:, h, :]
    scores = Qh @ Kh.t() / math.sqrt(d_head)
    scores = scores + mask
    soft = torch.softmax(scores, dim=-1)
    head_outs.append(soft @ Vh)
concat = torch.cat(head_outs, dim=-1)

out = concat @ W_O.t() + b_O

sd = {
    "q_proj.weight": W_Q, "q_proj.bias": b_Q,
    "k_proj.weight": W_K, "k_proj.bias": b_K,
    "v_proj.weight": W_V, "v_proj.bias": b_V,
    "out_proj.weight": W_O, "out_proj.bias": b_O,
}
torch.save(sd, "synth.pt")
X.detach().numpy().tofile("X_input.f32")
out.detach().numpy().tofile("expected.f32")

print(f"seq={seq} d_model={d_model} n_heads={n_heads} d_head={d_head} causal=1")
print(f"output[0, :4] = {out[0, :4].tolist()}")
print(f"output[3, :4] = {out[3, :4].tolist()}")
