"""Synthetic multi-head attention checkpoint.

seq=4, d_model=16, n_heads=4 (so d_head=4 per head), no causal mask.
HF-style key naming (q_proj/k_proj/v_proj/out_proj).

torch.manual_seed(23).
"""

import math
import torch

torch.manual_seed(23)

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
# (using d_head as both the score-scale denominator AND the per-head
# feature dim), concat, output project.
Q = X @ W_Q.t() + b_Q   # [seq, d_model]
K = X @ W_K.t() + b_K
V = X @ W_V.t() + b_V

# Reshape to (seq, n_heads, d_head)
Q_h = Q.view(seq, n_heads, d_head)
K_h = K.view(seq, n_heads, d_head)
V_h = V.view(seq, n_heads, d_head)

head_outs = []
for h in range(n_heads):
    Qh = Q_h[:, h, :]   # [seq, d_head]
    Kh = K_h[:, h, :]
    Vh = V_h[:, h, :]
    scores = Qh @ Kh.t() / math.sqrt(d_head)
    soft = torch.softmax(scores, dim=-1)
    head_outs.append(soft @ Vh)   # [seq, d_head]
concat = torch.cat(head_outs, dim=-1)  # [seq, d_model]

out = concat @ W_O.t() + b_O   # [seq, d_model]

sd = {
    "q_proj.weight": W_Q, "q_proj.bias": b_Q,
    "k_proj.weight": W_K, "k_proj.bias": b_K,
    "v_proj.weight": W_V, "v_proj.bias": b_V,
    "out_proj.weight": W_O, "out_proj.bias": b_O,
}
torch.save(sd, "synth.pt")
X.detach().numpy().tofile("X_input.f32")
out.detach().numpy().tofile("expected.f32")

print(f"seq={seq} d_model={d_model} n_heads={n_heads} d_head={d_head}")
print(f"output[0, :4] = {out[0, :4].tolist()}")
