"""Synthetic single-head Pre-LN transformer encoder block.

End-to-end integration test for v0.13 (attention) + v0.14 (LayerNorm) +
v0.15 (GELU) + v0.16 (residuals).  Arch:

    save_residual                ← snapshot input
    layernorm(16)                ← pre-LN
    attention(seq=4, d_in=16, d_head=16, causal=0)
    add_residual                 ← attention sub-block residual
    save_residual                ← snapshot post-attention
    layernorm(16)                ← pre-LN for FFN
    linear(16 → 32) → GELU → linear(32 → 16)
    add_residual                 ← FFN sub-block residual

This is the standard modern (Pre-LN) transformer-encoder-block
forward pass, single-head, no dropout (we don't pack stochastic
ops).  d_in and d_head match because residuals require the
attention output to fold back into the input shape — true
multi-head attention with a separate output projection is the
next roadmap item.

torch.manual_seed(17).
"""

import math
import torch

torch.manual_seed(17)

seq, d_model, d_ff = 4, 16, 32

# Attention weights
W_Q = torch.randn(d_model, d_model)
b_Q = torch.randn(d_model)
W_K = torch.randn(d_model, d_model)
b_K = torch.randn(d_model)
W_V = torch.randn(d_model, d_model)
b_V = torch.randn(d_model)

# LayerNorm 1 (pre-attention)
ln1_g = torch.randn(d_model)
ln1_b = torch.randn(d_model)

# LayerNorm 2 (pre-FFN)
ln2_g = torch.randn(d_model)
ln2_b = torch.randn(d_model)

# FFN weights
W_ff1 = torch.randn(d_ff, d_model)
b_ff1 = torch.randn(d_ff)
W_ff2 = torch.randn(d_model, d_ff)
b_ff2 = torch.randn(d_model)

X = torch.randn(seq, d_model)

# Reference forward — exactly mirrors the arch JSON above.
ln1 = torch.nn.LayerNorm(d_model, eps=1e-5)
ln1.weight.data = ln1_g.clone()
ln1.bias.data = ln1_b.clone()
ln2 = torch.nn.LayerNorm(d_model, eps=1e-5)
ln2.weight.data = ln2_g.clone()
ln2.bias.data = ln2_b.clone()
gelu = torch.nn.GELU(approximate='tanh')

# 1. attention sub-block
saved = X.clone()
y = ln1(X)
Q = y @ W_Q.t() + b_Q
K = y @ W_K.t() + b_K
V = y @ W_V.t() + b_V
scores = Q @ K.t() / math.sqrt(d_model)
y = torch.softmax(scores, dim=-1) @ V
y = y + saved

# 2. FFN sub-block
saved = y.clone()
y = ln2(y)
y = y @ W_ff1.t() + b_ff1
y = gelu(y)
y = y @ W_ff2.t() + b_ff2
out = y + saved

# Pack expects per-call inference; pick the first row.
X_one = X[0]                  # [d_model]
out_one = out[0]              # [d_model]

# Wait — the residuals use ALL seq tokens, so feeding only one
# row defeats the purpose.  Instead, feed the full [seq, d_model]
# = 64-element flat input and read the full [seq, d_model] output.
# The packer's gen_act_decls handles this for attention-first archs;
# with save_residual first, sizes[0] is set by the FIRST layer
# (= save_residual's snapshot, which inherits from… nothing yet).
# The arch's first non-pseudolayer is layernorm(16), so sizes[0] =
# d_model.  But we need seq*d_model.  Use ["attention", ...] as the
# size-defining layer (not the first arch entry — but the first
# size-changing one).
#
# Workaround for this test: prepend a no-op arch entry that sizes
# correctly.  Or just pass d_model elements (seq=1 effective).

X_seq = X                     # [seq, d_model] full
out_seq = out                 # full output
X_seq.detach().numpy().tofile("X_input.f32")
out_seq.detach().numpy().tofile("expected.f32")

sd = {
    # Both LNs go through the layers.{ni}.LayerNorm.weight pattern in
    # the packer's resolver — matches the typical HF stacked-block
    # convention (e.g. distilbert.transformer.layer.{i}.attention.LayerNorm).
    "layers.0.LayerNorm.weight": ln1_g, "layers.0.LayerNorm.bias": ln1_b,
    "layers.1.LayerNorm.weight": ln2_g, "layers.1.LayerNorm.bias": ln2_b,
    "q_proj.weight": W_Q, "q_proj.bias": b_Q,
    "k_proj.weight": W_K, "k_proj.bias": b_K,
    "v_proj.weight": W_V, "v_proj.bias": b_V,
    "fc1.weight": W_ff1, "fc1.bias": b_ff1,
    "fc2.weight": W_ff2, "fc2.bias": b_ff2,
}
torch.save(sd, "synth.pt")

print(f"seq={seq} d_model={d_model} d_ff={d_ff}")
print(f"input shape:  {tuple(X_seq.shape)}  ({X_seq.numel() * 4} bytes)")
print(f"output shape: {tuple(out_seq.shape)}  ({out_seq.numel() * 4} bytes)")
print(f"output[0, :4] = {out_seq[0, :4].tolist()}")
