"""Build a synthetic single-head attention checkpoint + reference output.

Used by the regression test that exercises aricode-pack's attention
wire-up.  Determinism: torch.manual_seed(42) on every run, so the
.f32 reference is reproducible bit-for-bit across runs of this
script (and across any aricode-pack changes that don't actually
alter the kernel math).

Usage:
    python make_synth.py
    # writes synth.pt, X_input.f32, expected.f32

Then pack and run:
    python ../../tools/aricode_pack.py \
        --checkpoint synth.pt --arch arch.json \
        --input-format embedded --input-file "$(pwd)/X_input.f32" \
        --embed --no-argmax --out attn_min
    aric attn_min.ari -o attn_min
    ./attn_min  # prints 64 f32 outputs; compare against expected.f32
"""

import math
import torch

torch.manual_seed(42)

seq, d_in, d_head = 4, 8, 16

W_Q = torch.randn(d_head, d_in)
b_Q = torch.randn(d_head)
W_K = torch.randn(d_head, d_in)
b_K = torch.randn(d_head)
W_V = torch.randn(d_head, d_in)
b_V = torch.randn(d_head)

X = torch.randn(seq, d_in)

# Reference forward — same algebra as attention_f32.ari's
# attention_forward_f32 (project -> scores+softmax -> combine).
Q = X @ W_Q.t() + b_Q
K = X @ W_K.t() + b_K
V = X @ W_V.t() + b_V
scores = Q @ K.t() / math.sqrt(d_head)
soft = torch.softmax(scores, dim=-1)
out = soft @ V

# HuggingFace-style key names — the packer's state_dict resolver
# accepts q_proj.weight / k_proj.weight / v_proj.weight directly.
sd = {
    "q_proj.weight": W_Q, "q_proj.bias": b_Q,
    "k_proj.weight": W_K, "k_proj.bias": b_K,
    "v_proj.weight": W_V, "v_proj.bias": b_V,
}
torch.save(sd, "synth.pt")
X.numpy().tofile("X_input.f32")
out.numpy().tofile("expected.f32")

print(f"seq={seq} d_in={d_in} d_head={d_head}")
print(f"input shape:  {tuple(X.shape)}  (X_input.f32, {X.numel() * 4} bytes)")
print(f"output shape: {tuple(out.shape)}  (expected.f32, {out.numel() * 4} bytes)")
print(f"output[0, :4] = {out[0, :4].tolist()}")
