"""Synthetic RMSNorm + Linear regression checkpoint.

Locks in pack.py's RMSNorm pipeline:
- `["rmsnorm", dim]` arch op resolves to the new rmsnorm_affine_f32
  helper inlined into the prologue when needs_rmsnorm_helper(arch).
- State_dict key `rmsnorm0.weight` (a fallback name in pack.py's
  rmsnorm resolver) feeds the γ tensor.

torch.manual_seed(13) for reproducibility.
"""

import torch
import torch.nn.functional as F

torch.manual_seed(13)

dim = 16
out_features = 8

# RMSNorm γ (no β; Llama convention).
gamma = torch.randn(dim)

# Final linear head.
W = torch.randn(out_features, dim)
b = torch.randn(out_features)

# Reference: RMSNorm(x) = x * γ / sqrt(mean(x²) + eps), eps=1e-5.
def rmsnorm(x, gamma, eps=1e-5):
    ms = (x * x).mean(dim=-1, keepdim=True)
    return x * gamma / torch.sqrt(ms + eps)

X = torch.randn(dim)                          # [dim]
y = rmsnorm(X, gamma)                         # [dim]
out = y @ W.t() + b                           # [out_features]

sd = {
    "rmsnorm0.weight": gamma,                 # fallback key in pack.py resolver
    "fc1.weight":      W,
    "fc1.bias":        b,
}
torch.save(sd, "synth.pt")
X.detach().numpy().tofile("X_input.f32")
out.detach().numpy().tofile("expected.f32")

print(f"dim={dim} out_features={out_features}")
print(f"input bytes:  {X.numel() * 4}")
print(f"output[:4] = {out[:4].tolist()}")
