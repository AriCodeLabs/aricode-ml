"""Synthetic LayerNorm + Linear regression checkpoint.

Tests the LayerNorm pack-side wiring: γ and β are loaded from the
state_dict, applied affinely after arr_f32_layernorm normalises the
input.  The Linear that follows verifies the output buffer is
correctly threaded through the in-place LayerNorm.

torch.manual_seed(7) for reproducibility.
"""

import torch

torch.manual_seed(7)

K = 4   # batch / seq, treated as just rows in the (K, dim) input
dim = 16
out_features = 8

# LayerNorm with affine
gamma = torch.randn(dim)
beta = torch.randn(dim)

# Final linear head
W = torch.randn(out_features, dim)
b = torch.randn(out_features)

# Fixed input
X = torch.randn(K, dim)

# Reference: LayerNorm(X) → Linear → output flat
ln = torch.nn.LayerNorm(dim, eps=1e-5)
ln.weight.data = gamma.clone()
ln.bias.data = beta.clone()
y = ln(X)              # [K, dim]
out = y @ W.t() + b    # [K, out_features]

# We pack as a single forward producing one [out_features] vector,
# so to match we'll feed a single row at a time.  But the packer
# assumes one forward per call — so pick the first row as the test
# input and reference to that row's output.
X_one = X[0]                        # [dim]
y_one = ln(X_one.unsqueeze(0))      # [1, dim]
out_one = y_one @ W.t() + b         # [1, out_features]

sd = {
    "LayerNorm.weight": gamma,
    "LayerNorm.bias":   beta,
    "fc1.weight":       W,
    "fc1.bias":         b,
}
torch.save(sd, "synth.pt")
X_one.detach().numpy().tofile("X_input.f32")
out_one.squeeze(0).detach().numpy().tofile("expected.f32")

print(f"K={K} dim={dim} out_features={out_features}")
print(f"input shape:  {tuple(X_one.shape)}  ({X_one.numel() * 4} bytes)")
print(f"output shape: {tuple(out_one.squeeze(0).shape)}  "
      f"({out_one.squeeze(0).numel() * 4} bytes)")
print(f"output[:4] = {out_one.squeeze(0)[:4].tolist()}")
