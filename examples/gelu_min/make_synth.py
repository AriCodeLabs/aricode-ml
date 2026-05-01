"""Synthetic Linear → GELU → Linear (a tiny FFN block).

Exercises the GELU activation arch entry: aricode-pack emits the
gelu_f32 user-fn helper (tanh approximation, matching
torch.nn.GELU(approximate='tanh')) between the two Linears.

torch.manual_seed(11) for reproducibility.
"""

import torch

torch.manual_seed(11)

d_in = 8
d_ff = 16
d_out = 4

W1 = torch.randn(d_ff, d_in)
b1 = torch.randn(d_ff)
W2 = torch.randn(d_out, d_ff)
b2 = torch.randn(d_out)

X = torch.randn(d_in)

# Reference: tanh-approx GELU (matches the user-fn helper)
gelu = torch.nn.GELU(approximate='tanh')
y = X @ W1.t() + b1
y = gelu(y)
out = y @ W2.t() + b2

sd = {
    "fc1.weight": W1, "fc1.bias": b1,
    "fc2.weight": W2, "fc2.bias": b2,
}
torch.save(sd, "synth.pt")
X.detach().numpy().tofile("X_input.f32")
out.detach().numpy().tofile("expected.f32")

print(f"d_in={d_in} d_ff={d_ff} d_out={d_out}")
print(f"output[:4] = {out[:4].tolist()}")
