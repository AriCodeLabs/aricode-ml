"""Synthetic residual-FFN block: x' = x + FFN(x).

Tests the save_residual / add_residual arch entries.  The arch:

    Linear(8→8) → save_residual → Linear(8→8) → GELU → Linear(8→8) → add_residual

is the standard Pre-LN-style FFN sub-block (without the LN, for
isolation of the residual logic).  Output should equal the
PyTorch reference: input → projection → (residual + ffn(residual)).

torch.manual_seed(13).
"""

import torch

torch.manual_seed(13)
d = 8

W0 = torch.randn(d, d)
b0 = torch.randn(d)
W1 = torch.randn(d, d)
b1 = torch.randn(d)
W2 = torch.randn(d, d)
b2 = torch.randn(d)

X = torch.randn(d)

gelu = torch.nn.GELU(approximate='tanh')

# Forward: linear0 → save → linear1 → gelu → linear2 → add(saved)
y = X @ W0.t() + b0       # initial projection
saved = y
y = saved @ W1.t() + b1
y = gelu(y)
y = y @ W2.t() + b2
out = y + saved

sd = {
    "fc1.weight": W0, "fc1.bias": b0,
    "fc2.weight": W1, "fc2.bias": b1,
    "fc3.weight": W2, "fc3.bias": b2,
}
torch.save(sd, "synth.pt")
X.detach().numpy().tofile("X_input.f32")
out.detach().numpy().tofile("expected.f32")

print(f"d={d}")
print(f"output[:4] = {out[:4].tolist()}")
