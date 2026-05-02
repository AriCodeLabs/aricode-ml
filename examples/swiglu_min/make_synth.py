"""Synthetic SwiGLU FFN regression checkpoint.

SwiGLU is the Llama / Mistral / TinyLlama feed-forward block:
   y = down_proj(silu(gate_proj(x)) * up_proj(x))
where silu(z) = z * sigmoid(z) = z / (1 + exp(-z)).

No biases (Llama convention).  Matches the new pack.py
`["swiglu_ffn", d_model, d_ffn]` arch op + the silu_mul_f32 helper +
the gate/up/down state_dict resolver fallbacks.

torch.manual_seed(19) for reproducibility.
"""

import torch
import torch.nn.functional as F

torch.manual_seed(19)

D_MODEL = 16
D_FFN   = 32

gate_W = torch.randn(D_FFN, D_MODEL)
up_W   = torch.randn(D_FFN, D_MODEL)
down_W = torch.randn(D_MODEL, D_FFN)

X = torch.randn(D_MODEL)

# Reference forward.
gate = X @ gate_W.t()
up   = X @ up_W.t()
hidden = F.silu(gate) * up      # element-wise
out = hidden @ down_W.t()       # back to D_MODEL

# Use the swiglu fallback keys so pack.py's resolver finds them
# without needing a Llama-shaped state_dict tree.
sd = {
    "swiglu0.gate.weight": gate_W,
    "swiglu0.up.weight":   up_W,
    "swiglu0.down.weight": down_W,
}
torch.save(sd, "synth.pt")
X.detach().numpy().tofile("X_input.f32")
out.detach().numpy().tofile("expected.f32")

print(f"D_MODEL={D_MODEL} D_FFN={D_FFN}")
print(f"output[:4] = {out[:4].tolist()}")
