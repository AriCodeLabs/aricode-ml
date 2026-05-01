"""End-to-end Pre-LN transformer encoder, single block, every layer.

Comprehensive integration test for every layer in v0.13–v0.21:

    Token-IDs[seq] → Embedding(100, 32) → +PositionalEmbedding(16, 32)
                  → LN → MHA(d=32, h=4) → +residual
                  → LN → FFN(32→64→32 with GELU) → +residual
                  → LN → classifier Linear(32→5)
                  → logits[5]   (we use full output, no argmax)

If aricode-pack assembles all 21 generations of layers correctly,
the packed binary's logits match PyTorch's forward within float
quantisation noise.

torch.manual_seed(41).
"""

import math
import numpy as np
import torch

torch.manual_seed(41)

vocab, max_pos, d_model, seq, d_ff, n_classes = 100, 16, 32, 4, 64, 5
n_heads = 4
d_head = d_model // n_heads

# Embedding tables
tok_emb = torch.randn(vocab, d_model)
pos_emb = torch.randn(max_pos, d_model)

# Block: pre-LN attention
ln_attn_g = torch.randn(d_model); ln_attn_b = torch.randn(d_model)
W_Q = torch.randn(d_model, d_model); b_Q = torch.randn(d_model)
W_K = torch.randn(d_model, d_model); b_K = torch.randn(d_model)
W_V = torch.randn(d_model, d_model); b_V = torch.randn(d_model)
W_O = torch.randn(d_model, d_model); b_O = torch.randn(d_model)

# Block: pre-LN FFN
ln_ffn_g = torch.randn(d_model); ln_ffn_b = torch.randn(d_model)
W_ff1 = torch.randn(d_ff, d_model); b_ff1 = torch.randn(d_ff)
W_ff2 = torch.randn(d_model, d_ff); b_ff2 = torch.randn(d_model)

# Final LN + classifier
ln_final_g = torch.randn(d_model); ln_final_b = torch.randn(d_model)
W_cls = torch.randn(n_classes, d_model); b_cls = torch.randn(n_classes)

# Input
token_ids = [7, 13, 42, 99]
gelu = torch.nn.GELU(approximate='tanh')

def lnorm(x, g, b, eps=1e-5):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight=g, bias=b, eps=eps)

# Forward — exactly mirrors arch.json
x = tok_emb[token_ids]               # [seq, d_model]
x = x + pos_emb[:seq]                # +positional

# Pre-LN attention sub-block
saved = x.clone()
y = lnorm(x, ln_attn_g, ln_attn_b)
Q = y @ W_Q.t() + b_Q
K = y @ W_K.t() + b_K
V = y @ W_V.t() + b_V
Q_h = Q.view(seq, n_heads, d_head)
K_h = K.view(seq, n_heads, d_head)
V_h = V.view(seq, n_heads, d_head)
heads = []
for h in range(n_heads):
    scores = Q_h[:, h] @ K_h[:, h].t() / math.sqrt(d_head)
    soft = torch.softmax(scores, dim=-1)
    heads.append(soft @ V_h[:, h])
concat = torch.cat(heads, dim=-1)
y = concat @ W_O.t() + b_O
y = y + saved

# Pre-LN FFN sub-block
saved = y.clone()
y = lnorm(y, ln_ffn_g, ln_ffn_b)
y = y @ W_ff1.t() + b_ff1
y = gelu(y)
y = y @ W_ff2.t() + b_ff2
y = y + saved

# Final LN + classifier head
y = lnorm(y, ln_final_g, ln_final_b)
# classifier produces [seq, n_classes]; the packer's batched-Linear
# loop applies it per row, so the output is [seq * n_classes] flat.
out = y @ W_cls.t() + b_cls
out_flat = out.view(-1)

# State dict: HF-style nested where the per-layer counters line up
# with the packer's generators.
sd = {
    "embeddings.word_embeddings.weight":     tok_emb,
    "embeddings.position_embeddings.weight": pos_emb,

    # LN counter starts at 0 (attention pre-LN)
    "layers.0.LayerNorm.weight":             ln_attn_g,
    "layers.0.LayerNorm.bias":               ln_attn_b,

    # MHA — only 1 multi-head, plain prefixes
    "q_proj.weight":   W_Q, "q_proj.bias":   b_Q,
    "k_proj.weight":   W_K, "k_proj.bias":   b_K,
    "v_proj.weight":   W_V, "v_proj.bias":   b_V,
    "out_proj.weight": W_O, "out_proj.bias": b_O,

    # LN counter 1 (FFN pre-LN)
    "layers.1.LayerNorm.weight":             ln_ffn_g,
    "layers.1.LayerNorm.bias":               ln_ffn_b,

    # FFN linears (default --keys is fc{idx_plus_1}.{kind})
    "fc1.weight": W_ff1, "fc1.bias": b_ff1,
    "fc2.weight": W_ff2, "fc2.bias": b_ff2,

    # LN counter 2 (final LN)
    "layers.2.LayerNorm.weight":             ln_final_g,
    "layers.2.LayerNorm.bias":               ln_final_b,

    # Classifier
    "fc3.weight": W_cls, "fc3.bias": b_cls,
}
torch.save(sd, "synth.pt")

np.array(token_ids, dtype=np.uint8).tofile("tokens.bin")
out_flat.detach().numpy().tofile("expected.f32")

print(f"vocab={vocab} d_model={d_model} d_ff={d_ff} n_heads={n_heads} "
      f"n_classes={n_classes} seq={seq}")
print(f"tokens={token_ids}")
print(f"output[:5] (logits row 0) = {out[0].tolist()}")
print(f"flat output shape = {tuple(out_flat.shape)}")
