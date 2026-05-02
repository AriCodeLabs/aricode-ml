"""tiny_llama_min — synthetic 1-block Llama-style decoder.

Architecture (matching arch.json):
- vocab=32, d_model=16, n_heads=4, n_kv_heads=2, d_ffn=32, max_seq=8.
- Embedding (token).
- Block 0: RMSNorm → GQA-KV with RoPE (theta=10000) → residual
           → RMSNorm → SwiGLU FFN → residual.
- Final RMSNorm → LM head.

State_dict uses HuggingFace Llama key conventions so pack.py's
resolvers find every tensor:
  model.embed_tokens.weight                        (we use the
                                                   `embed.weight`
                                                   fallback alias since
                                                   pack.py's embedding
                                                   resolver checks both)
  model.layers.0.input_layernorm.weight            pre-attn RMSNorm
  model.layers.0.self_attn.{q,k,v,o}_proj.weight   GQA projections
  model.layers.0.post_attention_layernorm.weight   pre-FFN RMSNorm
  model.layers.0.mlp.{gate,up,down}_proj.weight    SwiGLU FFN
  model.norm.weight                                final RMSNorm
  lm_head.weight + lm_head.bias                    LM head linear

The PyTorch reference applies RoPE the same way rope_apply_f32 does
(INTERLEAVED pairs convention) — see apply_rope_torch() below — so the
greedy token sequence is reproducible across Python and aricode.

torch.manual_seed(31) for reproducibility.
"""

import math
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(31)

VOCAB        = 32
D_MODEL      = 16
N_HEADS      = 4
N_KV_HEADS   = 2
D_HEAD       = D_MODEL // N_HEADS
GROUP_SIZE   = N_HEADS // N_KV_HEADS
KV_DIM       = N_KV_HEADS * D_HEAD
D_FFN        = 32
MAX_SEQ      = 8
ROPE_THETA   = 10000.0
PROMPT       = [1, 5, 7]
MAX_NEW      = 4
EPS          = 1e-5


def build_rope_table(max_seq, d_head, theta):
    """Build a [max_seq, d_head] interleaved (cos, sin) table that
    EXACTLY mirrors rope_alloc_f32 in rope_f32.ari (same exp/log path
    so even fp64 rounding details line up — important for
    token-for-token agreement)."""
    table = torch.zeros(max_seq, d_head, dtype=torch.float32)
    log_theta = math.log(theta)
    half = d_head // 2
    for pos in range(max_seq):
        for i in range(half):
            exponent = -2.0 * i / d_head
            freq = math.exp(exponent * log_theta)
            angle = pos * freq
            table[pos, 2 * i]     = math.cos(angle)
            table[pos, 2 * i + 1] = math.sin(angle)
    return table


ROPE_TABLE = build_rope_table(MAX_SEQ, D_HEAD, ROPE_THETA)


def apply_rope_torch(vec, pos, table):
    """In-place RoPE rotation matching rope_apply_f32 in rope_f32.ari.
    `vec` is [d_head]; pairs are (vec[2i], vec[2i+1]).  Returns a new
    tensor (PyTorch tensors are functional)."""
    d_head = vec.shape[-1]
    half = d_head // 2
    out = vec.clone()
    for i in range(half):
        c = table[pos, 2 * i].item()
        s = table[pos, 2 * i + 1].item()
        x0 = vec[2 * i].item()
        x1 = vec[2 * i + 1].item()
        out[2 * i]     = x0 * c - x1 * s
        out[2 * i + 1] = x0 * s + x1 * c
    return out


def rmsnorm(x, gamma, eps=EPS):
    ms = (x * x).mean(dim=-1, keepdim=True)
    return x * gamma / torch.sqrt(ms + eps)


class TinyLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.gamma_in   = nn.Parameter(torch.randn(D_MODEL))
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_proj = nn.Linear(D_MODEL, KV_DIM, bias=False)
        self.v_proj = nn.Linear(D_MODEL, KV_DIM, bias=False)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.gamma_post = nn.Parameter(torch.randn(D_MODEL))
        self.gate_proj = nn.Linear(D_MODEL, D_FFN, bias=False)
        self.up_proj   = nn.Linear(D_MODEL, D_FFN, bias=False)
        self.down_proj = nn.Linear(D_FFN, D_MODEL, bias=False)
        self.gamma_final = nn.Parameter(torch.randn(D_MODEL))
        self.lm_head = nn.Linear(D_MODEL, VOCAB)   # has bias — pack
                                                   # auto-loads bias for
                                                   # `linear`.

    def forward(self, ids):
        # ids: [seq] (long tensor).  We mirror the .ari decode loop:
        # process tokens one at a time so the KV cache and RoPE
        # rotations sequence exactly like the runtime.
        seq = ids.shape[0]
        x = self.embed(ids)                        # [seq, d_model]
        # Per-step caches (rotated K, raw V).
        Kc = torch.zeros(MAX_SEQ, N_KV_HEADS, D_HEAD)
        Vc = torch.zeros(MAX_SEQ, N_KV_HEADS, D_HEAD)
        outs = []
        for t in range(seq):
            x_t = x[t]                              # [d_model]
            # Pre-attn RMSNorm.
            h = rmsnorm(x_t, self.gamma_in)
            # Q/K/V projections.
            Q = self.q_proj(h).view(N_HEADS, D_HEAD)
            K = self.k_proj(h).view(N_KV_HEADS, D_HEAD)
            V = self.v_proj(h).view(N_KV_HEADS, D_HEAD)
            # Apply RoPE to Q (per Q-head) and K (per KV-head) at pos=t.
            for hh in range(N_HEADS):
                Q[hh] = apply_rope_torch(Q[hh], t, ROPE_TABLE)
            for kh in range(N_KV_HEADS):
                K[kh] = apply_rope_torch(K[kh], t, ROPE_TABLE)
            # Write rotated K, raw V into cache row t.
            Kc[t] = K
            Vc[t] = V
            # Per Q-head scoring against its group's KV head.
            head_outs = []
            inv_sqrt = 1.0 / math.sqrt(D_HEAD)
            for hh in range(N_HEADS):
                kv_h = hh // GROUP_SIZE
                # Scores over keys 0..t inclusive.
                scores = (Kc[: t + 1, kv_h] @ Q[hh]) * inv_sqrt
                soft = F.softmax(scores, dim=-1)
                head_outs.append(soft @ Vc[: t + 1, kv_h])
            concat = torch.cat(head_outs, dim=-1)           # [d_model]
            attn_out = self.o_proj(concat)
            # Residual 1.
            x_t = x_t + attn_out
            # Pre-FFN RMSNorm + SwiGLU + residual 2.
            h2 = rmsnorm(x_t, self.gamma_post)
            gate = self.gate_proj(h2)
            up   = self.up_proj(h2)
            ffn  = self.down_proj(F.silu(gate) * up)
            x_t = x_t + ffn
            outs.append(x_t)
        h_final = torch.stack(outs, dim=0)                  # [seq, d_model]
        # Final RMSNorm + LM head (per-row).
        h_final = rmsnorm(h_final, self.gamma_final)
        return self.lm_head(h_final)


model = TinyLlama()
model.eval()

# Build a state_dict matching HF Llama naming so pack.py's resolver
# finds every tensor without --keys overrides.
sd_native = model.state_dict()
sd = {
    "embed.weight": sd_native["embed.weight"],
    "model.layers.0.input_layernorm.weight":          sd_native["gamma_in"],
    "model.layers.0.self_attn.q_proj.weight":         sd_native["q_proj.weight"],
    "model.layers.0.self_attn.k_proj.weight":         sd_native["k_proj.weight"],
    "model.layers.0.self_attn.v_proj.weight":         sd_native["v_proj.weight"],
    "model.layers.0.self_attn.o_proj.weight":         sd_native["o_proj.weight"],
    "model.layers.0.post_attention_layernorm.weight": sd_native["gamma_post"],
    "model.layers.0.mlp.gate_proj.weight":            sd_native["gate_proj.weight"],
    "model.layers.0.mlp.up_proj.weight":              sd_native["up_proj.weight"],
    "model.layers.0.mlp.down_proj.weight":            sd_native["down_proj.weight"],
    "model.norm.weight":                              sd_native["gamma_final"],
    # LM head: pack.py picks `fc{idx_plus_1}.weight` by default; we
    # emit both names so --keys "fc{idx_plus_1}.{kind}" resolves the
    # final linear.
    "fc1.weight": sd_native["lm_head.weight"],
    "fc1.bias":   sd_native["lm_head.bias"],
}
torch.save(sd, "synth.pt")

with open("prompt.bin", "wb") as f:
    for tok in PROMPT:
        f.write(struct.pack("<B", tok))   # vocab=32 ≤ 256 → 1 byte/token

ids = list(PROMPT)
generated = []
with torch.no_grad():
    for _ in range(MAX_NEW):
        logits = model(torch.tensor(ids, dtype=torch.long))
        next_tok = int(logits[-1].argmax().item())
        generated.append(next_tok)
        ids.append(next_tok)

print(f"prompt:    {PROMPT}")
print(f"generated: {generated}")
print(f"full ids:  {ids}")
print(f"n params:  {sum(p.numel() for p in model.parameters())}")

with open("expected_tokens.bin", "wb") as f:
    for tok in generated:
        f.write(struct.pack("<i", tok))
