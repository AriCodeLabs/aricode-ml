"""tinyllama_1b — pack the real HuggingFace TinyLlama-1.1B as a static
ELF and validate token-for-token greedy match against the HF reference.

This is the matching trophy for the Llama family (after gpt2_small for
the GPT-2 family).  Architecture: 22 transformer blocks, vocab=32000,
d_model=2048, n_heads=32, n_kv_heads=4, ff=5632, RoPE θ=10000.

# RoPE convention surgery

HuggingFace Llama applies RoPE in the SPLIT-HALF convention:
  q'[j]              = q[j] * cos[θ_j] - q[j + d_head/2] * sin[θ_j]
  q'[j + d_head/2]   = q[j + d_head/2] * cos[θ_j] + q[j] * sin[θ_j]
                                                        for j < d_head/2

Our `rope_apply_f32` kernel uses the INTERLEAVED convention:
  q'[2k]             = q[2k]   * cos[θ_k] - q[2k+1] * sin[θ_k]
  q'[2k+1]           = q[2k+1] * cos[θ_k] + q[2k]   * sin[θ_k]

Both rotate the same set of (cos, sin) angles; they only differ in
which physical positions each pair occupies.  By permuting the d_out
rows of W_Q and W_K within each head, we make q[2k] and q[2k+1] in
our representation hold what HF would have placed at indices
(k, k + d_head/2).  Then our interleaved RoPE produces the same q'
vector HF's split-half RoPE would have produced.

W_V and W_O are untouched (V is not rotated; O is post-attention).

# Output layout

Single state_dict (synth.pt) keyed for pack.py's resolvers:

  embed.weight                                   → embedding (vocab × d_model)
  model.layers.{0..21}.input_layernorm.weight     → RMSNorm γ (pre-attn)
  model.layers.{0..21}.self_attn.q_proj.weight    → Q projection (PERMUTED)
  model.layers.{0..21}.self_attn.k_proj.weight    → K projection (PERMUTED)
  model.layers.{0..21}.self_attn.v_proj.weight    → V projection (untouched)
  model.layers.{0..21}.self_attn.o_proj.weight    → O projection (untouched)
  model.layers.{0..21}.post_attention_layernorm.weight → RMSNorm γ (pre-FFN)
  model.layers.{0..21}.mlp.gate_proj.weight       → SwiGLU gate
  model.layers.{0..21}.mlp.up_proj.weight         → SwiGLU up
  model.layers.{0..21}.mlp.down_proj.weight       → SwiGLU down
  model.norm.weight                              → final RMSNorm γ
  fc1.weight                                     → LM head (Linear li=0)

Plus prompt.bin (uint16 LE token IDs) and expected_tokens.bin
(int32 LE greedy reference).
"""

import math
import struct
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
PROMPT   = "Once upon a time, there was a"
MAX_NEW  = 8


def permute_qk_rows(W, n_heads_or_kv, d_head):
    """Re-order W's d_out rows within each head from HF's split-half
    layout to our interleaved layout.

    Within each head's d_head consecutive rows:
        new[2k]     = old[k]
        new[2k+1]   = old[k + d_head/2]
                                                 for k in 0..d_head/2

    W shape: (n_heads_or_kv * d_head, d_in).
    """
    out = W.clone()
    half = d_head // 2
    for h in range(n_heads_or_kv):
        base = h * d_head
        for k in range(half):
            out[base + 2 * k]     = W[base + k]
            out[base + 2 * k + 1] = W[base + k + half]
    return out


def main():
    print(f"loading {MODEL_ID}...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                 torch_dtype=torch.float32)
    model.eval()
    print(f"  ok, took {time.time() - t0:.1f}s", flush=True)

    cfg = model.config
    vocab    = cfg.vocab_size
    d_model  = cfg.hidden_size
    n_layer  = cfg.num_hidden_layers
    n_heads  = cfg.num_attention_heads
    n_kv     = cfg.num_key_value_heads
    d_head   = d_model // n_heads
    ff       = cfg.intermediate_size
    # rope_theta moved into a sub-dict in newer transformers versions
    # (`rope_parameters['rope_theta']`); legacy `cfg.rope_theta` still
    # works on older releases.  Fall back to 10000 (the Llama default).
    rope_params = getattr(cfg, "rope_parameters", None) or {}
    theta = float(rope_params.get("rope_theta",
                                  getattr(cfg, "rope_theta", 10000.0)))
    eps      = float(cfg.rms_norm_eps)

    print(f"  vocab={vocab} d_model={d_model} n_layer={n_layer} "
          f"n_heads={n_heads} n_kv={n_kv} d_head={d_head} ff={ff} "
          f"theta={theta} eps={eps}", flush=True)
    if eps != 1e-5:
        print(f"  WARNING: rms_norm_eps={eps} differs from our hard-coded "
              f"1e-5; output will drift.  Patch RMSNORM_HELPER if needed.",
              flush=True)

    sd_in = model.state_dict()
    sd_out = {}

    # Token embedding.
    sd_out["embed.weight"] = sd_in["model.embed_tokens.weight"].contiguous()

    for i in range(n_layer):
        prefix = f"model.layers.{i}"
        # RMSNorms — γ only, no β.
        sd_out[f"{prefix}.input_layernorm.weight"] = \
            sd_in[f"{prefix}.input_layernorm.weight"].contiguous()
        sd_out[f"{prefix}.post_attention_layernorm.weight"] = \
            sd_in[f"{prefix}.post_attention_layernorm.weight"].contiguous()

        # Attention — permute Q and K for interleaved RoPE.
        Wq = sd_in[f"{prefix}.self_attn.q_proj.weight"]
        Wk = sd_in[f"{prefix}.self_attn.k_proj.weight"]
        Wv = sd_in[f"{prefix}.self_attn.v_proj.weight"]
        Wo = sd_in[f"{prefix}.self_attn.o_proj.weight"]
        sd_out[f"{prefix}.self_attn.q_proj.weight"] = \
            permute_qk_rows(Wq, n_heads, d_head).contiguous()
        sd_out[f"{prefix}.self_attn.k_proj.weight"] = \
            permute_qk_rows(Wk, n_kv, d_head).contiguous()
        sd_out[f"{prefix}.self_attn.v_proj.weight"] = Wv.contiguous()
        sd_out[f"{prefix}.self_attn.o_proj.weight"] = Wo.contiguous()

        # SwiGLU FFN — no biases, no rotation needed.
        sd_out[f"{prefix}.mlp.gate_proj.weight"] = \
            sd_in[f"{prefix}.mlp.gate_proj.weight"].contiguous()
        sd_out[f"{prefix}.mlp.up_proj.weight"] = \
            sd_in[f"{prefix}.mlp.up_proj.weight"].contiguous()
        sd_out[f"{prefix}.mlp.down_proj.weight"] = \
            sd_in[f"{prefix}.mlp.down_proj.weight"].contiguous()

    # Final RMSNorm + LM head.
    sd_out["model.norm.weight"] = sd_in["model.norm.weight"].contiguous()
    # Llama's LM head is NOT tied with the embedding by default for
    # TinyLlama-1.1B (verify).  Save under fc1 (the only Linear in the
    # arch — pack.py's `fc{idx_plus_1}` template resolves li=0 → fc1).
    sd_out["fc1.weight"] = sd_in["lm_head.weight"].contiguous()
    # No bias on Llama's lm_head; pack.py's Linear loader requires one
    # so we synthesise zeros.
    sd_out["fc1.bias"] = torch.zeros(vocab, dtype=torch.float32)

    print(f"  re-keyed {len(sd_out)} tensors", flush=True)
    torch.save(sd_out, "synth.pt")
    print(f"  wrote synth.pt", flush=True)

    # Tokenize prompt.
    tokens = tokenizer.encode(PROMPT, add_special_tokens=True)
    print(f"  prompt: {PROMPT!r}", flush=True)
    print(f"  prompt tokens ({len(tokens)}): {tokens}", flush=True)
    with open("prompt.bin", "wb") as f:
        for tok in tokens:
            f.write(struct.pack("<H", tok))   # uint16 LE
    print(f"  wrote prompt.bin ({len(tokens) * 2} bytes)", flush=True)

    # Greedy reference: generate MAX_NEW tokens from the prompt.
    print(f"  generating {MAX_NEW} tokens with HF reference...", flush=True)
    t0 = time.time()
    ids = list(tokens)
    generated = []
    with torch.no_grad():
        for _ in range(MAX_NEW):
            out = model(torch.tensor(ids, dtype=torch.long).unsqueeze(0))
            next_tok = int(out.logits[0, -1].argmax().item())
            generated.append(next_tok)
            ids.append(next_tok)
    print(f"  generated in {time.time() - t0:.1f}s", flush=True)
    print(f"  generated tokens: {generated}", flush=True)
    print(f"  decoded: {tokenizer.decode(generated)!r}", flush=True)

    with open("expected_tokens.bin", "wb") as f:
        for tok in generated:
            f.write(struct.pack("<i", tok))   # int32 LE


if __name__ == "__main__":
    main()
