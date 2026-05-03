"""llama2_7b — pack the real Llama-2-7B (NousResearch's open
redistribution of Meta's Llama-2-7B-hf) as a static ELF.

Architecture: 32 transformer blocks, vocab=32000, d_model=4096,
n_heads=32, n_kv_heads=32 (no GQA — degrades to plain multi-head),
d_head=128, ff=11008, RoPE θ=10000, RMSNorm eps=1e-5.  Same shape
as TinyLlama-1.1B just scaled up ≈6×.

# Memory discipline

The state_dict is held in bf16 (~13 GB) so the load fits comfortably
in our 31 GB host.  We then iterate tensor-by-tensor, converting each
to f32 + applying the Q/K row permutation + saving to a single
re-keyed `.pt` file.  The intermediate f32 tensor for the LARGEST
weight (down_proj at 11008 × 4096 = 180 MB f32) never has more than
one copy live at a time, so peak RSS stays well under 16 GB.

# RoPE convention surgery

HF Llama uses split-half RoPE; our `rope_apply_pairs` builtin uses
INTERLEAVED pairs.  Same fix as the TinyLlama prepare: permute Q and
K weight rows per head so:
    new_W[h*d_head + 2k]     = old_W[h*d_head + k]
    new_W[h*d_head + 2k + 1] = old_W[h*d_head + k + d_head/2]

Mathematically equivalent — only the physical layout differs.
"""

import math
import struct
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "NousResearch/Llama-2-7b-hf"
PROMPT   = "Once upon a time, there was a"
MAX_NEW  = 4    # 7B greedy on CPU is ~30-60 s/token; keep small.


def permute_qk_rows(W, n_heads_or_kv, d_head):
    """Re-order W's d_out rows within each head from HF's split-half
    layout to our interleaved layout.  Vectorised via reshape +
    fancy-indexing assignment so the per-tensor cost is dominated by
    the memcpy rather than 2048 Python-level iterations on Llama-2-7B."""
    half = d_head // 2
    out = W.clone()
    # View as (n_heads, d_head, ...) — works for both 1-D (b_*) and
    # 2-D (W_*) inputs since the leading dim is n_heads * d_head.
    Wv  = W.view(n_heads_or_kv, d_head, -1) if W.ndim > 1 else \
          W.view(n_heads_or_kv, d_head)
    ov  = out.view(n_heads_or_kv, d_head, -1) if out.ndim > 1 else \
          out.view(n_heads_or_kv, d_head)
    ov[:, 0::2] = Wv[:, :half]
    ov[:, 1::2] = Wv[:, half:]
    return out


def main():
    print(f"loading {MODEL_ID} (bf16)...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
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
    rope_p   = getattr(cfg, "rope_parameters", None) or {}
    theta    = float(rope_p.get("rope_theta",
                                getattr(cfg, "rope_theta", 10000.0)))
    eps      = float(cfg.rms_norm_eps)

    print(f"  vocab={vocab} d_model={d_model} n_layer={n_layer} "
          f"n_heads={n_heads} n_kv={n_kv} d_head={d_head} ff={ff} "
          f"theta={theta} eps={eps}", flush=True)

    sd_in = model.state_dict()
    sd_out = {}

    # Keep tensors in bf16 to fit in ~13 GB RAM; pack.py's int8 staging
    # path calls `.detach().cpu().numpy().astype("float32")` per tensor,
    # so the f32 conversion happens lazily one tensor at a time at
    # pack-time.  Peak RSS during prepare stays under 16 GB; peak RSS
    # during pack stays under 14 GB (bf16 model + one f32 tensor max).
    def keep(t):
        return t.detach().contiguous()

    print(f"  re-keying + permuting Q/K (32 blocks, bf16 throughout)...",
          flush=True)
    t0 = time.time()
    sd_out["embed.weight"] = keep(sd_in["model.embed_tokens.weight"])
    for i in range(n_layer):
        prefix = f"model.layers.{i}"
        sd_out[f"{prefix}.input_layernorm.weight"] = \
            keep(sd_in[f"{prefix}.input_layernorm.weight"])
        sd_out[f"{prefix}.post_attention_layernorm.weight"] = \
            keep(sd_in[f"{prefix}.post_attention_layernorm.weight"])

        Wq = keep(sd_in[f"{prefix}.self_attn.q_proj.weight"])
        Wk = keep(sd_in[f"{prefix}.self_attn.k_proj.weight"])
        Wv = keep(sd_in[f"{prefix}.self_attn.v_proj.weight"])
        Wo = keep(sd_in[f"{prefix}.self_attn.o_proj.weight"])
        sd_out[f"{prefix}.self_attn.q_proj.weight"] = \
            permute_qk_rows(Wq, n_heads, d_head)
        sd_out[f"{prefix}.self_attn.k_proj.weight"] = \
            permute_qk_rows(Wk, n_kv, d_head)
        sd_out[f"{prefix}.self_attn.v_proj.weight"] = Wv
        sd_out[f"{prefix}.self_attn.o_proj.weight"] = Wo

        sd_out[f"{prefix}.mlp.gate_proj.weight"] = \
            keep(sd_in[f"{prefix}.mlp.gate_proj.weight"])
        sd_out[f"{prefix}.mlp.up_proj.weight"] = \
            keep(sd_in[f"{prefix}.mlp.up_proj.weight"])
        sd_out[f"{prefix}.mlp.down_proj.weight"] = \
            keep(sd_in[f"{prefix}.mlp.down_proj.weight"])

    sd_out["model.norm.weight"] = keep(sd_in["model.norm.weight"])
    sd_out["fc1.weight"] = keep(sd_in["lm_head.weight"])
    sd_out["fc1.bias"]   = torch.zeros(vocab, dtype=torch.float32)
    print(f"  ok, took {time.time() - t0:.1f}s", flush=True)

    print(f"  re-keyed {len(sd_out)} tensors (bf16)", flush=True)
    print(f"  saving synth.pt (~13 GB bf16)...", flush=True)
    t0 = time.time()
    torch.save(sd_out, "synth.pt")
    print(f"  saved in {time.time() - t0:.1f}s", flush=True)

    tokens = tokenizer.encode(PROMPT, add_special_tokens=True)
    print(f"  prompt: {PROMPT!r}", flush=True)
    print(f"  prompt tokens ({len(tokens)}): {tokens}", flush=True)
    with open("prompt.bin", "wb") as f:
        for tok in tokens:
            f.write(struct.pack("<H", tok))

    # Greedy reference (bf16 since the model is loaded that way; greedy
    # argmax should be stable vs f32 except in razor-thin logit cases).
    print(f"  generating {MAX_NEW} reference tokens with HF (bf16)...",
          flush=True)
    t0 = time.time()
    ids = list(tokens)
    generated = []
    with torch.no_grad():
        for _ in range(MAX_NEW):
            out = model(torch.tensor(ids, dtype=torch.long).unsqueeze(0))
            next_tok = int(out.logits[0, -1].argmax().item())
            generated.append(next_tok)
            ids.append(next_tok)
            print(f"    [{len(generated)}/{MAX_NEW}] {next_tok}",
                  flush=True)
    print(f"  generated in {time.time() - t0:.1f}s", flush=True)
    print(f"  decoded: {tokenizer.decode(generated)!r}", flush=True)

    with open("expected_tokens.bin", "wb") as f:
        for tok in generated:
            f.write(struct.pack("<i", tok))


if __name__ == "__main__":
    main()
