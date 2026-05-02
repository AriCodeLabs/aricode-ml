"""gpt2_small — pack HuggingFace GPT-2 (124M, 12 blocks, d_model=768) as
a static ELF.  This is the trophy demo: a real LLM packed end-to-end.

Re-keys HF's c_attn (fused QKV in Conv1D layout) into the per-layer
q_proj/k_proj/v_proj/out_proj keys pack.py's MHA-KV resolver expects;
re-keys the MLP c_fc/c_proj into fc1/fc2-style numbered Linears the
default --keys template can address; ties the LM head to the input
embedding by saving an explicit copy (the resolver wants a 2-D tensor
+ a bias slot, so we synthesise a zeros bias too).

HF's Conv1D stores weights as (in_features, out_features) — the
opposite of nn.Linear.  Every c_attn / c_proj / c_fc weight has to be
transposed before it can live in pack.py's nn.Linear-shaped resolver.

Layer / counter layout (matches arch.json):
  embedding (ei=0)        embed.weight                       (50257, 768)
  positional_embedding    wpe.weight                         (1024,  768)
  block i = 0..11:
    ln_{2i}.{w,b}         block-i pre-attn LN                (768,)
    layers.i.q_proj.*     Wq from c_attn (split + transpose) (768, 768)
    layers.i.k_proj.*     Wk                                  (768, 768)
    layers.i.v_proj.*     Wv                                  (768, 768)
    layers.i.out_proj.*   Wo from c_proj.t()                  (768, 768)
    ln_{2i+1}.{w,b}       block-i pre-FFN LN                  (768,)
    fc{2i+1}.{w,b}        FFN1 from c_fc.t()                  (3072, 768)
    fc{2i+2}.{w,b}        FFN2 from c_proj_mlp.t()            (768, 3072)
  ln_24.{w,b}             final ln_f                           (768,)
  fc25.weight             LM head — TIED COPY of wte.weight   (50257, 768)
  fc25.bias               zeros                                (50257,)
"""

import struct

import torch

torch.manual_seed(0)

MODEL_NAME = "gpt2"
N_BLOCKS   = 12
D_MODEL    = 768
N_HEADS    = 12
FF_DIM     = 3072
VOCAB      = 50257
MAX_POS    = 1024
PROMPT_TXT = "Once upon a time, there was a"
MAX_NEW    = 16


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"loading {MODEL_NAME}...")
    tok = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()
    cfg = model.config
    assert cfg.n_layer == N_BLOCKS, f"unexpected n_layer {cfg.n_layer}"
    assert cfg.n_embd == D_MODEL,   f"unexpected n_embd {cfg.n_embd}"
    assert cfg.n_head == N_HEADS,   f"unexpected n_head {cfg.n_head}"
    assert cfg.vocab_size == VOCAB, f"unexpected vocab {cfg.vocab_size}"

    sd = {}
    transformer = model.transformer

    # --- token + positional embeddings ---------------------------------
    sd["embed.weight"] = transformer.wte.weight.detach().clone()
    sd["wpe.weight"]   = transformer.wpe.weight.detach().clone()

    # --- per-block ----------------------------------------------------
    for i, block in enumerate(transformer.h):
        # pre-attn LN
        sd[f"ln_{2 * i}.weight"] = block.ln_1.weight.detach().clone()
        sd[f"ln_{2 * i}.bias"]   = block.ln_1.bias.detach().clone()

        # c_attn fuses Q|K|V along its OUTPUT dimension.  HF Conv1D
        # stores weights as (in_features, out_features), so c_attn.weight
        # has shape (D_MODEL, 3*D_MODEL).  Transpose to nn.Linear
        # convention (out_features, in_features) = (3*D_MODEL, D_MODEL),
        # then split along dim 0 into three (D_MODEL, D_MODEL) chunks.
        Wqkv = block.attn.c_attn.weight.detach().t().contiguous()
        bqkv = block.attn.c_attn.bias.detach().clone()
        Wq, Wk, Wv = Wqkv.chunk(3, dim=0)
        bq, bk, bv = bqkv.chunk(3, dim=0)
        sd[f"layers.{i}.q_proj.weight"] = Wq.contiguous()
        sd[f"layers.{i}.q_proj.bias"]   = bq.contiguous()
        sd[f"layers.{i}.k_proj.weight"] = Wk.contiguous()
        sd[f"layers.{i}.k_proj.bias"]   = bk.contiguous()
        sd[f"layers.{i}.v_proj.weight"] = Wv.contiguous()
        sd[f"layers.{i}.v_proj.bias"]   = bv.contiguous()

        # output projection (Conv1D in HF; same transpose as c_attn).
        sd[f"layers.{i}.out_proj.weight"] = (
            block.attn.c_proj.weight.detach().t().contiguous())
        sd[f"layers.{i}.out_proj.bias"]   = block.attn.c_proj.bias.detach().clone()

        # pre-FFN LN
        sd[f"ln_{2 * i + 1}.weight"] = block.ln_2.weight.detach().clone()
        sd[f"ln_{2 * i + 1}.bias"]   = block.ln_2.bias.detach().clone()

        # FFN: c_fc (D_MODEL → FF_DIM) and c_proj (FF_DIM → D_MODEL),
        # both Conv1D so transpose to nn.Linear convention.  fc-keys
        # are 1-indexed and global across blocks, so block i's FFNs
        # become fc{2i+1} (in: D_MODEL → FF_DIM) and fc{2i+2}
        # (in: FF_DIM → D_MODEL).
        Wff1 = block.mlp.c_fc.weight.detach().t().contiguous()   # (FF_DIM, D_MODEL)
        bff1 = block.mlp.c_fc.bias.detach().clone()              # (FF_DIM,)
        Wff2 = block.mlp.c_proj.weight.detach().t().contiguous() # (D_MODEL, FF_DIM)
        bff2 = block.mlp.c_proj.bias.detach().clone()            # (D_MODEL,)
        sd[f"fc{2 * i + 1}.weight"] = Wff1
        sd[f"fc{2 * i + 1}.bias"]   = bff1
        sd[f"fc{2 * i + 2}.weight"] = Wff2
        sd[f"fc{2 * i + 2}.bias"]   = bff2

    # --- final LN -----------------------------------------------------
    sd[f"ln_{2 * N_BLOCKS}.weight"] = transformer.ln_f.weight.detach().clone()
    sd[f"ln_{2 * N_BLOCKS}.bias"]   = transformer.ln_f.bias.detach().clone()

    # --- LM head (tied to wte) ----------------------------------------
    # The LM head index follows the FFN counter: fc{2*N_BLOCKS + 1}.
    # HF stores lm_head.weight as a VIEW of wte; we explicit-clone here
    # so pack.py's resolver sees a stand-alone tensor it can serialise.
    # No bias on the tied head — synthesise a zeros tensor since pack.py's
    # linear resolver requires (weight, bias) pairs.
    head_idx = 2 * N_BLOCKS + 1
    sd[f"fc{head_idx}.weight"] = transformer.wte.weight.detach().clone()
    sd[f"fc{head_idx}.bias"]   = torch.zeros(VOCAB, dtype=torch.float32)

    print(f"state-dict tensors: {len(sd)}")
    print(f"  embed.weight       : {tuple(sd['embed.weight'].shape)}")
    print(f"  wpe.weight         : {tuple(sd['wpe.weight'].shape)}")
    print(f"  layers.0.q_proj.W  : {tuple(sd['layers.0.q_proj.weight'].shape)}")
    print(f"  layers.0.out_proj.W: {tuple(sd['layers.0.out_proj.weight'].shape)}")
    print(f"  fc1.weight         : {tuple(sd['fc1.weight'].shape)}")
    print(f"  fc2.weight         : {tuple(sd['fc2.weight'].shape)}")
    print(f"  ln_0.weight        : {tuple(sd['ln_0.weight'].shape)}")
    print(f"  ln_24.weight       : {tuple(sd['ln_24.weight'].shape)}")
    print(f"  fc{head_idx}.weight        : {tuple(sd[f'fc{head_idx}.weight'].shape)}")

    torch.save(sd, "synth.pt")
    print("wrote synth.pt")

    # --- prompt + reference greedy decode -----------------------------
    ids = tok.encode(PROMPT_TXT)
    print(f"prompt text:   {PROMPT_TXT!r}")
    print(f"prompt tokens: {ids}")

    with open("prompt.bin", "wb") as f:
        for tid in ids:
            assert 0 <= tid < 65536, f"vocab too large for 2-byte encoding: {tid}"
            f.write(struct.pack("<H", tid))   # little-endian uint16
    print(f"wrote prompt.bin ({len(ids) * 2} bytes; {len(ids)} tokens)")

    # Reference greedy decode using the in-memory model.
    generated = []
    with torch.no_grad():
        cur = list(ids)
        for _ in range(MAX_NEW):
            x = torch.tensor([cur], dtype=torch.long)
            out = model(x).logits[0, -1]
            nxt = int(out.argmax().item())
            generated.append(nxt)
            cur.append(nxt)
    print(f"ref greedy:    {generated}")
    print(f"ref text:      {tok.decode(generated)!r}")

    with open("expected_tokens.bin", "wb") as f:
        for tid in generated:
            f.write(struct.pack("<i", tid))
    print(f"wrote expected_tokens.bin ({MAX_NEW} int32s)")


if __name__ == "__main__":
    main()
