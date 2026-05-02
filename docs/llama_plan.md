# Llama-style packing plan (TinyLlama-1.1B target)

Audit of what is missing to extend `aricode_pack.py` (currently validated on
GPT-2-small, 16/16 token greedy match) to a Llama-family decoder.
References point at the GPT-2 baseline already in tree.

## Kernels needed (new builtins in codegen_builtins.c)

- `arr_f32_silu(buf)` — NEW. SiLU is `x * sigmoid(x)`. The `vec_exp` body at
  `codegen_builtins.c:1075` plus the f32 promote/narrow pattern used by
  `arr_f32_exp` (`:4784`) and `arr_f32_softmax` (`:4835`) is the template.
  ~80 lines, mostly copy of the exp body with one extra `vmulps` against the
  pre-promotion x. Optional first cut: keep it as a stdlib helper using
  scalar `math_exp` (mirroring `gelu_f32` in `aricode_pack.py:484-516`); a
  builtin can be added later when the FFN dominates wall.
- `arr_f32_rmsnorm(buf, dim, eps)` — NEW. The existing `arr_f32_layernorm`
  (`codegen_builtins.c:5078`) is the closest sibling: same outer K-group
  loop, same per-group scalar passes. RMSNorm just drops the mean-subtract
  and replaces `var = E[(x-μ)²]` with `ms = E[x²]`. Easiest implementation:
  duplicate the `arr_f32_layernorm` block, delete the mean pass and the
  subtract, square `xmm1` instead of `xmm1-xmm2`, leave the sqrt + divide
  loop intact. ~120 lines (vs ~250 for layernorm — about half because
  there's no mean pass). Affine γ-scale is a separate stdlib helper (no β
  in Llama).
- (No new builtin needed for RoPE.) RoPE is two `vmulps`/`vaddsubps` per
  pair-of-floats and a per-position lookup of `(cos, sin)`. The math fits
  cleanly into a stdlib `.ari` file using the existing `math_sin` / `math_cos`
  scalars (`codegen_builtins.c:3346` / `:3369`). A vectorised
  `arr_f32_rope_apply` builtin is a future drop-in once decode wall is
  measured.

## Stdlib additions (new .ari files)

- `rope_f32.ari` — NEW. ~70 lines.
  - `rope_alloc_f32(max_seq, d_head, theta) -> i32` — pre-computes a
    `[max_seq, d_head]` table of interleaved `(cos, sin)` pairs at alloc
    time using `math_sin` / `math_cos`. State stores theta, d_head,
    max_seq, table. Mirrors `attn_kv_alloc_f32`'s state-array idiom
    (`attention_kv_f32.ari:39-70`).
  - `rope_apply_f32(table, vec, pos, d_head)` — in-place rotate. For
    `i in 0..d_head/2`: `(x0,x1) = (vec[2i], vec[2i+1]); c = table[pos*d_head + 2i]; s = table[pos*d_head + 2i+1]; vec[2i] = x0*c - x1*s; vec[2i+1] = x0*s + x1*c;`
    Scalar inner loop is fine at d_head=64..128.
  - HF stores nothing in the state_dict for RoPE in TinyLlama (the
    `rotary_emb.inv_freq` cache is non-persistent). pack.py reads
    `theta` (config: `rope_theta`, default 10000.0) directly from the JSON
    arch entry; the table is built at runtime by the static binary itself.
- `rmsnorm_f32.ari` — NEW. ~25 lines. Wraps the new builtin to apply the
  affine γ pass after normalisation. Same pattern as
  `LAYERNORM_HELPER` in `aricode_pack.py:447-465` minus β.
- `attention_gqa_kv_f32.ari` — NEW (~180 lines). The current
  `attention_mh_kv_f32.ari` (142 lines) hard-codes `n_kv_heads == n_heads`
  by allocating one `attn_kv_alloc_f32` state per Q head
  (`attention_mh_kv_f32.ari:52-58`). For GQA we need:
  - `n_kv_heads` KV cache states (allocated as today), each shaped
    `[max_seq, d_head]`.
  - `n_heads` per-Q-head loops, where the inner step computes
    `Q_h = W_Q[h*d_head:(h+1)*d_head, :] · X` against the KV cache of
    `kv_h = h / (n_heads / n_kv_heads)`.
  - W_K / W_V become `(n_kv_heads * d_head, d_model)` instead of
    `(d_model, d_model)` — `tensor_specs` for the new arch op must yield
    `n_kv_heads * d_head * d_model` weight elements for K and V (vs
    `d_model * d_model` for Q and O).
  - RoPE is applied on Q (for the current head) and K (for the current KV
    head) inside the loop, BEFORE writing K to the cache and BEFORE the
    score matvec. Cached K is already-rotated (HF reference matches).

## pack.py arch op additions

- `["rmsnorm", dim]` — adds a branch in `tensor_specs` (`:725-730`,
  yields `(dim, 0)` — γ only, no β), in `tensor_names` (`:774`,
  use `Wn`/no bias), in `weight_tensors`, in `gen_act_decls` pseudo
  list (`:884` and `:538`), in `gen_forward` (mirrors the layernorm
  case at `:984`, calling `rmsnorm_affine_f32`). State_dict key:
  `model.layers.{i}.input_layernorm.weight` and
  `model.layers.{i}.post_attention_layernorm.weight`.
  Helper-emit path mirrors `LAYERNORM_HELPER`. ~60 lines added in pack.py.
- `["swiglu_ffn", d_in, d_ffn, d_out]` — the cleanest seam. Bundles the
  three Linears (`gate_proj`, `up_proj`, `down_proj`) and the elementwise
  `silu(gate(x)) * up(x)` step into ONE arch op. Justification: the
  intermediate buffers have shape `[d_ffn]` (≈ 5632 for TinyLlama vs
  d_model=2048) — exposing them as separate Linear ops would force
  `gen_act_decls` to grow the activation-buffer set. Bundling keeps two
  `[d_ffn]` scratches local to the op. `tensor_specs` yields THREE
  `(W, b=0)` pairs with suffixes `gate`/`up`/`down`. State_dict pattern:
  `model.layers.{i}.mlp.{gate,up,down}_proj.weight` (no biases in Llama).
  `gen_forward` emits: `arr_f32_matvec(W_gate, x, zb, gate_buf, ...)` →
  `silu_f32(gate_buf)` → `arr_f32_mul(gate_buf, up_buf, gate_buf)`
  (existing builtin at `codegen_builtins.c:4346`) →
  `arr_f32_matvec(W_down, gate_buf, zb, dst, ...)`. ~110 lines in pack.py.
- `["multi_head_attention_gqa_kv", max_seq, d_model, n_heads, n_kv_heads, theta]` —
  parallels `multi_head_attention_kv` (`:711-724`, `:821-828`,
  `:963-968`, `:1078-1099`, `:1548-1564`). When `n_kv_heads == n_heads`
  (Llama-2-7B), it would degrade to the current MHA-KV path; we still
  emit the new path so the rope_theta plumbing is uniform.
  - `tensor_specs` yields four entries: `q` size `d_model²`, `k` and `v`
    sizes `n_kv_heads * d_head * d_model`, `o` size `d_model²`.
  - `gen_scratch` allocates a `rope_alloc_f32(max_seq, d_head, theta)`
    table once and a `attn_gqa_kv_alloc_f32` state. Total ~150 lines
    across pack.py for this op.
- `["embedding", vocab, d_model, 1]` already works; tied LM head logic
  (`gen_decoder_main:1670-1682`) needs a knob: when the state_dict has a
  separate `lm_head.weight`, use it; when missing, alias from
  `model.embed_tokens.weight`. ~10-line patch around `:2516`.

## Estimated complexity

- New builtins: ~120 lines (`arr_f32_rmsnorm`) in `codegen_builtins.c`.
  `arr_f32_silu` deferrable (~80 lines) — start with stdlib helper.
- New stdlib: ~275 lines total across `rope_f32.ari` (~70), `rmsnorm_f32.ari`
  (~25), `attention_gqa_kv_f32.ari` (~180).
- `aricode_pack.py` changes: ~330 lines additions touching ~12 sites
  (arch dispatch in 8 functions: `tensor_specs`, `tensor_names`,
  `weight_tensors`, `gen_act_decls`, `residual_slots`, `gen_forward`,
  `gen_scratch`, `_load_pytorch_state_dict` key-mapper near `:2465`).
  Two new helpers: `RMSNORM_HELPER`, `SWIGLU_HELPER` (mirror
  `LAYERNORM_HELPER`/`GELU_HELPER`).
- Tokenizer/prompt: SentencePiece happens in the Python prepare script
  that writes `prompt.bin`; pack.py is unchanged (uses 4-byte tokens
  since vocab=32000 fits 2-byte but Llama-3 at 128K wants 2-byte too —
  `_embedding_token_bytes` already returns 2 for vocab≤65536, `:427-439`).

Total session estimate: **3 to 4 focused sessions** (~12-16 hours).
Bottleneck is correctness validation (numerics-match against
HuggingFace reference), not LOC.

## Recommended order of attack

1. **RMSNorm first.** Smallest scope, no new dependencies. Add
   `arr_f32_rmsnorm` builtin + `rmsnorm_f32.ari` helper + `["rmsnorm", dim]`
   arch op. Validate by replacing one LayerNorm in the GPT-2 arch with
   RMSNorm against a custom-trained tiny RMSNorm checkpoint.
2. **SwiGLU second.** Builds only on existing `arr_f32_mul` +
   `arr_f32_matvec` + a SiLU helper. Validate by training a 1-layer FFN
   with SwiGLU vs GELU on the same task; greedy-match.
3. **RoPE third.** Pure stdlib (`rope_f32.ari`); the rotation lives
   inside a hand-written test arch op before being folded into GQA.
   Validate by computing Q-after-RoPE for known positions vs
   `transformers.LlamaRotaryEmbedding`.
4. **GQA fourth.** Pulls in (1)-(3). Implement
   `attention_gqa_kv_f32.ari` + `["multi_head_attention_gqa_kv", ...]`.
   Validate first with `n_kv_heads == n_heads` (degrades to MHA-KV)
   against the existing GPT-2 checkpoint, then with TinyLlama's 32/4 split
   against a 1-layer reference.
5. **Wire TinyLlama-1.1B end-to-end.** 22 layers × (RMSNorm + GQA-KV +
   RMSNorm + SwiGLU) + final RMSNorm + LM head. Final greedy-match
   against PyTorch on a fixed prompt — same validation harness as the
   GPT-2 pass.
