# Cross-arch dispatch consistency audit

Audit target: `aricode-stdlib/aricode-ml/tools/aricode_pack.py` (3105 lines, audited at HEAD `7ddcb6e`).

Scope: every weight-bearing or shape-changing arch op, cross-checked against ALL eight dispatch sites it must touch. Read-only — no source edits.

## Coverage matrix

Sites legend:  TS = `tensor_specs` (756);  TN = `tensor_names` (872);  WT = `weight_tensors` (924);  AD = `gen_act_decls` first-layer + size walk (1032);  SC = `gen_scratch` per-op alloc + 2-pass batched-Linear walker (1188);  FW = `gen_forward` (1555);  RM = state-dict resolver in `main()` (~2531);  PS = `pseudo` tuple (3 sites: 617, 1044, 1352);  IS = `is_decoder_arch`/`needs_kv_attention_lib` (417/428);  RS = `residual_slots` (602).

| op                              | TS | TN | WT | AD | SC | FW | RM | PS | IS | RS |
|---------------------------------|----|----|----|----|----|----|----|----|----|----|
| linear                          | OK | OK | OK | OK | OK | OK | OK | n/a| n/a| OK |
| conv2d_3x3_p1                   | OK | OK | OK | OK | OK | OK | OK | n/a| n/a| OK |
| maxpool_2x2                     | n/a| n/a| n/a| OK | OK | OK | n/a| n/a| n/a| OK |
| attention                       | OK | OK | OK | OK | OK | OK | OK | n/a| n/a| OK |
| multi_head_attention            | OK | OK | OK | OK | OK | OK | WARN(2650/2658)| n/a| n/a| OK |
| multi_head_attention_kv         | OK | OK | OK (mi shared) | OK | OK | OK | OK | n/a| OK | MISS |
| multi_head_attention_gqa_kv     | OK | OK | OK (gi sep)| OK | partial | OK | OK | n/a| OK | MISS |
| embedding                       | OK | OK | OK | OK | OK | OK | OK | n/a| n/a| OK |
| positional_embedding            | OK | OK | OK | OK | OK | OK | OK | OK | n/a| MISS-pseudo-only|
| layernorm                       | OK | OK | OK | OK | n/a| OK | OK | OK | n/a| OK (impl) |
| rmsnorm                         | OK | OK | OK | OK | n/a| OK | OK | OK | n/a| MISS |
| swiglu_ffn                      | OK | OK | OK | OK | OK | OK | OK | n/a| n/a| MISS |
| save_residual / add_residual    | n/a| n/a| n/a| OK | OK (alloc) | OK | n/a| OK | n/a| OK |
| flatten / relu / sigmoid / tanh / softmax / gelu | n/a| n/a| n/a| OK | n/a| OK | n/a| OK | n/a| OK |

`partial` / `MISS` / `WARN` rows are detailed below.

## Gaps found

1. **`multi_head_attention` resolver: undefined `key_stem` in error path** — `tools/aricode_pack.py:2650` and `:2658`. Both `raise SystemExit` strings interpolate `{key_stem}` but the variable was never bound (the loop variable is `proj` and the dict lookup yields `hf_key, dbert_key`). If a state-dict is missing a Q/K/V/O projection or has the wrong shape, the failure path raises `NameError: name 'key_stem' is not defined` instead of the intended diagnostic. Latent — only triggers on a packing error. **Fix:** replace `{key_stem}` with `{proj}` (or `{hf_key}`) in both f-strings.

2. **`residual_slots` walker missing `multi_head_attention_kv`, `multi_head_attention_gqa_kv`, `swiglu_ffn`, `rmsnorm`, `embedding`-as-pseudo** — `tools/aricode_pack.py:602–678`. The first-layer dispatch (623–639) handles `linear / conv2d_3x3_p1 / attention / multi_head_attention / layernorm / embedding` and then `else: raise`. It has NO branch for `rmsnorm`, `multi_head_attention_kv`, `multi_head_attention_gqa_kv`, `positional_embedding`, or `swiglu_ffn`. The size-tracking walker (656–674) is also missing branches for `multi_head_attention_kv`, `multi_head_attention_gqa_kv`, `embedding`, and `swiglu_ffn`. **Condition that exposes:** any decoder arch that uses `save_residual` would route here. Today no decoder example uses explicit `save_residual` (the standard residual is folded into MHA-KV's internal step), but the gen_forward dispatch DOES emit `save_residual`/`add_residual` for archs that contain them — and the moment someone adds Pre-LN-style explicit residuals around an MHA-KV or GQA block, this walker raises (or, worse for swiglu_ffn, silently skips the size-update because the walker has no branch — the next save would record a stale `cur_size`). **Fix:** mirror the `gen_act_decls` size table: copy each branch from `gen_act_decls:1119–1166` into `residual_slots`.

3. **`gen_scratch` 2-pass batched-Linear walker omits `multi_head_attention_gqa_kv` from one of the two `first_lin` blocks** — `tools/aricode_pack.py:1409–1421`. The FIRST `first_lin` dispatch (1355–1374) DOES branch on `multi_head_attention_gqa_kv` (1367–1369). The SECOND copy of the same dispatch (1409–1421) is missing the `multi_head_attention_gqa_kv` branch entirely; only `multi_head_attention_kv` is handled. **Condition that exposes:** an arch whose first non-pseudo layer is `multi_head_attention_gqa_kv`, AND that has at least one batched Linear later (so `any_batched=True`). Currently no arch has GQA as its first layer (the prepare scripts always start with `embedding`), so this is latent. **Fix:** copy the missing `elif first_lin[0] == "multi_head_attention_gqa_kv": cur2 = first_lin[2]` branch into the second walker.

4. **`gen_scratch` 2-pass walker first block also missing `swiglu_ffn`, `layernorm`, `rmsnorm` first-layer branches** — `tools/aricode_pack.py:1351–1374` and `1404–1421`. The `pseudo` tuple at 1352–1354 includes `layernorm`/`rmsnorm` so they're correctly skipped past, but `swiglu_ffn` is NOT in the pseudo tuple, so a `swiglu_ffn`-first arch would fall through to `cur = 0` (1374). That gives wrong batched-Linear scratch sizing. **Condition that exposes:** an arch starting with `["swiglu_ffn", d_model, d_ffn]` followed by a Linear that would batch. No current example triggers this. **Fix:** add `elif first_lin[0] == "swiglu_ffn": cur = first_lin[1]` to BOTH walkers.

5. **`pseudo` tuple drift across the 3 occurrences** — `tools/aricode_pack.py:617`, `:1044`, `:1352`. All three currently contain the SAME 11 entries: `save_residual, add_residual, flatten, relu, sigmoid, tanh, softmax, gelu, layernorm, rmsnorm, positional_embedding`. No drift today. ⚠ However, `swiglu_ffn` is NOT in any of the three pseudo tuples even though it preserves shape (gen_act_decls treats it as size-changing because it allocates a new buffer; that's fine for `gen_act_decls` but means `residual_slots` and the `gen_scratch` walker's `first_lin` lookup will pick `swiglu_ffn` as the "first real layer" — fine if and only if every walker has the `swiglu_ffn` first-layer branch (which #4 shows is missing in `gen_scratch`).

6. **`gen_act_decls` first-layer dispatch supports `["multi_head_attention_gqa_kv", ...]` (line 1073) but the symmetric `residual_slots` does NOT** — see gap #2. Also supports `["rmsnorm", dim]` and `["swiglu_ffn", d_model, d_ffn]` first; `residual_slots` does not.

7. **First-layer dispatch in `gen_act_decls` is permissive but the error message is stale** — `tools/aricode_pack.py:1100`: `"first layer must be linear, conv2d_3x3_p1, attention, or layernorm"`. The actual whitelist now includes 9+ kinds. Cosmetic; only triggers on a malformed arch.

8. **State-dict resolver candidate keys for `multi_head_attention` lack the `model.layers.{i}.self_attn.{proj}_proj` HF-Llama form** — `tools/aricode_pack.py:2635–2644`. The MHA branch covers `q_proj`, `attn.q_proj`, `attention.q_proj`, `layers.{mi}.attn.q_proj`, `layers.{mi}.q_proj`, and the two distilbert `transformer.layer.{i}.attention.{q_lin}` variants — but NOT `model.layers.{i}.self_attn.q_proj`. The **GQA** branch (2727) has the `model.layers.` prefix; the plain MHA branch does not. **Condition that exposes:** packing a Llama-style FORWARD-pass (non-KV) MHA from a HF Llama checkpoint. Llama always uses MHA-KV in practice, so this is latent. **Fix:** add the `model.layers.{mi}.self_attn.{hf}.weight` and `transformer.h.{mi}.attn.c_attn.weight` (GPT-2) candidates. Symmetrically, the **MHA-KV** branch (2680) is also missing the `model.layers.{i}.self_attn.{proj}` and GPT-2's `transformer.h.{i}.attn.c_attn` fused-QKV form.

9. **Counter sharing between MHA and MHA-KV is documented but not enforced** — `tensor_names:888–894` says the names are deliberately identical. `weight_tensors:958–970` and the resolver in `main:2620, 2665` BOTH use `mi` and increment it. ⚠ But: if an arch interleaves `multi_head_attention` and `multi_head_attention_kv` (an unusual but valid configuration: prefill with MHA then decode with MHA-KV using the same weights), the resolver tries the SAME candidate-key set twice with the same `mi` — fine for tensor binding, but the second pass would read tensors already in `collected` redundantly. No correctness bug, just wasted I/O. Latent. Documented at 893–894 explicitly: *"a runtime requirement we don't currently exercise."*

10. **`gen_forward.embedding` path raises `SystemExit` if `cur_a != 0`** — `tools/aricode_pack.py:1866`. This means an arch like `["embedding", ...] + ["embedding", ...]` (a stacked embedding) would be rejected. The error message references `ei` which IS in scope. Correct behavior, just noting it.

## Test suite status (23/23 passing)

All requested tests pass:

- 12 example `run_test.sh`: mha_causal_min, tiny_decoder_min, tiny_decoder_packed, tiny_decoder_2block, rmsnorm_min, swiglu_min, rope_min, gqa_min, rmsnorm_swiglu_decoder, tiny_llama_min, gpt2_small, layernorm_min — **all green**
- 3 example `run_test_int8.sh`: tiny_decoder_packed (token-for-token match), tiny_decoder_2block (token-for-token match), gpt2_small (coherent text, expected int8 drift, 2/16 prefix match) — **all green**
- 8 unit tests under `tests/`: test_attention_kv, test_attention_mh_kv, test_sampling, test_kv_reset, test_kv_overflow, test_rope_edges, test_sampling_edges, test_codegen_quirks — **all OK / GOLD**

For each gap, which test would catch it:

- Gap 1 (`key_stem`): no current test catches — would only fire on a packing config error. **Propose:** a negative test that points an MHA arch at a checkpoint missing `out_proj.weight` and asserts the SystemExit message contains `'o'` not a `NameError`.
- Gap 2 (`residual_slots` missing kinds): no current test catches — no example uses explicit `save_residual` around MHA-KV / GQA / SwiGLU / RMSNorm. **Propose:** an arch JSON with `["save_residual"], ["rmsnorm", 8], ["swiglu_ffn", 8, 16], ["add_residual"]` packed against a synthetic checkpoint.
- Gap 3 (second walker missing GQA-KV first): no current test — no example starts with GQA. **Propose:** unit-level Python test that invokes `gen_scratch` with an arch starting with `multi_head_attention_gqa_kv` followed by a batched Linear.
- Gap 4 (swiglu_ffn first layer): no current test — examples always start with `embedding`. **Propose:** same form as gap 3, with a `swiglu_ffn` first.
- Gap 5 (pseudo tuple drift): no test catches diff today, but adding ANY new shape-preserving op risks 1-of-3 update. **Propose:** assert in pack.py at import time that the three tuples are identical (or hoist into a module constant).
- Gap 6: subsumed by gap 2.
- Gap 7: cosmetic.
- Gap 8 (HF-Llama MHA candidate keys): no test catches — Llama uses MHA-KV, not plain MHA.
- Gap 9: documented; no observed misbehaviour.
- Gap 10: correct behaviour.

## Suggested fixes (ordered by severity)

1. **`tools/aricode_pack.py:2650, 2658`** — replace `{key_stem}` with `{proj}` (high: any MHA pack-error becomes a confusing `NameError`).
2. **`tools/aricode_pack.py:602–678`** — make `residual_slots`' first-layer + walker dispatch a 1:1 mirror of `gen_act_decls`. Add branches for `rmsnorm`, `swiglu_ffn`, `multi_head_attention_kv`, `multi_head_attention_gqa_kv`, `embedding`-as-shape-source. (medium: blocks decoder + explicit residual configurations).
3. **`tools/aricode_pack.py:1409–1421`** — add the missing `multi_head_attention_gqa_kv` branch so the second walker matches the first; add `swiglu_ffn` to BOTH walkers (1351–1374 and 1409–1421). (low-medium: only triggers on unusual first-layer choices).
4. **`tools/aricode_pack.py:1100`** — refresh the error string to list every supported first-layer kind. (cosmetic).
5. **`tools/aricode_pack.py:617, 1044, 1352`** — promote `pseudo` to a module-level `PSEUDO_OPS = (...)` constant referenced by all three sites (and by a future `assert` in `__main__`). (preventive).
6. **`tools/aricode_pack.py:2635, 2680`** — add `model.layers.{i}.self_attn.{proj}_proj.weight` (and GPT-2's `transformer.h.{i}.attn.c_attn.weight` fused form) to MHA + MHA-KV candidate lists for forward-compat with HF naming uniformity. (low: covers paths not yet exercised).
7. **No fix, just defensive comment**: 893–894 already calls out the MHA / MHA-KV `mi` aliasing constraint; consider an `assert` in `weight_tensors` that flags an arch interleaving the two kinds.

## File:line index of dispatch sites (for cross-reference)

- `tensor_specs`: 756 — every kind branch present.
- `tensor_names`: 872 — every kind branch present.
- `weight_tensors`: 924 — every kind branch present; counters: `li, ci, ai, ni, mi, gi, ei, pei, ri, si`.
- `gen_act_decls` first: 1056–1101; walk: 1103–1181.
- `gen_scratch` per-op: 1223–1323; first-walker: 1351–1374; second-walker: 1409–1421; per-Linear: 1422–1448.
- `gen_forward`: 1591–1924; counters: `li, ci, pi, ai, ni, mi, gi, ei, pei, ri, si`.
- Resolver in `main`: 2511–2966; counters: `li, ci, ai, ni, mi, gi, ei, pei, ri, si`.
- `pseudo` tuple: 617 (residual_slots), 1044 (gen_act_decls), 1352 (gen_scratch).
- `needs_kv_attention_lib`: 417; `is_decoder_arch`: 428.
- `residual_slots`: 602.
