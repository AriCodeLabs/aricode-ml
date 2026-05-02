# Track A stdlib code review

## Files reviewed
- `attention_kv_f32.ari` (164 lines) — single-head incremental KV decode
- `attention_mh_kv_f32.ari` (142 lines) — multi-head wrapper around kv
- `attention_gqa_kv_f32.ari` (271 lines) — grouped-query + RoPE
- `rope_f32.ari` (77 lines) — RoPE table + in-place rotation
- `sampling_f32.ari` (124 lines) — greedy / temperature / top-k
- `attention_f32.ari` (174 lines, pre-Track-A, cross-compare only)

## Critical issues (would cause incorrect output)
None found. All slot indices align between alloc / step / reset
within each file, and scores-tail poisoning + `arr_f32_matvec_T` truncation
to `n` rows correctly isolates the active prefix.

## Important issues (correctness boundaries)

- **`sampling_f32.ari:31` — `sample_temperature_f32` divides by
  `temperature` with no `temperature == 0` guard.** Caller passing
  `temperature=0.0` (a common UX shorthand for "greedy") gets `inv_t = inf`
  and the softmax collapses to NaN. Recommend either reject `temperature ≤ 0`
  or branch to `sample_greedy_f32` when `temperature < 1e-6`.

- **`sampling_f32.ari:81` — `sample_topk_filter_f32` k=0 path.** Loop
  `while (pass < k)` with `k=0` skips entirely, leaving `kth = logits[0]`
  from the init at line 89; everything below `logits[0]` then gets poisoned.
  Result is unpredictable (depends on logits[0]'s rank). Recommend
  early-return on `k <= 0`.

- **`sampling_f32.ari:97-110` — k-th largest by repeated max with strict-less
  filter (`v < prev`).** Ties are dropped: if the top-k contains duplicate
  logits (extremely common right after an FP layer collapse), some passes
  produce `cur_max = -1e30` which then becomes `prev` and the next pass
  early-aborts. The kth result will be wrong, and downstream the topk filter
  poisons too aggressively.

- **`sampling_f32.ari:43-49` — softmax max-find inside
  `sample_temperature_f32`** assumes `n >= 1`. `arr_len(logits) == 0`
  reads index 0 OOB. Same in `sample_greedy_f32:12`.

- **`attention_kv_f32.ari:113` — `count >= max_seq` returns `-1` but
  caller never checks.** Both `attn_mh_kv_f32.ari:134` and
  `attention_gqa_kv_f32.ari` do not propagate the failure. A model run that
  exceeds `max_seq` silently no-ops the step and emits stale `out`. Recommend
  bubble the return code or assert in step.

- **`attention_gqa_kv_f32.ari:188` — same overflow check, same silent
  failure path.** `attn_gqa_kv_step_f32` returns `-1` but writes nothing
  to `out`, leaving caller-visible buffer untouched.

- **`attention_mh_kv_f32.ari:55` — `kv.attn_kv_alloc_f32(max_seq, d_model, d_head)`
  per head allocates a fresh `zero_bias[max_seq]` for *every* head**
  (slot 17 inside the kv state). For `n_heads = 32, max_seq = 4096`, that's
  32 redundant 16KB f32 buffers. Not incorrect, but wasteful — could share
  one. Same waste in GQA (`n_kv_heads` instead of `n_heads`).

- **`attention_gqa_kv_f32.ari:209-214` — `b_K != 0` / `b_V != 0` test.**
  When the bias slot is bound to a real (non-NULL) buffer, the slice copy
  uses `kh * d_head` as src offset assuming the bias is a flat
  `[n_kv_heads * d_head]` array. If a caller mis-sizes b_K to `[d_model]`
  (full Q-style bias), the slice silently wraps onto neighbouring KV heads.
  Document the assumed bias layout near alloc.

- **`attention_kv_f32.ari:144` / `attention_gqa_kv_f32.ari:154` — score-tail
  poison value `-60.0` is a magic number.** As the review prompt notes,
  the scaled scores `Q · K / sqrt(d_head)` for typical d_head (4..128) sit
  comfortably in `[-50, 50]`, so `-60 - max` ≈ `-110`, exp ≈ `1e-48`. Safe
  for typical decoders, but undocumented at the call site. Recommend a
  named constant `SCORE_TAIL_SENTINEL` with a one-line justification.

- **`rope_f32.ari:43` — `0.0 - 2.0 * int_to_float(i) / int_to_float(d_head)`.**
  Operator precedence: `0.0 - (2.0 * i / d_head)` is what's intended, and
  C-style left-to-right evaluation here happens to give it. Worth wrapping
  in parens for clarity given the unusual `0.0 -` unary-negate idiom (used
  because there's no unary minus in the surface lang per code patterns here).

- **`rope_f32.ari:34` — `d_head must be even` is in the doc but not asserted.**
  Odd `d_head` silently rounds `half = d_head/2` and leaves `vec[d_head-1]`
  un-rotated. The rest of the pipeline assumes a full rotation.

## Style / consistency improvements

- **`attention_kv_f32.ari:40` allocates `arr_new(20)` but only uses slots 0..17.**
  Slots 18, 19 are unused — header says "slot indices documented at alloc"
  but doesn't note these as reserved. Either shrink to 18 or document.

- **`attention_mh_kv_f32.ari:42` — `arr_new(24)` but only uses 0..22.**
  Slot 23 unused, undocumented.

- **`attention_gqa_kv_f32.ari:70` — `arr_new(30)` but only uses 0..26.**
  Slots 27..29 unused, undocumented. Header at line 33 claims "30 slots"
  matching alloc size but enumerates only through slot 26.

- **`attention_kv_f32.ari:19` (header)** says caller binds slot 12 (`out`),
  matching the step body. But `attention_mh_kv_f32.ari:50` rebinds slot 12
  *inside* the kv state to `head_out` mid-step (line 132). The aliasing —
  the parent struct's slot 12 is NOT a kv state's slot 12 — is confusing on
  first read. Consider adding a comment "note: kv state slot 12 is an
  internal output binding rebound per-head; not the same as parent slot 12".

- **`attention_kv_f32.ari` body order is alloc → reset → step.**
  `attention_mh_kv_f32.ari` body order is alloc → reset → step. **Good consistency.**
  `attention_gqa_kv_f32.ari` order is alloc → reset → `_gqa_score_head` → step.
  Minor: helper `_gqa_score_head` is between reset and step; readers expect
  helpers either above alloc or below step.

- **`attention_gqa_kv_f32.ari:33` (header doc)** lists slots 0..26 explicitly.
  Excellent. **`attention_mh_kv_f32.ari:21` does the same.** **`attention_kv_f32.ari`
  scatters slot semantics across alloc+step comments rather than collecting
  them in the header.** Reader of `attn_kv_step_f32` must scroll up to alloc
  to find slot 17's purpose. Promote alloc's inline slot comments to a
  header table mirroring the MH/GQA convention.

- **Magic constant `1.0e30` in `sampling_f32.ari:95,98`** for "infinity".
  `1.0e30` is fine for f32 (max ≈ 3.4e38) but could be a named constant
  like `POS_INF_SENTINEL` / `NEG_INF_SENTINEL`. Same with `1.0e6` at line 114.

- **`sampling_f32.ari:81-110`** O(n*k) k-th largest is documented as
  intentional ("avoids allocator dependency"), but the algorithm name
  ("repeated max with strict-less filter") is non-standard — call it out
  as an inline comment so future maintainers don't try to "fix" it to
  a partial-sort.

- **`rope_f32.ari` import block: none.** Good — no needless imports.

- **`attention_mh_kv_f32.ari:39` imports `attention_kv_f32.ari` as `kv`.**
  Used. **`attention_gqa_kv_f32.ari:61` imports both.** Used. Both clean.

- **Naming inconsistency: `attention_kv_f32.ari` exposes
  `attn_kv_alloc_f32 / attn_kv_step_f32 / attn_kv_reset` — the reset is
  missing the `_f32` suffix.** `attn_mh_kv_reset` / `attn_gqa_kv_reset`
  inherit the same inconsistency. Either drop `_f32` from all three (it's
  the file suffix, not the function suffix) or add it to all reset
  functions for parity.

## Dead code candidates

- **`attention_kv_f32.ari:40-69` slots 18 and 19 of `arr_new(20)`** —
  allocated, never written, never read. Either shrink to `arr_new(18)`
  or document as reserved.

- **`attention_mh_kv_f32.ari:42` slot 23** — allocated, never used.

- **`attention_gqa_kv_f32.ari:70` slots 27, 28, 29** — allocated, never
  used. Header says "30 slots" but the layout enum stops at 26.

- **`attention_gqa_kv_f32.ari:55` `bO_zero` (slot 26)** — only read when
  `b_O == 0` (line 264). If GQA callers always pass real biases (Llama
  drops them, but caller may bind a zero-buffer themselves), the
  pre-allocated bO_zero is dead. Conservative; keep.

- **`sampling_f32.ari:90-91`** — `kth_count` is declared and never read
  or written again. Dead variable. Removed.

- **`sampling_f32.ari:89` `kth = arr_f32_get(logits, 0)`** — overwritten
  on every loop iteration before any use. Initial assignment is dead;
  could be `let kth: f64 = 0.0;`.

- **`attention_kv_f32.ari:39-44` six `arr_set(s, n, 0)` lines for
  weight slots**. The default state of a fresh `arr_new` is implementation-
  defined; if it's already zero-init, these are redundant. If it's
  garbage-init, they're necessary. Verify and either drop or document.

## Cross-kernel observations

- **Slot 0 = X is consistent across all three kv attention files.**
  Good convention.

- **"out" slot is NOT consistent: `attn_kv_f32` slot 12, `attn_mh_kv_f32`
  slot 9, `attn_gqa_kv_f32` slot 9.** The single-head `_kv` chose slot 12
  because it placed scratches 7..11 between weights and out; the
  multi-head wrappers compacted to slot 9. If a generic decoder driver
  wants to swap kernels under a uniform `arr_get(s, OUT_SLOT)` interface,
  this asymmetry breaks it. Consider standardising "out is always slot 9"
  (rearrange single-head) or document the divergence in `llama_plan.md`.

- **`count` slot: `attn_kv_f32` slot 16, `attn_gqa_kv_f32` slot 25,
  `attn_mh_kv_f32` has no top-level `count` (counts live inside per-head
  kv states).** The asymmetry is structurally justified (MH delegates to
  kv, GQA can't because it injects RoPE between project and write), but
  it means the GQA `count` and the kv-state per-head counts must be
  kept in lock-step — and `attn_gqa_kv_step_f32` does *not* increment the
  per-kv-head counts (only the parent slot 25). Yet `_gqa_score_head:151`
  reads `count` from the parent (passed-in arg). This works but means the
  per-kv-head `count` slots stay frozen at 0 forever — confusing if a
  reader expects them to track. Worth a comment at GQA alloc clarifying
  "kv-state count slot is unused; GQA owns the canonical count at slot 25".

- **`attention_f32.ari` (pre-Track-A) uses descriptor with reserved slots
  20..23.** New `_kv_*` files inherit the reserved-tail pattern but
  enumerate stricter (the GQA file claims 30 slots but uses 27). Pick a
  convention: either always over-allocate by 4 for future expansion, or
  always size exactly. Mixing is confusing.

- **Bias-NULL-via-zero pattern**: only `attention_gqa_kv_f32.ari`
  uses `if (b_X != 0)` to detect missing biases. `attention_kv_f32.ari`
  and `attention_mh_kv_f32.ari` require non-NULL biases. Inconsistent
  contract — Llama-style users of the simpler kernels would have to
  pre-allocate zero buffers themselves.

- **`-60` softmax tail sentinel in `attention_kv_f32.ari:150`,
  `attention_gqa_kv_f32.ari:157`, and `attention_f32.ari:121`.** Three
  copies of the same constant with the same justification comment in
  one (`attention_f32.ari:117-120`) and a brief comment in the others.
  Promote to a single named constant exposed by `attention_f32.ari` or
  a shared util — single source of truth.

## Constants worth elevating to named bindings

- **`-60.0`** — softmax-tail mask sentinel. Found at:
  - `attention_kv_f32.ari:150`
  - `attention_gqa_kv_f32.ari:157`
  - `attention_f32.ari:121`
  Suggest: `const SOFTMAX_MASK_NEG: f64 = -60.0;`

- **`1.0e30`** — `±∞` sentinel inside top-k filter
  (`sampling_f32.ari:95,98`).
  Suggest: `const POS_INF: f64 = 1.0e30; const NEG_INF: f64 = -1.0e30;`

- **`1.0e6`** — top-k poison value (`sampling_f32.ari:114`).
  Should logically equal `-SOFTMAX_MASK_NEG` exponent-wise; right now
  it's an unrelated magic. Recommend reuse `-60.0` for consistency
  (sampling and attention end up both feeding into a softmax, no reason
  to differ).

- **`10000.0` / `500000.0`** — RoPE base theta. Already a runtime
  parameter, good. But the "Llama-2 → 10k, Llama-3 → 500k" mapping is
  buried in `rope_f32.ari:24-25` — promote to `llama_plan.md` or a
  lookup table when the model spec gets larger.

- **`d_head` divisibility / parity assumptions** — RoPE needs d_head
  even, GQA needs `n_heads % n_kv_heads == 0`. Neither is asserted.
  Add cheap guards in alloc (`if (d_head & 1) panic …`) so misuse is
  caught at alloc time, not as silent corruption.

- **Slot count totals**: `arr_new(20) / arr_new(24) / arr_new(30)`
  are scattered. If you adopt a "headroom of 4" convention, define
  `STATE_HEADROOM = 4` and write `arr_new(USED_SLOTS + STATE_HEADROOM)`.
