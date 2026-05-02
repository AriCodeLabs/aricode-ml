# pack.py refactor plan

Audit target: `/home/serverbig/git-proyect/aricode-stdlib/aricode-ml/tools/aricode_pack.py` (3105 lines).

## Inventory of arch-op dispatch sites

| arch_op                        | tensor_specs | tensor_names | weight_tensors | gen_act_decls (first / body) | gen_scratch                | gen_forward | state_dict (main) |
|--------------------------------|--------------|--------------|----------------|------------------------------|----------------------------|-------------|-------------------|
| `linear`                       | 763-765      | 877-878      | 943-947        | 1056-1057 / 1104-1118        | batched _lin_in/out 1422-1434 | 1592-1603 | 2511-2516, 2532-2549 |
| `conv2d_3x3_p1`                | 766-768      | 879-880      | 948-952        | 1058-1060 / 1119-1121        | padded(_multi) 1196-1207   | 1604-1650   | 2550-2584         |
| `maxpool_2x2` (no weights)     | —            | —            | —              | — / 1122-1124                | pool_a 1224-1227           | 1651-1682   | —                 |
| `attention`                    | 769-775      | 881-882      | 953-957        | 1061-1064 / 1125-1129        | attn_desc 1228-1233        | 1719-1740   | 2585-2619         |
| `multi_head_attention`         | 776-789      | 883-887      | 958-962        | 1065-1068 / 1130-1134        | mha_desc + slices 1251-1275 | 1741-1808  | 2620-2664         |
| `multi_head_attention_kv`      | 790-803      | 888-894      | 963-970        | 1069-1072 / 1135-1140        | mhkv_state + binds 1276-1297 | 1809-1825 | 2665-2708         |
| `multi_head_attention_gqa_kv`  | 804-828      | 895-901      | 971-978        | 1073-1077 / 1141-1144        | gqa_state + binds 1298-1313 | 1826-1844  | 2709-2751         |
| `embedding`                    | 852-858      | 914-918      | 994-998        | 1078-1083 / 1145-1155        | (none) 1234-1238           | 1858-1898   | 2902-2936         |
| `positional_embedding`         | 859-867      | 919-920      | 999-1003       | (pseudo) / 1156-1159         | pe_window 1239-1250        | 1900-1911   | 2937-2966         |
| `layernorm`                    | 829-834      | 902-903      | 979-983        | 1084-1091 / 1160-1162        | (in-place)                 | 1683-1691   | 2752-2821         |
| `rmsnorm`                      | 835-840      | 904-908      | 984-988        | 1092-1095 / 1163-1165        | (in-place)                 | 1692-1698   | 2822-2862         |
| `swiglu_ffn`                   | 841-851      | 909-913      | 989-993        | 1096-1098 / 1166-1173        | _swg_gate/_up 1218-1222, 1314-1323 | 1699-1718 | 2863-2901    |

`pseudo` tuple: 3 sites — 617-619, 1044-1046, 1351-1354 (last is an
inline literal).  Per-kind counter walk: 5 sites — `weight_tensors`
924-1003, `gen_scratch` allocs 1208-1323, `gen_scratch` second pass
1408-1448, `gen_forward` 1569-1925, main() 2521-2966.  Cross-checked:
all five agree (mi shared by MHA + MHA-KV; gi distinct for GQA-KV).

## Duplication assessment

Three real, recurrence-prone duplications:

1. **Per-kind counter walk × 5.**  Adding an arch op forces a new
   `<op>i += 1` branch in every walker.  Past bug: `multi_head_
   attention_kv` was added to `weight_tensors` + `gen_forward` but
   missed in `gen_act_decls`, producing shape mismatches.

2. **`pseudo` tuple × 3** (617, 1044, 1351).  Adding `rmsnorm`
   required identical edits at all three; was fixed via global `sed`.

3. **First-layer size dispatch × 4** (`gen_act_decls` 1056-1101,
   `residual_slots` 623-639, `gen_scratch` first_lin 1355-1376,
   `gen_scratch` cur2 1409-1421).  When GQA-KV was added, `gen_scratch`'s
   first_lin block omitted it; tests passed only because GQA examples
   start with embedding (→ embedding branch supplied cur).  An arch
   leading with GQA-KV would have crashed.

Out of scope: `tensor_specs` / `tensor_names` (self-contained, never
broke); `gen_forward` op bodies (genuinely op-specific); imports/
naming (taste).

## Proposed refactors (smallest first)

### Refactor 1 — Module-level `PSEUDO_OPS` constant
- **Replaces** lines 617-619, 1044-1046, 1351-1354.
- **New abstraction**: one constant near imports (~line 64):
  ```python
  PSEUDO_OPS = ("save_residual", "add_residual", "flatten",
                "relu", "sigmoid", "tanh", "softmax", "gelu",
                "layernorm", "rmsnorm", "positional_embedding")
  ```
  Each call-site becomes `if l[0] not in PSEUDO_OPS`.
- **LOC delta**: −9 / +5 = **net −4**.
- **Test plan**: re-run all 16 example `run_test.sh` and 8
  `tests/test_*` scripts; gate is byte-identical `.ari` / `.f32`
  output. Critical: `transformer_block_min`, `distilbert_2block_min`,
  `rmsnorm_swiglu_decoder`, `tiny_decoder_2block`, `residual_min`.
- **Risk**: zero — every site already binds the same literal.

### Refactor 2 — `first_size_for(layer)` helper
- **Replaces** lines 623-639 (`residual_slots`), 1056-1101
  (`gen_act_decls`), 1355-1376 + 1409-1421 (`gen_scratch`).
- **New abstraction** (right after `tensor_specs`):
  ```python
  def first_size_for(layer):
      """sizes[0] for the given first non-pseudo arch entry."""
      kind = layer[0]
      if kind == "linear":               return layer[1]
      if kind == "conv2d_3x3_p1":        return layer[1] * 28 * 28
      if kind == "attention":            return layer[1] * layer[2]
      if kind == "multi_head_attention": return layer[1] * layer[2]
      if kind == "multi_head_attention_kv":     return layer[2]
      if kind == "multi_head_attention_gqa_kv": return layer[2]
      if kind == "embedding":            return layer[3] * layer[2]
      if kind in ("layernorm","rmsnorm","swiglu_ffn"): return layer[1]
      raise ValueError(f"first_size_for: no rule for {kind!r}")
  ```
  Each call-site collapses to:
  ```python
  first = next((l for l in arch if l[0] not in PSEUDO_OPS), arch[0])
  cur_size = first_size_for(first)
  ```
- **LOC delta**: −97 / +18 = **net −79**.
- **Test plan**: same byte-diff gate; focus on first-kind diversity
  — `mnist` (conv first), `embedding_min` / `embedding_2byte_min`,
  `layernorm_min` / `rmsnorm_min` (norm-first), `gqa_min` / `tiny_
  llama_min`, `tests/test_dense.ari` (linear first).
- **Risk** (low-medium): `gen_scratch`'s 1373-1374 silent
  `cur2 = 0` fallback becomes a raise — strictly tighter, acceptable
  (the silent path crashed at runtime anyway).  Each helper branch
  must match what its four call sites used.

### Refactor 3 — `arch_walk(arch)` shared iterator
- **Replaces** counter init blocks + `<op>i += 1` increments in:
  `weight_tensors` 924-1003 (becomes a thin adapter); `gen_scratch`
  1208-1323 + second pass 1408-1448; `gen_forward` 1569-1925; main()
  2521-2966.
- **New abstraction**:
  ```python
  def arch_walk(arch):
      """Yield (kind, idx, args).  idx is the per-kind running
      counter; pseudo-ops yield idx=None.  mha_shared is shared by
      multi_head_attention and multi_head_attention_kv (so a state
      dict keyed for one form loads into the other); GQA-KV uses a
      distinct counter."""
      counters = {"linear":0,"conv2d_3x3_p1":0,"attention":0,
                  "mha_shared":0,"multi_head_attention_gqa_kv":0,
                  "layernorm":0,"rmsnorm":0,"swiglu_ffn":0,
                  "embedding":0,"positional_embedding":0}
      key = {"linear":"linear","conv2d_3x3_p1":"conv2d_3x3_p1",
             "attention":"attention",
             "multi_head_attention":"mha_shared",
             "multi_head_attention_kv":"mha_shared",
             "multi_head_attention_gqa_kv":"multi_head_attention_gqa_kv",
             "layernorm":"layernorm","rmsnorm":"rmsnorm",
             "swiglu_ffn":"swiglu_ffn","embedding":"embedding",
             "positional_embedding":"positional_embedding"}
      for kind, *args in arch:
          if kind in key:
              k = key[kind]; idx = counters[k]
              yield (kind, idx, args); counters[k] += 1
          else:
              yield (kind, None, args)
  ```
  `weight_tensors` collapses to ~10 lines:
  ```python
  def weight_tensors(arch):
      for kind, idx, args in arch_walk(arch):
          if idx is None: continue
          for suffix, nw, nb in tensor_specs(kind, *args):
              wname, bname = tensor_names(kind, idx, suffix)
              yield (kind, idx, suffix, nw, nb, wname, bname)
  ```
- **LOC delta**: −180 / +35 = **net −145**.
- **Test plan**: byte-diff gate against pre-refactor `.ari` snapshots
  for all 24 scripts.  Counter-sensitive examples that MUST diff to
  zero: `transformer_block_min`, `distilbert_2block_min`
  (mi-shared + ni-interleaved), `tiny_decoder_2block` (MHA-KV mi
  shared), `gqa_min` (gi distinct), `rmsnorm_swiglu_decoder` (ri+si
  interleaved).
- **Risk** (medium): yield-then-increment ordering must match
  today's "yield with current idx, then bump" pattern — a flip
  shifts every weight-tensor name by +1 (silent state-dict miss at
  runtime).  The mha_shared aliasing is critical; missing it
  resurrects the most subtle past bug.  `gen_scratch`'s
  `any_batched` gate must survive the rewrite or batched-Linear
  examples (distilbert) silently lose `_lin_in_*` scratch.

## Order of attack

1. **Refactor 1** (`PSEUDO_OPS`) — trivial, zero risk.
2. **Refactor 2** (`first_size_for`) — independent of #3.
3. **Refactor 3** (`arch_walk`) — largest blast radius, lands on
   cleaner state from #1 + #2.

Each step is independently revertible.

## Regressions to gate on

Snapshot every `.ari` produced by running each script on `main`,
re-run on the refactor branch, `diff -r` snapshots; non-zero exit
on any difference.

Examples (`examples/<name>/run_test.sh`):
attention_min, distilbert_2block_min, distilbert_sst2,
embedding_min, embedding_2byte_min, encoder_full_min, gelu_min,
gpt2_small, gqa_min, layernorm_min, mha_causal_min, mnist,
multi_head_min, positional_min, residual_min, rmsnorm_min,
rmsnorm_swiglu_decoder, rope_min, swiglu_min, tiny_decoder_min,
tiny_decoder_2block, tiny_decoder_packed, tiny_llama_min,
transformer_block_min.

Tests (`tests/test_*`): test_attention, test_attention_kv,
test_attention_mh_kv, test_codegen_quirks, test_kv_overflow,
test_kv_reset, test_rope_edges, test_sampling, test_sampling_edges.
