#!/usr/bin/env python3
"""
aricode-pack — model → static binary compiler.

Given a PyTorch state_dict and an explicit architecture spec, emit:
    <out>.ari   aricode source for the forward pass
    <out>.f32   raw little-endian f32 weights blob (in arch order)
    <out>_*.i8  per-tensor int8 staging (with --quantize int8 + --embed)

The `aric` compiler then turns <out>.ari into a static x86_64+AVX2
binary.  With --embed, weights bake into .text via embed_file (single
self-contained ELF, up to ~2 GiB inline before rel32 CALL hits its
±2 GB cap); without --embed, the binary is ~few hundred KB and loads
the .f32 sidecar at startup via file_read.  No PyTorch, no Python,
no CUDA at runtime — just libc / ld-linux.

Layer vocabulary (extending across families):
  MLP / classifier:      linear, relu, sigmoid, tanh, softmax
  CNN:                   conv2d_3x3_p1, maxpool_2x2, flatten
  Encoder transformers:  embedding, positional_embedding, layernorm,
                         attention, multi_head_attention, gelu,
                         save_residual / add_residual
  Decoder transformers:  multi_head_attention_kv (KV cache),
                         multi_head_attention_gqa_kv (GQA + RoPE),
                         rmsnorm, swiglu_ffn

Validated end-to-end (token-for-token greedy match against PyTorch):
  GPT-2-small (124M, BPE)            → 622 MB embedded ELF / 0.7 s
  TinyLlama-1.1B (Llama family)      → 4.1 GB sidecar f32 / ~6 s
  TinyLlama-1.1B int8                → 1.83 GB embedded ELF / 2.6 s
  distilbert-SST-2 (encoder)         → ~67 MB int8
  MNIST CNN / MLP                    → ~kB-scale

Architecture spec is a Python list of (kind, *args) tuples:

    ARCH = [
        ("linear", 784, 64),      # in_features, out_features
        ("relu",),
        ("linear", 64, 10),
        # The runtime adds an argmax over the final layer for
        # classification — opt out with `--no-argmax`.
    ]

Weight order in the blob matches the layer order: every Linear
emits W (out × in row-major) then b (out,) before the next layer's
weights.  state_dict keys are matched positionally — pass `--keys`
to override the default `fc{N}.weight / fc{N}.bias` pattern.

Usage:
    python aricode_pack.py \\
        --checkpoint trained.pt \\
        --arch arch.json \\
        --input-format mnist \\
        --input-images t10k-images-idx3-ubyte \\
        --input-labels t10k-labels-idx1-ubyte \\
        --n-test 10000 \\
        --out my_classifier

    aric my_classifier.ari -o my_classifier
    ./my_classifier
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None  # only needed when reading .pt; fail late and clearly.

try:
    from safetensors.torch import load_file as _safetensors_load
except ImportError:
    _safetensors_load = None  # only needed for .safetensors checkpoints.


def load_state_dict(path: str):
    """Load a state_dict from either a PyTorch .pt/.pth or a HuggingFace
    .safetensors file.  Format dispatch is by extension; both produce the
    same flat {tensor_name → tensor} dict the rest of pack consumes.

    Why support .safetensors directly: it's the de-facto deploy format
    on HuggingFace Hub.  Reading it without a torch.load round-trip
    means you can pack a HF checkpoint without ever instantiating the
    model in Python (and without touching CUDA libs in the venv)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".pt", ".pth"):
        if torch is None:
            raise SystemExit(
                "error: PyTorch is required to read .pt/.pth checkpoints; "
                "install with `pip install torch` or convert to "
                ".safetensors first.")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        return sd
    if suffix == ".safetensors":
        if _safetensors_load is None:
            raise SystemExit(
                "error: the `safetensors` package is required to read "
                ".safetensors checkpoints.  Install with "
                "`pip install safetensors`.")
        return _safetensors_load(path)
    raise SystemExit(
        f"error: unsupported checkpoint extension '{suffix}'.  "
        "Expected .pt, .pth, or .safetensors.")


# ──────────────────────────────────────────────────────────────────────
#  Layer schema — every supported layer registers (i) the weight slots
#  it expects in the state_dict, (ii) how big a scratch buffer it
#  needs, (iii) the .ari snippet that runs its forward pass.
# ──────────────────────────────────────────────────────────────────────


def linear_weights(idx: int, in_f: int, out_f: int, sd, key_pattern: str):
    w_key = key_pattern.format(idx=idx, kind="weight")
    b_key = key_pattern.format(idx=idx, kind="bias")
    if w_key not in sd or b_key not in sd:
        raise KeyError(
            f"linear layer #{idx}: missing key '{w_key}' or '{b_key}' "
            f"in state_dict.  Pass --keys to customise."
        )
    W = sd[w_key].detach().float().cpu().numpy()
    b = sd[b_key].detach().float().cpu().numpy()
    if W.shape != (out_f, in_f) or b.shape != (out_f,):
        raise ValueError(
            f"linear #{idx}: shape mismatch.  expected W=({out_f},{in_f}) "
            f"b=({out_f},), got W={tuple(W.shape)} b={tuple(b.shape)}."
        )
    return [W, b]


def emit_linear(idx, in_f, out_f, src_var, dst_var, w_var, b_var,
                quant_scale=None, seq=1, bits=8):
    """Emit a Linear-layer forward.

    `seq` defaults to 1 — the original single-row matvec form used by
    classifier heads, MLPs, and post-flatten CNN classifiers.  When
    `seq > 1` (the transformer-block case where the input is
    `[seq, in_f]` flat after an attention block), emit a loop that
    applies the matvec per row, writing into the matching `[seq, out_f]`
    flat output.  Per-Linear scratch buffers `_lin_in_{idx}` /
    `_lin_out_{idx}` are allocated by gen_scratch — one pair per Linear,
    sized exactly to that Linear's in_f / out_f.  Per-Linear (not
    shared-max) sizing is required for the int8 path: arr_i8_matvec_f32
    derives `n` from x's length header at [x-8], so a buffer sized
    larger than the actual input would have the kernel try to read
    past the W weights and trip the bounds check.

    `quant_scale` is None for the f32 path (single arr_f32_matvec call)
    or non-None when the weights are int8.  Linear uses the per-row
    int8 path (arr_i8_matvec_f32_perrow + an f32 scales buffer named
    `{w_var}_scales` allocated by gen_load); the scales-buffer name is
    derived from w_var so callers don't need to thread it explicitly.
    """
    if seq <= 1:
        if quant_scale is None:
            return [
                f"    arr_f32_matvec({w_var}, {src_var}, {b_var}, {dst_var}, {out_f}, {in_f});",
            ]
        if bits == 4:
            # Inline int4 matmul.  Walks the packed byte buffer:
            #   byte_at({w_var}, j * in_f/2 + i) holds two nibbles.
            # Sign-extend each nibble via ((n + 8) % 16) - 8, multiply
            # by scale[j] AT THE END (one mul per row instead of per
            # element), add bias.  Pure inline f32 accumulator — no
            # SIMD kernel — so it's slower per element than the int8
            # path but the weights occupy half the bytes.
            half = in_f // 2
            return [
                f"    // int4 inline matmul (linear #{idx}): "
                f"in_f={in_f} out_f={out_f}",
                f"    let _lmj_{idx}: i32 = 0;",
                f"    while (_lmj_{idx} < {out_f}) {{",
                f"        let _scl_{idx}: f64 = "
                f"arr_f32_get({w_var}_scales, _lmj_{idx});",
                f"        let _acc_{idx}: f64 = 0.0;",
                f"        let _lmi_{idx}: i32 = 0;",
                f"        let _roff_{idx}: i32 = _lmj_{idx} * {half};",
                f"        while (_lmi_{idx} < {half}) {{",
                f"            let _raw_{idx}: i32 = "
                f"byte_at({w_var}, _roff_{idx} + _lmi_{idx});",
                f"            let _lo_{idx}: i32 = _raw_{idx} % 16;",
                f"            let _hi_{idx}: i32 = _raw_{idx} / 16;",
                f"            let _los_{idx}: i32 = "
                f"(_lo_{idx} + 8) % 16 - 8;",
                f"            let _his_{idx}: i32 = "
                f"(_hi_{idx} + 8) % 16 - 8;",
                f"            _acc_{idx} = _acc_{idx} + "
                f"int_to_float(_los_{idx}) * "
                f"arr_f32_get({src_var}, _lmi_{idx} * 2);",
                f"            _acc_{idx} = _acc_{idx} + "
                f"int_to_float(_his_{idx}) * "
                f"arr_f32_get({src_var}, _lmi_{idx} * 2 + 1);",
                f"            _lmi_{idx} = _lmi_{idx} + 1;",
                f"        }}",
                f"        arr_f32_set({dst_var}, _lmj_{idx}, "
                f"_acc_{idx} * _scl_{idx} + "
                f"arr_f32_get({b_var}, _lmj_{idx}));",
                f"        _lmj_{idx} = _lmj_{idx} + 1;",
                f"    }}",
            ]
        return [
            f"    arr_i8_matvec_f32_perrow({w_var}, {src_var}, {dst_var}, {out_f}, {w_var}_scales);",
            f"    arr_f32_add_scaled({dst_var}, {b_var}, 1.0);",
        ]
    # Batched: loop seq times, slicing [in_f] rows of src into the
    # per-Linear scratch and writing [out_f] rows of dst from it.
    # The scratch is allocated by gen_scratch to exactly in_f / out_f
    # so the int8 matvec's implicit-n-from-header math is correct.
    si = f"_lin_in_{idx}"
    so = f"_lin_out_{idx}"
    if quant_scale is None:
        body = [
            f"    let _li{idx}: i32 = 0;",
            f"    while (_li{idx} < {seq}) {{",
            f"        arr_f32_copy_slice({src_var}, _li{idx} * {in_f}, {si}, 0, {in_f});",
            f"        arr_f32_matvec({w_var}, {si}, {b_var}, {so}, {out_f}, {in_f});",
            f"        arr_f32_copy_slice({so}, 0, {dst_var}, _li{idx} * {out_f}, {out_f});",
            f"        _li{idx} = _li{idx} + 1;",
            f"    }}",
        ]
    else:
        body = [
            f"    let _li{idx}: i32 = 0;",
            f"    while (_li{idx} < {seq}) {{",
            f"        arr_f32_copy_slice({src_var}, _li{idx} * {in_f}, {si}, 0, {in_f});",
            f"        arr_i8_matvec_f32_perrow({w_var}, {si}, {so}, {out_f}, {w_var}_scales);",
            f"        arr_f32_add_scaled({so}, {b_var}, 1.0);",
            f"        arr_f32_copy_slice({so}, 0, {dst_var}, _li{idx} * {out_f}, {out_f});",
            f"        _li{idx} = _li{idx} + 1;",
            f"    }}",
        ]
    return body


def emit_relu(var):
    return [f"    arr_f32_relu({var});"]


def emit_sigmoid(var):
    return [f"    arr_f32_sigmoid({var});"]   # f64 builtin; user must promote


def emit_tanh(var):
    return [f"    arr_f32_tanh({var});"]


def emit_softmax(var):
    return [f"    arr_f32_softmax({var});"]


def emit_gelu(var):
    return [f"    gelu_f32({var});"]


# ──────────────────────────────────────────────────────────────────────
#  Code generator: walks the arch list and emits a self-contained
#  aricode source that loads the weight blob, runs the test loop, and
#  reports accuracy.
# ──────────────────────────────────────────────────────────────────────


PROLOGUE = """\
// Generated by aricode-pack — DO NOT EDIT.
//
// model: {model_name}
// arch:  {arch_repr}
// loader: {loader}
// quantization: {quantize}
// (run aricode_pack.py to regenerate after retraining.)

fn arr_f32_from_file_into(fd: i32, buf: i32, n: i32) -> i32 {{
    file_read(fd, buf, n * 4);
    return 0;
}}"""

# Emitted only when the generated main actually calls it — i.e. when
# packing is `--quantize int8` AND the arch has a multi-channel conv
# layer AND the multi-channel int8 path is configured to dequant-at-
# startup (default for batch loaders).  Single-shot CLI builds keep
# weights int8 in RAM and never call this, so leaving it out of the
# prelude shrinks the binary by ~120 B.
DEQUANT_HELPER = """
// Dequantise n int8 bytes from `q` into f32 buffer `dst`, multiplying by
// `scale`.  arr_i8_get does the sign-extend in hardware (movsx); the
// loop is the unchanged scalar pattern used when the binary was packed
// with `--quantize int8` and the multi-channel conv layer has chosen
// the dequant-at-startup path (--input-format mnist; the stdin path
// keeps weights int8 in RAM).  Runs once per weight tensor at startup.
fn dequant_int8_to_f32(q: i32, dst: i32, n: i32, scale: f64) -> i32 {
    let i: i32 = 0;
    while (i < n) {
        arr_f32_set(dst, i, int_to_float(arr_i8_get(q, i)) * scale);
        i = i + 1;
    }
    return 0;
}"""


# Inlined verbatim from aricode-stdlib/aricode-ml/attention_f32.ari.
# Emitted only when the arch contains an attention layer; pure aricode
# (no extra builtins beyond the matvec/softmax/scale/copy_slice/set
# already used elsewhere in packed binaries), so the .ari output stays
# self-contained — no `use ... from` import needed at compile time.
#
# Keep this in sync with the source; the scoreboard is the test in
# examples/attention_min/ which compares packed-binary output against
# a PyTorch reference.
ATTENTION_LIB = """
fn attn_alloc_desc_f32(seq: i32, d_in: i32, d_head: i32, causal: i32) -> i32 {
    let d: i32 = arr_new(24);
    arr_set(d,  7, arr_f32_new(seq * d_head));
    arr_set(d,  8, arr_f32_new(seq * d_head));
    arr_set(d,  9, arr_f32_new(seq * d_head));
    arr_set(d, 10, arr_f32_new(seq * seq));
    arr_set(d, 12, arr_f32_new(d_in));
    arr_set(d, 13, arr_f32_new(d_head));
    arr_set(d, 14, arr_f32_new(seq));
    let zb: i32 = arr_f32_new(seq);
    arr_f32_fill(zb, 0.0);
    arr_set(d, 15, zb);
    arr_set(d, 16, seq);
    arr_set(d, 17, d_in);
    arr_set(d, 18, d_head);
    arr_set(d, 19, causal);
    return d;
}

fn attn_project_qkv_f32(desc: i32) -> i32 {
    let X: i32      = arr_get(desc,  0);
    let W_Q: i32    = arr_get(desc,  1);  let b_Q: i32 = arr_get(desc,  2);
    let W_K: i32    = arr_get(desc,  3);  let b_K: i32 = arr_get(desc,  4);
    let W_V: i32    = arr_get(desc,  5);  let b_V: i32 = arr_get(desc,  6);
    let Q: i32      = arr_get(desc,  7);
    let K: i32      = arr_get(desc,  8);
    let V: i32      = arr_get(desc,  9);
    let row_in: i32 = arr_get(desc, 12);
    let row_qkv: i32= arr_get(desc, 13);
    let seq: i32    = arr_get(desc, 16);
    let d_in: i32   = arr_get(desc, 17);
    let d_head: i32 = arr_get(desc, 18);
    let i: i32 = 0;
    while (i < seq) {
        arr_f32_copy_slice(X, i * d_in, row_in, 0, d_in);
        arr_f32_matvec(W_Q, row_in, b_Q, row_qkv, d_head, d_in);
        arr_f32_copy_slice(row_qkv, 0, Q, i * d_head, d_head);
        arr_f32_matvec(W_K, row_in, b_K, row_qkv, d_head, d_in);
        arr_f32_copy_slice(row_qkv, 0, K, i * d_head, d_head);
        arr_f32_matvec(W_V, row_in, b_V, row_qkv, d_head, d_in);
        arr_f32_copy_slice(row_qkv, 0, V, i * d_head, d_head);
        i = i + 1;
    }
    return 0;
}

fn attn_scores_softmax_f32(desc: i32) -> i32 {
    let Q: i32         = arr_get(desc,  7);
    let K: i32         = arr_get(desc,  8);
    let scores: i32    = arr_get(desc, 10);
    let row_qkv: i32   = arr_get(desc, 13);
    let row_score: i32 = arr_get(desc, 14);
    let zero_bias: i32 = arr_get(desc, 15);
    let seq: i32       = arr_get(desc, 16);
    let d_head: i32    = arr_get(desc, 18);
    let causal: i32    = arr_get(desc, 19);
    let inv_sqrt_dh: f64 = 1.0 / math_sqrt(int_to_float(d_head));
    let neg_big: f64 = 0.0 - 60.0;
    let i: i32 = 0;
    while (i < seq) {
        arr_f32_copy_slice(Q, i * d_head, row_qkv, 0, d_head);
        arr_f32_matvec(K, row_qkv, zero_bias, row_score, seq, d_head);
        arr_f32_scale(row_score, inv_sqrt_dh);
        if (causal != 0) {
            let j: i32 = i + 1;
            while (j < seq) {
                arr_f32_set(row_score, j, neg_big);
                j = j + 1;
            }
        }
        arr_f32_softmax(row_score);
        arr_f32_copy_slice(row_score, 0, scores, i * seq, seq);
        i = i + 1;
    }
    return 0;
}

fn attn_combine_f32(desc: i32) -> i32 {
    let V: i32         = arr_get(desc,  9);
    let scores: i32    = arr_get(desc, 10);
    let out: i32       = arr_get(desc, 11);
    let row_qkv: i32   = arr_get(desc, 13);
    let row_score: i32 = arr_get(desc, 14);
    let seq: i32       = arr_get(desc, 16);
    let d_head: i32    = arr_get(desc, 18);
    let i: i32 = 0;
    while (i < seq) {
        arr_f32_copy_slice(scores, i * seq, row_score, 0, seq);
        arr_f32_matvec_T(V, row_score, row_qkv, seq, d_head);
        arr_f32_copy_slice(row_qkv, 0, out, i * d_head, d_head);
        i = i + 1;
    }
    return 0;
}

fn attention_forward_f32(desc: i32) -> i32 {
    attn_project_qkv_f32(desc);
    attn_scores_softmax_f32(desc);
    attn_combine_f32(desc);
    return 0;
}
"""


# Arch ops that don't change activation shape and don't carry a per-
# layer counter — `gen_act_decls`, `residual_slots`, and `gen_scratch`
# all need to peek past these to find the first size-defining layer.
# Adding a new shape-preserving op?  Append it here and every caller
# picks it up — beats the prior 3-site sed sweep that the rmsnorm
# addition required.  Note: `swiglu_ffn` is intentionally NOT pseudo
# (its FFN has a 3-Linear data path that allocates a fresh activation
# buffer and the residual_slots size walker cares about it).
PSEUDO_OPS = ("save_residual", "add_residual", "flatten",
              "relu", "sigmoid", "tanh", "softmax", "gelu",
              "layernorm", "rmsnorm", "positional_embedding")


def needs_attention_lib(arch):
    # Multi-head attention also dispatches into single-head's
    # attention_forward_f32, so the library is needed for either kind.
    # The KV variant has its own (separate) kernel; only emit the
    # one-shot lib if a one-shot attention layer is present.
    return any(layer[0] in ("attention", "multi_head_attention")
               for layer in arch)


# KV-cache + multi-head + sampling library — loaded at pack time from
# the canonical .ari sources in aricode-stdlib/aricode-ml/.  Single
# source of truth: the KV stdlib lives in aricode-ml/*.ari.  We inline
# it at pack time so generated binaries are self-contained without
# external import paths.  The .ari originals use `import "..." as kv;`
# plus namespaced calls (kv.attn_kv_step_f32, samp.sample_greedy_f32);
# we strip the imports and the `kv.` / `mhkv.` / `samp.` prefixes so
# the resulting blob is one flat module.
_NAMESPACE_PREFIX_RE = re.compile(r"\b(?:kv|mhkv|samp|rope|gqa)\.")
_IMPORT_LINE_RE = re.compile(r"^\s*import\s+\"[^\"]+\"\s+as\s+\w+\s*;\s*$")


def _strip_kv_lib_source(src: str) -> str:
    """Remove import statements and namespace prefixes from one .ari
    file, so its functions can be inlined alongside their callees."""
    out_lines = []
    for line in src.splitlines():
        if _IMPORT_LINE_RE.match(line):
            continue
        out_lines.append(_NAMESPACE_PREFIX_RE.sub("", line))
    return "\n".join(out_lines)


def _load_kv_attention_lib() -> str:
    """Read attention_kv_f32.ari, attention_mh_kv_f32.ari, rope_f32.ari,
    attention_gqa_kv_f32.ari, and sampling_f32.ari from the stdlib and
    return their concatenation with imports + namespace prefixes
    stripped.  Order matters:
      - attention_kv must precede attention_mh_kv (mhkv calls kv).
      - rope must precede attention_gqa_kv (gqa calls into rope).
      - attention_kv must precede attention_gqa_kv (gqa reuses attn_kv
        state buffers via kv.attn_kv_alloc_f32 / kv.attn_kv_reset_f32)."""
    ml_dir = Path(__file__).resolve().parent.parent
    files = [
        "attention_kv_f32.ari",
        "attention_mh_kv_f32.ari",
        "rope_f32.ari",
        "attention_gqa_kv_f32.ari",
        "sampling_f32.ari",
    ]
    parts = []
    for name in files:
        src = (ml_dir / name).read_text()
        parts.append(_strip_kv_lib_source(src))
    return "\n".join(parts)


def needs_kv_attention_lib(arch):
    """True iff any layer is the incremental KV-cached variant.  When
    set, we import attention_mh_kv_f32 (which transitively imports
    attention_kv_f32) and the sampling primitives in the prologue.
    Also fires for the GQA variant since it shares the KV state machinery
    and uses RoPE (both bundled into the same prologue blob)."""
    return any(layer[0] in ("multi_head_attention_kv",
                            "multi_head_attention_gqa_kv")
               for layer in arch)


def is_decoder_arch(arch):
    """A decoder arch is one that contains at least one MHA-KV layer.
    These archs run in single-token mode end-to-end, so every layer
    sees a [d_model] activation rather than [seq, d_model], and main()
    wraps the forward in an autoregressive token loop."""
    return needs_kv_attention_lib(arch)


def _embedding_token_bytes(vocab_size):
    """How many bytes per token does the input loader / embedding
    forward consume?  Auto-sized from vocab_size — 1 byte for tiny
    demo models (≤256), 2 bytes for typical HF transformer vocabs
    (BERT/distilbert: 30522), 4 bytes for the largest (LLaMA: 32K
    fits 2-byte; Llama-3 128K still fits; only specialty vocabs
    (e.g. Whisper: 51865 — fits 2-byte) push past).  Most real
    transformers will use the 2-byte path."""
    if vocab_size <= 256:
        return 1
    if vocab_size <= 65536:
        return 2
    return 4


# Affine LayerNorm: arr_f32_layernorm normalises in-place but doesn't
# apply the learnable scale (γ) and shift (β) every real transformer
# uses.  This helper does the affine pass after the builtin normalises.
# Scalar inner loop is acceptable at d_model = 64..512 (a typical
# distilbert d_model is 768 → ~50K ops for 64 tokens, < 25 µs).
LAYERNORM_HELPER = """
fn layernorm_affine_f32(buf: i32, dim: i32, gamma: i32, beta: i32) -> i32 {
    arr_f32_layernorm(buf, dim, 0.00001);
    let n: i32 = arr_len(buf);
    let K: i32 = n / dim;
    let k: i32 = 0;
    while (k < K) {
        let i: i32 = 0;
        while (i < dim) {
            let off: i32 = k * dim + i;
            arr_f32_set(buf, off,
                arr_f32_get(buf, off) * arr_f32_get(gamma, i) + arr_f32_get(beta, i));
            i = i + 1;
        }
        k = k + 1;
    }
    return 0;
}
"""


def needs_layernorm_helper(arch):
    return any(layer[0] == "layernorm" for layer in arch)


# GELU helper: tanh-approximation form used by HuggingFace, OpenAI's
# GPT lineage, BERT/distilbert, etc. (matches torch.nn.GELU with
# `approximate='tanh'`).  Computes
#
#   gelu(x) = 0.5 · x · (1 + tanh(c · (x + 0.044715 · x³)))
#
# with c = sqrt(2/π) ≈ 0.79788.  tanh is computed via math_exp using the
# numerically-stable tanh(z) = 1 − 2/(e^(2z) + 1) form so we don't need
# a scalar math_tanh builtin.  Scalar inner pass — fine at d_ff up to
# ~3072 where the FFN dominates wall regardless.  A vectorised
# arr_f32_gelu builtin would be a future drop-in.
GELU_HELPER = """
fn gelu_f32(buf: i32) -> i32 {
    let n: i32 = arr_len(buf);
    let i: i32 = 0;
    let c: f64 = 0.7978845608028654;
    while (i < n) {
        let x: f64 = arr_f32_get(buf, i);
        let inner: f64 = c * (x + 0.044715 * x * x * x);
        let two_inner: f64 = 2.0 * inner;
        // Saturation clamp: aricode's math_exp doesn't IEEE-saturate at
        // overflow (returns garbage for |arg| ≳ 700 instead of ±inf / 0).
        // For activations large enough to push 2*inner past ±20, tanh is
        // already within ~2e-9 of ±1, so we branch instead of touching
        // math_exp.  Without this clamp, FFN GELU silently flips its
        // saturation tail and corrupts the residual stream — the
        // distilbert_2block divergence reduces from ~6.18 to ~6e-6 with
        // this guard alone.
        let t: f64 = 0.0;
        if (two_inner > 20.0) {
            t = 1.0;
        } else {
            if (two_inner < 0.0 - 20.0) {
                t = 0.0 - 1.0;
            } else {
                let e2z: f64 = math_exp(two_inner);
                t = 1.0 - 2.0 / (e2z + 1.0);
            }
        }
        arr_f32_set(buf, i, 0.5 * x * (1.0 + t));
        i = i + 1;
    }
    return 0;
}
"""


# RMSNorm with affine γ, no bias.  Used by Llama / TinyLlama / Mistral.
# Skips the mean-subtract pass that LayerNorm does — divides by RMS
# (root-mean-square) directly.  eps 1e-5 matches HF Llama.
#
# Two paths:
#   K=1 (decoder mode, single-token activation): both passes are SIMD —
#       arr_f32_rmsnorm normalises in place, arr_f32_mul applies γ.
#   K>1 (batched / encoder-style): the rmsnorm builtin handles all K
#       groups internally; the γ pass falls back to a per-row scalar
#       loop because γ is shape (dim,) but buf is shape (K, dim).
RMSNORM_HELPER = """
fn rmsnorm_affine_f32(buf: i32, dim: i32, gamma: i32) -> i32 {
    arr_f32_rmsnorm(buf, dim, 0.00001);
    let n: i32 = arr_len(buf);
    if (n == dim) {
        arr_f32_mul(buf, buf, gamma);
        return 0;
    }
    let K: i32 = n / dim;
    let k: i32 = 0;
    while (k < K) {
        let off: i32 = k * dim;
        let i: i32 = 0;
        while (i < dim) {
            arr_f32_set(buf, off + i,
                arr_f32_get(buf, off + i) * arr_f32_get(gamma, i));
            i = i + 1;
        }
        k = k + 1;
    }
    return 0;
}
"""


def needs_rmsnorm_helper(arch):
    return any(layer[0] == "rmsnorm" for layer in arch)


# SwiGLU element-wise step: up_then_dst[i] = silu(gate[i]) * up_then_dst[i].
# Used by Llama / Mistral / Falcon FFNs after the two parallel
# projections.  Two SIMD passes:
#   arr_f32_silu(gate)              -- gate[i] = silu(gate[i])
#   arr_f32_mul(up_then_dst, gate, up_then_dst)
#                                   -- up_then_dst[i] *= gate[i]
# arr_f32_silu (codegen_builtins.c, 4-lane f64-promote-then-narrow,
# inherits the math_exp clamp from vec_exp_body) replaces a scalar
# 8-branch loop with explicit |g| > 20 saturation; arr_f32_mul is
# the existing 8-lane vmulps.  Net: ~5-10× faster on d_ffn=5632
# (TinyLlama).  `d` parameter retained for source compatibility but
# unused — both builtins read length from arr_len(buf - 8).
SWIGLU_HELPER = """
fn silu_mul_f32(gate: i32, up_then_dst: i32, d: i32) -> i32 {
    arr_f32_silu(gate);
    arr_f32_mul(up_then_dst, gate, up_then_dst);
    return 0;
}
"""


def needs_swiglu_helper(arch):
    return any(layer[0] == "swiglu_ffn" for layer in arch)


def needs_gelu_helper(arch):
    return any(layer[0] == "gelu" for layer in arch)


def first_size_for(layer):
    """Return the activation element count produced by `layer` when it
    is the first non-pseudo layer in an arch.  Consolidates the four
    duplicated dispatches in `residual_slots`, `gen_act_decls`, and the
    two `gen_scratch` `cur` trackers — every call site computed the
    same `sizes[0]` from the same first-layer tuple.

    Adding a new arch op?  Add a single branch here and the four
    callers pick it up automatically.  Past bug: `multi_head_attention_
    gqa_kv` was added to gen_act_decls but missed in gen_scratch's
    first_lin block; the four sites went out of sync silently because
    GQA arches happened to start with `embedding` (which supplied
    `cur` instead).  An arch leading with GQA-KV would have crashed.
    """
    kind = layer[0]
    if kind == "linear":
        # ["linear", in_features, out_features] — input element count is
        # in_features (single-row default; batched callers reshape).
        return layer[1]
    if kind == "conv2d_3x3_p1":
        # ["conv2d_3x3_p1", c_in, c_out] — flattened [c_in, 28, 28].
        return layer[1] * 28 * 28
    if kind == "attention":
        # ["attention", seq, d_in, d_head, causal] — flat [seq, d_in].
        return layer[1] * layer[2]
    if kind == "multi_head_attention":
        # ["multi_head_attention", seq, d_model, n_heads, causal] —
        # flat [seq, d_model].
        return layer[1] * layer[2]
    if kind == "multi_head_attention_kv":
        # ["multi_head_attention_kv", max_seq, d_model, n_heads] —
        # decoder mode operates on a single token, so the activation
        # is just d_model wide.
        return layer[2]
    if kind == "multi_head_attention_gqa_kv":
        # ["multi_head_attention_gqa_kv", max_seq, d_model, n_heads,
        #  n_kv_heads, theta] — same single-token contract as MHA-KV.
        return layer[2]
    if kind == "embedding":
        # ["embedding", vocab, d_model, seq] → output [seq, d_model].
        return layer[3] * layer[2]
    if kind == "layernorm":
        # ["layernorm", dim] — single-row default (K=1); see
        # gen_act_decls for the K>1 workaround.
        return layer[1]
    if kind == "rmsnorm":
        # ["rmsnorm", dim] — same single-row default as LayerNorm.
        return layer[1]
    if kind == "swiglu_ffn":
        # ["swiglu_ffn", d_model, d_ffn] — input dim = d_model.
        return layer[1]
    raise ValueError(f"first_size_for: no rule for first-layer kind {kind!r}")


def residual_slots(arch):
    """Walk the arch and return a list of (slot_idx, n_elems) for every
    `save_residual` entry, in source order.  The slot's element count is
    the activation size at the moment of save (= same as the matched
    add_residual, since we don't reshape between save and add).

    Stack-validates the arch as a side effect: every save must have a
    matching add later in the arch, and every add must follow some
    earlier unmatched save."""
    if not any(layer[0] in ("save_residual", "add_residual") for layer in arch):
        return []
    # We need per-position activation sizes.  Mirror gen_act_decls'
    # first-layer dispatch, including the same look-past-pseudo-ops
    # logic so an arch that opens with save_residual (the standard
    # Pre-LN transformer pattern) sizes correctly.
    first = next((l for l in arch if l[0] not in PSEUDO_OPS), None)
    if first is None:
        first = arch[0]
    cur_size = first_size_for(first)
    slots = []
    stack = []
    next_idx = 0
    for kind, *args in arch:
        if kind == "save_residual":
            slots.append((next_idx, cur_size))
            stack.append((next_idx, cur_size))
            next_idx += 1
        elif kind == "add_residual":
            if not stack:
                raise ValueError("add_residual without a matching save_residual")
            saved_idx, saved_size = stack.pop()
            if saved_size != cur_size:
                raise ValueError(
                    f"residual size mismatch: save was {saved_size} elements "
                    f"but add is at {cur_size} elements")
        elif kind == "linear":
            in_f, out_f = args
            if cur_size >= in_f and cur_size % in_f == 0:
                seq_in = cur_size // in_f
                cur_size = seq_in * out_f
            else:
                cur_size = out_f
        elif kind == "conv2d_3x3_p1":
            cur_size = args[1] * 28 * 28
        elif kind == "maxpool_2x2":
            c = args[0]
            cur_size = c * 14 * 14
        elif kind == "attention":
            seq, _, d_head, _ = args
            cur_size = seq * d_head
        elif kind == "multi_head_attention":
            seq, d_model, _, _ = args
            cur_size = seq * d_model
        elif kind == "multi_head_attention_kv":
            # Single-token decoder; output is d_model.
            cur_size = args[1]
        elif kind == "multi_head_attention_gqa_kv":
            cur_size = args[1]
        elif kind == "swiglu_ffn":
            # Output dim = d_model (first arg).
            cur_size = args[0]
        # layernorm, rmsnorm, flatten, embedding-after-first, in-place
        # activations: unchanged size.
    if stack:
        raise ValueError(
            f"{len(stack)} save_residual entries with no matching add_residual")
    return slots


# Continuation of PROLOGUE.  Split out so DEQUANT_HELPER can be slotted
# in conditionally between the two halves; load_byte_file / argmax_f32
# are always needed regardless of quantisation choices.
PROLOGUE_TAIL = """
fn load_byte_file(path_str: i32, byte_count: i32) -> i32 {
    let slots: i32 = (byte_count + 7) / 8;
    let buf: i32 = arr_new(slots);
    let fd: i32 = file_open(path_str, 0);
    file_read(fd, buf, byte_count);
    file_close(fd);
    return buf;
}

fn argmax_f32(buf: i32, n: i32) -> i32 {
    let best_i: i32 = 0;
    let best_v: f64 = arr_f32_get(buf, 0);
    let i: i32 = 1;
    while (i < n) {
        let v: f64 = arr_f32_get(buf, i);
        if (v > best_v) { best_v = v; best_i = i; }
        i = i + 1;
    }
    return best_i;
}
"""


def needs_dequant_helper(arch, quantize, input_format):
    """The dequant_int8_to_f32 helper is only called from gen_load's
    int8 + multi-channel + batch path.  Mirrors the same condition
    used in gen_load and gen_forward to choose the runtime dispatch."""
    if quantize != "int8":
        return False
    if input_format == "stdin":
        return False        # multi_int8_runtime path keeps weights int8
    return any(kind == "conv2d_3x3_p1" and a[0] > 1
               for kind, *a in arch)


MNIST_LOADER = """\
fn load_image_at(img_buf: i32, offset: i32, out: i32) -> i32 {{
    let i: i32 = 0;
    while (i < {n_in}) {{
        let px: i32 = byte_at(img_buf, offset + i);
        // Standardised the same way as the trainer.
        arr_f32_set(out, i, (int_to_float(px) / 255.0 - {mean}) / {std});
        i = i + 1;
    }}
    return 0;
}}
"""


def weighted_layers(arch):
    """Yield (kind, idx_within_kind, *args) for every layer that has its
    own weights/biases (Linear and Conv2d so far).  Used to enumerate
    the W/b naming order in both the source and the staging files."""
    li = 0   # linear index
    ci = 0   # conv index
    for kind, *args in arch:
        if kind == "linear":
            yield ("linear", li, *args)
            li += 1
        elif kind == "conv2d_3x3_p1":
            yield ("conv2d_3x3_p1", ci, *args)
            ci += 1


def linear_layers(arch):
    """Yield (idx, in_f, out_f) for every Linear in the arch order."""
    for kind, idx, *rest in weighted_layers(arch):
        if kind == "linear":
            yield idx, rest[0], rest[1]


def tensor_specs(kind, *args):
    """Yield (suffix, n_weight_elems, n_bias_elems) per (W, b) tensor pair
    inside a layer.  Single-tensor layers yield exactly one entry with
    suffix='' (so existing naming stays untouched); multi-tensor layers
    like attention yield multiple entries with disambiguating suffixes
    ('q', 'k', 'v').  Used by both the staging-file writer and the code
    generator to enumerate every weight tensor uniformly."""
    if kind == "linear":
        in_f, out_f = args
        yield ("", out_f * in_f, out_f)
    elif kind == "conv2d_3x3_p1":
        c_in, c_out = args
        yield ("", c_out * c_in * 9, c_out)
    elif kind == "attention":
        # ["attention", seq, d_in, d_head, causal]
        # Three (W, b) projections — Q, K, V — each shaped (d_head, d_in)
        # and (d_head,), matching the nn.Linear layout the kernel reads.
        seq, d_in, d_head, causal = args
        for proj in ("q", "k", "v"):
            yield (proj, d_head * d_in, d_head)
    elif kind == "multi_head_attention":
        # ["multi_head_attention", seq, d_model, n_heads, causal]
        # Four (W, b) pairs — Q/K/V projections each (d_model, d_model)
        # plus an output projection (d_model, d_model).  Matches the
        # standard HuggingFace MHA layout (q_proj/k_proj/v_proj/out_proj).
        # d_head = d_model / n_heads is computed at pack time and must
        # divide evenly.
        seq, d_model, n_heads, causal = args
        if d_model % n_heads != 0:
            raise ValueError(
                f"multi_head_attention: d_model ({d_model}) must be a "
                f"multiple of n_heads ({n_heads})")
        for proj in ("q", "k", "v", "o"):
            yield (proj, d_model * d_model, d_model)
    elif kind == "multi_head_attention_kv":
        # ["multi_head_attention_kv", max_seq, d_model, n_heads]
        # Same weight tensors as multi_head_attention (Q/K/V/O), but
        # consumed by attn_mh_kv_step_f32 (incremental, single-token,
        # causal-by-construction).  Causal flag is implicit — KV cache
        # only ever holds past tokens, so each step attends to the
        # exact upper-triangular set the one-shot causal mask defines.
        max_seq, d_model, n_heads = args
        if d_model % n_heads != 0:
            raise ValueError(
                f"multi_head_attention_kv: d_model ({d_model}) must be a "
                f"multiple of n_heads ({n_heads})")
        for proj in ("q", "k", "v", "o"):
            yield (proj, d_model * d_model, d_model)
    elif kind == "multi_head_attention_gqa_kv":
        # ["multi_head_attention_gqa_kv", max_seq, d_model, n_heads,
        #                                  n_kv_heads, theta]
        # Llama / TinyLlama / Mistral attention with GQA + RoPE.
        #   q : (d_model, d_model)
        #   k : (n_kv_heads * d_head, d_model)   ← narrower than Q
        #   v : (n_kv_heads * d_head, d_model)
        #   o : (d_model, d_model)
        # No biases (Llama convention).  d_head = d_model // n_heads.
        # n_heads must be a multiple of n_kv_heads (group_size = ratio).
        max_seq, d_model, n_heads, n_kv_heads, theta = args
        if d_model % n_heads != 0:
            raise ValueError(
                f"multi_head_attention_gqa_kv: d_model ({d_model}) must "
                f"be a multiple of n_heads ({n_heads})")
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"multi_head_attention_gqa_kv: n_heads ({n_heads}) must "
                f"be a multiple of n_kv_heads ({n_kv_heads})")
        d_head = d_model // n_heads
        kv_dim = n_kv_heads * d_head
        yield ("q", d_model * d_model, 0)
        yield ("k", kv_dim * d_model,  0)
        yield ("v", kv_dim * d_model,  0)
        yield ("o", d_model * d_model, 0)
    elif kind == "layernorm":
        # ["layernorm", dim] (eps defaults to 1e-5; or ["layernorm", dim, eps])
        # Single (γ, β) pair, both shape (dim,) — matches HuggingFace's
        # LayerNorm.weight (γ) / LayerNorm.bias (β) convention.
        dim = args[0]
        yield ("", dim, dim)
    elif kind == "rmsnorm":
        # ["rmsnorm", dim] — Llama/Mistral RMSNorm.  Single γ tensor, NO
        # β (HF stores `model.layers.{i}.input_layernorm.weight` only).
        # Yield n_bias = 0 so gen_weight_decls / gen_load skip the bias.
        dim = args[0]
        yield ("", dim, 0)
    elif kind == "swiglu_ffn":
        # ["swiglu_ffn", d_model, d_ffn] — Llama/Mistral FFN block.
        # Three weight tensors, NO biases (HF Llama drops them).  Output
        # = down(silu(gate(x)) * up(x)).
        #   gate (d_ffn, d_model)   — gate_proj
        #   up   (d_ffn, d_model)   — up_proj
        #   down (d_model, d_ffn)   — down_proj
        d_model, d_ffn = args
        yield ("gate", d_ffn * d_model, 0)
        yield ("up",   d_ffn * d_model, 0)
        yield ("down", d_model * d_ffn, 0)
    elif kind == "embedding":
        # ["embedding", vocab_size, d_model, seq]
        # nn.Embedding has only a weight tensor (no bias).  We yield
        # the bias size as 0; the consumers (gen_weight_decls, gen_load,
        # main pack staging) skip the bias path when n_bias == 0.
        vocab_size, d_model, _seq = args
        yield ("", vocab_size * d_model, 0)
    elif kind == "positional_embedding":
        # ["positional_embedding", max_pos, d_model, seq]
        # Same shape as nn.Embedding (max_pos, d_model), no bias.  HF's
        # learned positional table for BERT/distilbert.  Indexed by
        # token position 0..seq-1 (not by token ID), so the loader
        # path is different from token embedding — but the weight
        # tensor itself is identical in shape.
        max_pos, d_model, _seq = args
        yield ("", max_pos * d_model, 0)
    else:
        raise ValueError(f"tensor_specs: no weights for {kind!r}")


def tensor_names(kind, idx, suffix=""):
    """Return (wname, bname) for one (W, b) pair within a layer.
    Existing single-tensor convention preserved (W{idx}/b{idx}, Wc{idx}/
    bc{idx}); multi-tensor layers compose their suffix into the stem
    (e.g. attention yields Wq{idx}, bq{idx}, Wk{idx}, ...)."""
    if kind == "linear":
        return f"W{idx}", f"b{idx}"
    if kind == "conv2d_3x3_p1":
        return f"Wc{idx}", f"bc{idx}"
    if kind == "attention":
        return f"W{suffix}{idx}", f"b{suffix}{idx}"   # e.g. Wq0, bq0
    if kind == "multi_head_attention":
        # Distinct prefix from single-head so the same arch can host
        # both kinds without name collisions.  e.g. Wmq0/bmq0 for the
        # 0th MHA layer's Q projection.
        return f"Wm{suffix}{idx}", f"bm{suffix}{idx}"
    if kind == "multi_head_attention_kv":
        # Same weight tensor names as multi_head_attention so a static
        # checkpoint can be repurposed between prefill and decode forms
        # with no re-keying.  The kv index counter is shared with the
        # one-shot MHA counter only when both kinds appear in the same
        # arch — a runtime requirement we don't currently exercise.
        return f"Wm{suffix}{idx}", f"bm{suffix}{idx}"
    if kind == "multi_head_attention_gqa_kv":
        # Distinct prefix from MHA-KV's Wm so a single arch can host
        # both kinds (Llama-2 + GPT-2 hybrid for tests) without weight
        # tensor name collisions.  No biases (Llama drops them); the b*
        # name is reserved only for API symmetry — consumers gate on
        # n_bias==0 from tensor_specs and skip the alloc.
        return f"Wgqa{suffix}{idx}", f"bgqa{suffix}{idx}"
    if kind == "layernorm":
        return f"Wn{idx}", f"bn{idx}"   # γ=Wn (the "weight" in HF naming), β=bn
    if kind == "rmsnorm":
        # Distinct prefix from layernorm so the same arch can host both
        # without name collisions.  No bias — `br{idx}` is reserved
        # only for API symmetry; consumers gate on n_bias == 0.
        return f"Wr{idx}", f"br{idx}"
    if kind == "swiglu_ffn":
        # Three weight tensors per layer (gate / up / down) and no
        # biases.  Suffix carries the projection name; consumers gate
        # on n_bias == 0 and skip allocation of `b{suffix}{idx}`.
        return f"Wsw{suffix}{idx}", f"bsw{suffix}{idx}"
    if kind == "embedding":
        # No bias for nn.Embedding; bemb{idx} is reserved but never
        # allocated.  Returned only for API symmetry; consumers gate
        # on n_bias==0 from tensor_specs and skip it.
        return f"Wemb{idx}", f"bemb{idx}"
    if kind == "positional_embedding":
        return f"Wpos{idx}", f"bpos{idx}"
    raise ValueError(f"tensor_names: no weights for {kind!r}")


# Per-kind counter map used by `arch_walk`.  Maps each weight-bearing
# arch op to a counter slot; ops absent from the table (`save_residual`,
# `add_residual`, `flatten`, `relu`, `sigmoid`, `tanh`, `softmax`,
# `gelu`, `maxpool_2x2`) yield idx=None.
#
# Crucial aliasing — multi_head_attention and multi_head_attention_kv
# share the `multi_head_attention` slot so a state dict keyed for one
# form loads cleanly into the other (prefill ↔ decode interchange).
# The `tensor_names` helper relies on this: search for "Same weight
# tensor names as multi_head_attention" near MHA-KV.  The GQA-KV
# variant gets its own slot since Llama decoders use the distinct
# Wgqa{*}/bgqa{*} naming.
_ARCH_WALK_COUNTER_KEY = {
    "linear":                       "linear",
    "conv2d_3x3_p1":                "conv2d_3x3_p1",
    "attention":                    "attention",
    "multi_head_attention":         "multi_head_attention",
    "multi_head_attention_kv":      "multi_head_attention",   # SHARED
    "multi_head_attention_gqa_kv":  "multi_head_attention_gqa_kv",
    "layernorm":                    "layernorm",
    "rmsnorm":                      "rmsnorm",
    "swiglu_ffn":                   "swiglu_ffn",
    "embedding":                    "embedding",
    "positional_embedding":         "positional_embedding",
}
# `maxpool_2x2` carries no learnable weights and therefore has no
# `tensor_specs` entry — including it in the counter map would make
# `weight_tensors` crash trying to enumerate it.  `gen_scratch` and
# `gen_forward` track maxpool's per-layer `pool_a{pi}` argmax buffer
# manually with a local `pi` counter; the cost is one extra line at
# each of those two sites in exchange for `weight_tensors` staying a
# clean two-line loop.


def arch_walk(arch):
    """Yield (kind, idx, args) for each layer in arch, with `idx` being
    the per-kind running counter that today is reconstructed manually
    in five separate walkers (weight_tensors, gen_scratch's two passes,
    gen_forward, and main()'s state-dict resolver).  Pseudo / no-counter
    ops (residuals, flatten, activations, maxpool) yield idx=None.

    Yield-then-increment ordering: the value yielded for the n-th
    occurrence of a kind is `n` (0-based), and the counter is bumped
    AFTER the yield.  Flipping this shifts every weight-tensor name by
    +1 (silent state-dict miss at runtime); the previous five walkers
    all used this ordering so we preserve it byte-for-byte.

    `multi_head_attention_kv` shares the `multi_head_attention`
    counter (see `_ARCH_WALK_COUNTER_KEY`).  `multi_head_attention_
    gqa_kv` has its own counter.  Adding a new arch op?  Append it to
    `_ARCH_WALK_COUNTER_KEY` and every consumer picks it up.
    """
    counters = {key: 0 for key in set(_ARCH_WALK_COUNTER_KEY.values())}
    for layer in arch:
        kind, *args = layer
        slot = _ARCH_WALK_COUNTER_KEY.get(kind)
        if slot is None:
            yield (kind, None, args)
            continue
        idx = counters[slot]
        yield (kind, idx, args)
        counters[slot] += 1


def weight_tensors(arch):
    """Yield (kind, idx, suffix, nw, nb, wname, bname) per (W, b) tensor
    pair across the whole arch, in source order.  Thin adapter over
    `arch_walk` — handles the multi-tensor case transparently
    (attention yields three entries per layer with q/k/v suffixes) and
    exposes the canonical names without each callsite having to
    reinvent them."""
    for kind, idx, args in arch_walk(arch):
        if idx is None:
            continue
        for suffix, nw, nb in tensor_specs(kind, *args):
            wname, bname = tensor_names(kind, idx, suffix)
            yield (kind, idx, suffix, nw, nb, wname, bname)


def weight_size(kind, *args):
    """Legacy single-tensor accessor.  Kept so internal callers and any
    out-of-tree forks stay compiling; new code should use tensor_specs."""
    specs = list(tensor_specs(kind, *args))
    if len(specs) != 1:
        raise ValueError(f"weight_size: {kind!r} has {len(specs)} tensor pairs; "
                         f"use tensor_specs for the multi-tensor case.")
    _, nw, nb = specs[0]
    return nw, nb


def gen_weight_decls(arch, embed):
    """Either pre-allocated empty tensors (filled by file_read later) or
    nothing (embed_file inside gen_load declares the let bindings).
    Layers with no bias (n_bias==0, e.g. nn.Embedding) skip the bias
    decl entirely."""
    if embed:
        return []
    lines = []
    for kind, idx, suffix, nw, nb, wname, bname in weight_tensors(arch):
        lines.append(f"    let {wname}: i32 = arr_f32_new({nw});")
        if nb > 0:
            lines.append(f"    let {bname}: i32 = arr_f32_new({nb});")
    return lines


def gen_act_decls(arch):
    """One activation buffer per shape-changing step.  Tracks size in
    elements; spatial vs flat is implicit in how the next layer reads
    it.  Returns (decl_lines, sizes_list)."""
    if not arch:
        raise ValueError("empty arch")
    # Look past pseudo-layers to find the first size-defining entry.
    # save_residual / add_residual / flatten / activations / layernorm
    # all preserve the input shape, so they don't constrain sizes[0].
    # The standard Pre-LN transformer pattern starts with
    # save_residual + layernorm; without this skip, gen_act_decls
    # would either crash or pick the wrong size.
    first = next((l for l in arch if l[0] not in PSEUDO_OPS), None)
    if first is None:
        # Pure LN/activation chain — fall back to first layer's dim
        # (only meaningful for layernorm; activations don't carry a dim).
        first = arch[0]
        if first[0] != "layernorm":
            raise ValueError(
                f"arch must include at least one size-defining layer; "
                f"got pseudo-only with first={first[0]!r}")
    sizes = [first_size_for(first)]

    for kind, *args in arch:
        if kind == "linear":
            in_f, out_f = args
            # Batch-aware: if the current activation has more than `in_f`
            # elements (e.g. coming out of an attention block with
            # `seq * d_in` elements), apply Linear per row and the
            # output gets `(seq) * out_f` elements.  For seq=1 this
            # collapses to the classic single-row behaviour.
            cur = sizes[-1]
            if cur >= in_f and cur % in_f == 0:
                seq = cur // in_f
                sizes.append(seq * out_f)
            else:
                # Mismatch — keep the old assumption; the runtime call
                # would emit garbage.  Most likely a bug in the arch JSON.
                sizes.append(out_f)
        elif kind == "conv2d_3x3_p1":
            c_in, c_out = args
            sizes.append(c_out * 28 * 28)
        elif kind == "maxpool_2x2":
            c = args[0]
            sizes.append(c * 14 * 14)
        elif kind == "attention":
            # Output is [seq, d_head] — same row count as the input,
            # potentially different feature dim.
            seq, _, d_head, _ = args
            sizes.append(seq * d_head)
        elif kind == "multi_head_attention":
            # Output is [seq, d_model] (the heads are concatenated and
            # projected back to d_model) — same shape as the input.
            seq, d_model, _, _ = args
            sizes.append(seq * d_model)
        elif kind == "multi_head_attention_kv":
            # KV-cache MHA always operates on a single token (one
            # query at a time, attending against all cached keys).
            # Output is [d_model] — same shape as the single-row input.
            _, d_model, _ = args
            sizes.append(d_model)
        elif kind == "multi_head_attention_gqa_kv":
            # Same single-token, d_model-out contract as MHA-KV.
            _, d_model, _, _, _ = args
            sizes.append(d_model)
        elif kind == "embedding":
            # Token-ID lookup → [seq, d_model] f32.  When embedding is
            # the first layer, this is the same size as sizes[0] (set
            # by the first-layer branch above) — but iterating over
            # arch yields embedding too on the first pass, so we'd
            # double-append.  Skip if this is the first layer.
            if len(sizes) == 1 and arch[0][0] == "embedding":
                pass
            else:
                _, d_model, seq = args
                sizes.append(seq * d_model)
        elif kind == "positional_embedding":
            # In-place add to the existing [seq, d_model] activation —
            # no size change, no new buffer.
            pass
        elif kind == "layernorm":
            # In-place — same buffer, same size.
            pass
        elif kind == "rmsnorm":
            # In-place — same buffer, same size.
            pass
        elif kind == "swiglu_ffn":
            # ["swiglu_ffn", d_model, d_ffn] — shape-preserving (input
            # and output both d_model wide).  We still emit into a
            # fresh buffer because the down projection's output can't
            # safely alias the input; the size match means the new
            # buffer is sized exactly d_model.
            d_model, _ = args
            sizes.append(d_model)
        elif kind in ("save_residual", "add_residual"):
            # In-place; the save copies, the add accumulates.  Neither
            # changes the activation size.
            pass
        elif kind == "flatten":
            # No size change, just shape interpretation.  Reuse same buffer.
            pass
        # in-place activations: same buffer

    lines = [f"    let a{i}: i32 = arr_f32_new({sz});"
             for i, sz in enumerate(sizes)]
    return lines, sizes


def gen_scratch(arch):
    """Extra scratch buffers needed by spatial layers."""
    lines = []
    # Decoder mode emits its own per-token positional-embedding add
    # inline in gen_decoder_main (one row of d_model adds per step), so
    # the (seq*d_model) pe_window scratch is unused there.  Skip the
    # allocation in that case to avoid an orphan buffer.
    decoder_mode = is_decoder_arch(arch)
    if any(layer[0] == "conv2d_3x3_p1" for layer in arch):
        lines.append("    let padded: i32 = arr_f32_new(900);  // 30×30 zero-padded")
    # Multi-channel conv now goes through the native AVX2 builtin
    # arr_f32_conv2d_3x3_p1_multi, which expects [C_in, 30, 30] flat
    # input.  Allocate one shared padded_multi sized to max(C_in)*900
    # and reuse it across all multi-channel conv layers in the arch.
    multi_convs = [(c_in, c_out) for kind, *a in arch
                   if kind == "conv2d_3x3_p1"
                   for c_in, c_out in [tuple(a)] if c_in > 1]
    if multi_convs:
        max_c_in = max(c_in for c_in, _ in multi_convs)
        lines.append(f"    let padded_multi: i32 = arr_f32_new({max_c_in * 900});")
    # SwiGLU FFN needs a zero-bias scratch sized to max(d_ffn) across
    # all swiglu layers; matvec wants a real bias buffer (not NULL),
    # and Llama's FFN has zero biases by convention.  One shared buffer
    # works because matvec only reads bias[0..m-1], and m ≤ max(d_ffn).
    swiglu_dffns = [args[1] for kind, *args in arch if kind == "swiglu_ffn"]
    if swiglu_dffns:
        zb = max(swiglu_dffns)
        lines.append(f"    let _zb_swg: i32 = arr_f32_new({zb});")
        lines.append(f"    arr_f32_fill(_zb_swg, 0.0);")
    pi = 0   # maxpool_2x2 counter (manual; not in arch_walk because it
             # has no tensor_specs entry — see _ARCH_WALK_COUNTER_KEY)
    for kind, idx, args in arch_walk(arch):
        if kind == "maxpool_2x2":
            c = args[0]
            lines.append(f"    let pool_a{pi}: i32 = arr_f32_new({c * 14 * 14});")
            pi += 1
        elif kind == "attention":
            seq, d_in, d_head, causal = args
            lines.append(
                f"    let attn_desc_{idx}: i32 = "
                f"attn_alloc_desc_f32({seq}, {d_in}, {d_head}, {causal});")
        elif kind == "embedding":
            # No descriptor or scratch needed — the embedding forward
            # reads token IDs directly from the input-loader-emitted
            # _toks_raw buffer and copy_slice's into the activation.
            pass
        elif kind == "positional_embedding":
            # Need a window scratch sized [seq, d_model] so we can
            # arr_f32_add_scaled the leading seq*d_model slice of
            # the (max_pos, d_model) table into the activation in
            # one whole-array call.  Decoder mode emits the per-row
            # add inline in main() (single token per step), so it
            # doesn't need or use this buffer — skip the allocation.
            max_pos, d_model, seq = args
            if not decoder_mode:
                lines.append(
                    f"    let pe_window_{idx}: i32 = arr_f32_new({seq * d_model});")
        elif kind == "multi_head_attention":
            seq, d_model, n_heads, causal = args
            d_head = d_model // n_heads
            # Reuse one single-head descriptor across all n_heads heads —
            # its scratch is sized (seq, d_head), which doesn't change
            # within a layer.  Need SEPARATE Q/K/V slice scratches: the
            # single-head kernel binds W_Q/W_K/W_V into its descriptor and
            # then runs the project pass that reads all three in the same
            # loop, so reusing one scratch would let the V slice clobber
            # Q before Q's matvec runs.
            lines.append(
                f"    let mha_desc_{idx}: i32 = "
                f"attn_alloc_desc_f32({seq}, {d_model}, {d_head}, {causal});")
            for proj in ("q", "k", "v"):
                lines.append(
                    f"    let mha_W{proj}slice_{idx}: i32 = "
                    f"arr_f32_new({d_head * d_model});")
                lines.append(
                    f"    let mha_b{proj}slice_{idx}: i32 = "
                    f"arr_f32_new({d_head});")
            lines.append(
                f"    let mha_head_out_{idx}: i32 = arr_f32_new({seq * d_head});")
            lines.append(
                f"    let mha_concat_{idx}: i32 = arr_f32_new({seq * d_model});")
        elif kind == "multi_head_attention_kv":
            # Decode-time MHA: one persistent attn_mh_kv state per layer
            # holds the K/V cache plus all per-head slice scratches.
            # The state's Q/K/V/O weight slots get bound to the loaded
            # Wmq{i}/bmq{i}/... tensors right after allocation so the
            # token loop can call attn_mh_kv_step_f32 with no per-step
            # rebinding.  The X (input) and out (output) slots stay
            # rebindable since the per-step forward chooses fresh
            # activation buffers each iteration.
            #
            # `idx` here is the SHARED multi_head_attention counter
            # (mi-equivalent), so the Wmq{idx}/bmq{idx} names line up
            # with whatever weight_tensors / gen_load emitted.
            max_seq, d_model, n_heads = args
            lines.append(
                f"    let mhkv_state_{idx}: i32 = "
                f"attn_mh_kv_alloc_f32({max_seq}, {d_model}, {n_heads});")
            lines.append(f"    arr_set(mhkv_state_{idx}, 1, Wmq{idx});")
            lines.append(f"    arr_set(mhkv_state_{idx}, 2, bmq{idx});")
            lines.append(f"    arr_set(mhkv_state_{idx}, 3, Wmk{idx});")
            lines.append(f"    arr_set(mhkv_state_{idx}, 4, bmk{idx});")
            lines.append(f"    arr_set(mhkv_state_{idx}, 5, Wmv{idx});")
            lines.append(f"    arr_set(mhkv_state_{idx}, 6, bmv{idx});")
            lines.append(f"    arr_set(mhkv_state_{idx}, 7, Wmo{idx});")
            lines.append(f"    arr_set(mhkv_state_{idx}, 8, bmo{idx});")
        elif kind == "multi_head_attention_gqa_kv":
            # Llama-style GQA decode step: persistent state holds RoPE
            # table + n_kv_heads KV caches + per-head slice scratches.
            # No biases — slots 2/4/6/8 stay at the alloc-time 0
            # sentinel and the step kernel routes them through the
            # internal zero-fill scratches.
            max_seq, d_model, n_heads, n_kv_heads, theta = args
            lines.append(
                f"    let gqa_state_{idx}: i32 = "
                f"attn_gqa_kv_alloc_f32({max_seq}, {d_model}, {n_heads}, "
                f"{n_kv_heads}, {theta!r});")
            lines.append(f"    arr_set(gqa_state_{idx}, 1, Wgqaq{idx});")
            lines.append(f"    arr_set(gqa_state_{idx}, 3, Wgqak{idx});")
            lines.append(f"    arr_set(gqa_state_{idx}, 5, Wgqav{idx});")
            lines.append(f"    arr_set(gqa_state_{idx}, 7, Wgqao{idx});")
        elif kind == "swiglu_ffn":
            # Per-layer gate / up scratches sized to d_ffn.  silu_mul
            # writes silu(gate)*up into the up buffer in place; the
            # final down matvec reads from the up buffer.
            d_model, d_ffn = args
            lines.append(
                f"    let _swg_gate_{idx}: i32 = arr_f32_new({d_ffn});")
            lines.append(
                f"    let _swg_up_{idx}: i32 = arr_f32_new({d_ffn});")
    # Residual slot buffers: one per save_residual entry, sized to the
    # activation count at the save point (= same as the matched add).
    for slot_idx, n_elems in residual_slots(arch):
        lines.append(
            f"    let _resid_{slot_idx}: i32 = arr_f32_new({n_elems});")
    # Batched-Linear scratch: when any Linear receives more elements
    # than its in_f (i.e. operates on a [seq, in_f] flat input that
    # came from an attention block), emit_linear loops per row using
    # _lin_in_{li} / _lin_out_{li} — one pair per Linear, sized
    # EXACTLY to that Linear's in_f / out_f.  Per-Linear (not
    # shared-max) sizing is required because arr_i8_matvec_f32
    # derives `n` implicitly from x's length header; a shared
    # max-sized scratch would feed the kernel an n bigger than W's
    # row stride and trip the bounds check (caught in the distilbert
    # demo before this fix).
    #
    # MHA's output projection still uses the shared _row_in /
    # _row_out — its in_f and out_f are equal (both d_model) so it's
    # immune to the implicit-n problem; allocating per-MHA-layer
    # scratch would just bloat the binary without the correctness
    # constraint that drove the per-Linear split.
    linear_layers_in = [args for kind, *args in arch if kind == "linear"]
    mha_dims = [args[1] for kind, *args in arch
                if kind == "multi_head_attention"]
    if linear_layers_in or mha_dims:
        # Detect whether ANY Linear is batched by walking arch with a
        # cur_size tracker (mirrors gen_act_decls' batch logic).
        first_lin = next((l for l in arch if l[0] not in PSEUDO_OPS), None)
        cur = first_size_for(first_lin) if first_lin is not None else 0
        any_batched = False
        for kind, *args in arch:
            if kind == "linear":
                in_f, out_f = args
                if cur >= in_f and cur % in_f == 0 and cur > in_f:
                    any_batched = True
                cur = (cur // in_f) * out_f if (cur >= in_f and cur % in_f == 0) else out_f
            elif kind == "conv2d_3x3_p1":
                cur = args[1] * 28 * 28
            elif kind == "maxpool_2x2":
                cur = args[0] * 14 * 14
            elif kind == "attention":
                cur = args[0] * args[2]
            elif kind == "multi_head_attention":
                cur = args[0] * args[1]
            elif kind == "multi_head_attention_kv":
                cur = args[1]   # single-row d_model
            elif kind == "multi_head_attention_gqa_kv":
                cur = args[1]   # single-row d_model (GQA mirrors MHA-KV)
            elif kind == "swiglu_ffn":
                # Output dim = d_model (first arg).
                cur = args[0]
            elif kind == "embedding":
                cur = args[2] * args[1]   # seq * d_model
        if any_batched:
            # Per-Linear in/out scratches.  Walk arch once more, this
            # time tracking the per-Linear cur_size so we know which
            # Linears actually emit the batched loop (and therefore
            # need their per-Linear scratch).  Linears that fit the
            # seq=1 path don't need it.
            # Re-walk: replicate the cur tracking from earlier so we
            # only allocate scratch for Linears that actually batch.
            cur2 = first_size_for(first_lin) if first_lin is not None else 0
            li2 = 0
            for kind, *args in arch:
                if kind == "linear":
                    in_f, out_f = args
                    if cur2 >= in_f and cur2 % in_f == 0 and cur2 > in_f:
                        lines.append(
                            f"    let _lin_in_{li2}: i32 = arr_f32_new({in_f});")
                        lines.append(
                            f"    let _lin_out_{li2}: i32 = arr_f32_new({out_f});")
                    cur2 = ((cur2 // in_f) * out_f
                            if (cur2 >= in_f and cur2 % in_f == 0)
                            else out_f)
                    li2 += 1
                elif kind == "conv2d_3x3_p1":
                    cur2 = args[1] * 28 * 28
                elif kind == "maxpool_2x2":
                    cur2 = args[0] * 14 * 14
                elif kind == "attention":
                    cur2 = args[0] * args[2]
                elif kind == "multi_head_attention":
                    cur2 = args[0] * args[1]
                elif kind == "multi_head_attention_kv":
                    cur2 = args[1]
                elif kind == "multi_head_attention_gqa_kv":
                    cur2 = args[1]
                elif kind == "swiglu_ffn":
                    cur2 = args[0]
                elif kind == "embedding":
                    cur2 = args[2] * args[1]
        # MHA's output projection still uses the shared _row_in /
        # _row_out (both sized d_model — symmetric, so no implicit-n
        # mismatch).  Allocate them when any MHA is in the arch.
        if mha_dims:
            d = max(mha_dims)
            lines.append(f"    let _row_in: i32 = arr_f32_new({d});")
            lines.append(f"    let _row_out: i32 = arr_f32_new({d});")
    return lines


def names_for(kind, idx):
    """Legacy single-tensor accessor.  Equivalent to tensor_names(kind, idx)
    with the default empty suffix; kept for callers that don't yet use
    weight_tensors()."""
    return tensor_names(kind, idx)


def gen_load(arch, embed_dir, out_name, embed, quantize, scales,
             multi_int8_runtime=False, int4_embedding=False,
             int4_linear=False):
    """Three paths:
       - runtime file_read (embed=False)            : single .f32 sidecar
       - embed_file f32   (embed=True, quant=none)  : f32 baked into .text
       - embed_file_bytes int8 + dequant (quant=int8): int8 baked + dequant
                                                       at startup
    `scales` is a dict {(kind, idx, "W" | "b"): scale_f64} populated by
    main() when --quantize int8 is in effect; ignored otherwise.

    `multi_int8_runtime` toggles where the multi-channel int8 conv pays
    its dequant cost.  When False (default for batch workloads), it
    pre-dequantises weights to f32 once at startup so the kernel side
    is the f32 multi-channel conv (faster total wall on N>>1 samples).
    When True (single-shot CLI), weights stay int8 in RAM and the kernel
    is arr_i8_conv2d_3x3_p1_multi which dequantises inline per
    (c_out, c_in) — sacrifices ~18 % batch wall to halve cold-start.
    """
    if embed and quantize == "int8":
        # Linear: weights stay int8 in RAM (used directly by arr_i8_matvec_f32);
        #         biases remain f32 (small, no quantisation benefit).
        # Conv (C_in = 1):   weights stay int8 in RAM, used directly by
        #                    arr_i8_conv2d_3x3_p1 — no dequant pass.
        # Conv (C_in > 1):   if multi_int8_runtime, weights stay int8 in
        #                    RAM and arr_i8_conv2d_3x3_p1_multi handles
        #                    the inline dequant; otherwise weights are
        #                    dequantised once at startup (default for
        #                    batch input formats; better steady-state
        #                    throughput, slightly worse cold-start).
        # For the "is multi-channel conv?" check, peek at the same arch
        # entry the tensor came from.  We can't read it from the per-
        # tensor info alone since suffix-style layers don't carry C_in.
        def _conv_c_in(arch_entry):
            return arch_entry[1] if arch_entry[0] == "conv2d_3x3_p1" else None
        arch_by_idx = {("conv2d_3x3_p1", i): _conv_c_in(layer)
                       for i, layer in enumerate(
                           [l for l in arch if l[0] == "conv2d_3x3_p1"])}
        QUANTISABLE = ("linear", "conv2d_3x3_p1", "swiglu_ffn", "embedding")
        lines = []
        for kind, idx, suffix, nw, nb, wname, bname in weight_tensors(arch):
            c_in = arch_by_idx.get((kind, idx))
            if kind not in QUANTISABLE:
                # Not int8-supported (attention / MHA /
                # positional_embedding / layernorm / rmsnorm) — staged
                # as f32 by the main pack(), embed via embed_file
                # directly.
                lines.append(
                    f"    let {wname}: i32 = "
                    f"embed_file(\"{embed_dir}/{out_name}_{wname}.f32\");")
                if nb > 0:
                    lines.append(
                        f"    let {bname}: i32 = "
                        f"embed_file(\"{embed_dir}/{out_name}_{bname}.f32\");")
                continue
            scale_w = scales[(kind, idx, suffix, "W")]
            if kind == "linear":
                # Per-row scales: int8 weight + an f32 scales buffer of
                # length out_f.  Runtime calls arr_i8_matvec_f32_perrow
                # which loads scale[j] per output row.  Int4 variant
                # uses the same layout but packs 2 nibbles/byte (.i4)
                # and is dequantised inline by emit_linear.
                ext = "i4" if int4_linear else "i8"
                lines.append(f"    let {wname}: i32 = embed_file_bytes(\"{embed_dir}/{out_name}_{wname}.{ext}\");")
                lines.append(f"    let {wname}_scales: i32 = embed_file(\"{embed_dir}/{out_name}_{wname}_scales.f32\");")
                lines.append(f"    let {bname}: i32 = embed_file(\"{embed_dir}/{out_name}_{bname}.f32\");")
            elif kind == "swiglu_ffn":
                # 3 weight tensors per layer (gate / up / down), each
                # int8 with its OWN per-row scales buffer.  No biases
                # (Llama).  Runtime: arr_i8_matvec_f32_perrow(W, x, y, m,
                # W_scales).
                lines.append(
                    f"    let {wname}: i32 = "
                    f"embed_file_bytes(\"{embed_dir}/{out_name}_{wname}.i8\");")
                lines.append(
                    f"    let {wname}_scales: i32 = "
                    f"embed_file(\"{embed_dir}/{out_name}_{wname}_scales.f32\");")
                # nb == 0 for swiglu — skip bias emit entirely.
            elif kind == "embedding":
                # Per-row int8 or int4.  Both schemes share the same
                # f32 scales sidecar.  Int4 stores 2 nibbles per byte;
                # the unpack happens in gen_decoder_main's lookup loop
                # so the file is loaded as a plain byte buffer either
                # way.  Extension differs (.i4 vs .i8) so we don't
                # accidentally reuse a stale staging file.
                ext = "i4" if int4_embedding else "i8"
                lines.append(
                    f"    let {wname}: i32 = "
                    f"embed_file_bytes(\"{embed_dir}/{out_name}_{wname}.{ext}\");")
                lines.append(
                    f"    let {wname}_scales: i32 = "
                    f"embed_file(\"{embed_dir}/{out_name}_{wname}_scales.f32\");")
            elif kind == "conv2d_3x3_p1" and c_in == 1:
                # Single-channel conv: int8 weights stay int8.
                lines.append(f"    let {wname}: i32 = embed_file_bytes(\"{embed_dir}/{out_name}_{wname}.i8\");")
                lines.append(f"    let {bname}: i32 = embed_file(\"{embed_dir}/{out_name}_{bname}.f32\");")
            elif multi_int8_runtime:
                # Multi-channel conv, single-shot mode: keep int8.
                lines.append(f"    let {wname}: i32 = embed_file_bytes(\"{embed_dir}/{out_name}_{wname}.i8\");")
                lines.append(f"    let {bname}: i32 = embed_file(\"{embed_dir}/{out_name}_{bname}.f32\");")
            else:
                # Multi-channel conv, batch mode: dequant once at startup.
                lines.append(f"    let _q_{wname}: i32 = embed_file_bytes(\"{embed_dir}/{out_name}_{wname}.i8\");")
                lines.append(f"    let {wname}: i32 = arr_f32_new({nw});")
                lines.append(f"    dequant_int8_to_f32(_q_{wname}, {wname}, {nw}, {scale_w!r});")
                lines.append(f"    let {bname}: i32 = embed_file(\"{embed_dir}/{out_name}_{bname}.f32\");")
        return lines
    if embed:
        lines = []
        for kind, idx, suffix, nw, nb, wname, bname in weight_tensors(arch):
            lines.append(f"    let {wname}: i32 = embed_file(\"{embed_dir}/{out_name}_{wname}.f32\");")
            if nb > 0:
                lines.append(f"    let {bname}: i32 = embed_file(\"{embed_dir}/{out_name}_{bname}.f32\");")
        return lines
    lines = [f"    let weight_path: i32 = str_new(\"{out_name}.f32\");",
             "    let _wfd: i32 = file_open(weight_path, 0);"]
    for kind, idx, suffix, nw, nb, wname, bname in weight_tensors(arch):
        lines.append(f"    file_read(_wfd, {wname}, {nw * 4});")
        if nb > 0:
            lines.append(f"    file_read(_wfd, {bname}, {nb * 4});")
    lines.append("    file_close(_wfd);")
    return lines


def gen_forward(arch, scales=None, multi_int8_runtime=False, int4_linear=False):
    """Emit the per-sample forward pass on activation buffers a0..aN.

    cur_a tracks which activation buffer holds the running tensor.
    Layers that change the size advance to the next buffer; in-place
    activations and `flatten` (which is just a reshape) keep the same
    buffer.

    `scales` is a {(kind, idx, "W"|"b") → float} dict populated when
    --quantize int8 is in effect; emit_linear will switch to the native
    int8 matvec path when it sees a scale for its layer.
    """
    lines = []
    cur_a = 0
    pi = 0   # maxpool_2x2 counter (manual; not in arch_walk because
             # maxpool has no tensor_specs entry)
    # Stack of (slot_idx, n_elems) for every save_residual we've emitted
    # but not yet matched with an add_residual.  Pre-validated by
    # residual_slots (called from gen_scratch); here we just track
    # which slot each add belongs to.
    resid_stack = []
    next_resid_slot = 0
    # Recompute per-position activation sizes (mirrors gen_act_decls
    # exactly) so Linears that receive [seq, in_f] flat input know to
    # emit a per-row loop.  Could share the list with gen_act_decls'
    # return value, but recomputing is cheaper than threading through.
    _, sizes_for_seq = gen_act_decls(arch)
    for kind, idx, args in arch_walk(arch):
        if kind == "linear":
            in_f, out_f = args
            src = f"a{cur_a}"
            dst = f"a{cur_a + 1}"
            sc = scales.get(("linear", idx, "", "W")) if scales else None
            cur_size = sizes_for_seq[cur_a]
            seq_in = cur_size // in_f if (cur_size >= in_f and cur_size % in_f == 0) else 1
            lines += emit_linear(idx, in_f, out_f, src, dst,
                                 f"W{idx}", f"b{idx}", quant_scale=sc,
                                 seq=seq_in,
                                 bits=(4 if int4_linear and sc is not None else 8))
            cur_a += 1
        elif kind == "conv2d_3x3_p1":
            c_in, c_out = args
            src = f"a{cur_a}"
            dst = f"a{cur_a + 1}"
            conv_scale = scales.get(("conv2d_3x3_p1", idx, "", "W")) if scales else None
            if c_in == 1:
                # Fast path: single-channel input goes straight into the
                # AVX2 builtin.  Pad once, call once.  The int8 variant
                # carries the per-tensor scale and reads weights as i8;
                # otherwise fall back to the f32 conv builtin.
                lines.append(f"    arr_f32_fill(padded, 0.0);")
                lines.append(f"    let _y{idx}: i32 = 0;")
                lines.append(f"    while (_y{idx} < 28) {{")
                lines.append(f"        arr_f32_copy_slice({src}, _y{idx} * 28, padded, (_y{idx} + 1) * 30 + 1, 28);")
                lines.append(f"        _y{idx} = _y{idx} + 1;")
                lines.append(f"    }}")
                if conv_scale is not None:
                    lines.append(f"    arr_i8_conv2d_3x3_p1(padded, Wc{idx}, bc{idx}, {dst}, {c_out}, {conv_scale!r});")
                else:
                    lines.append(f"    arr_f32_conv2d_3x3_p1(padded, Wc{idx}, bc{idx}, {dst}, {c_out});")
            else:
                # Multi-channel input: pad each input channel into the
                # shared padded_multi [C_in, 30, 30] buffer and call the
                # native AVX2 multi-channel kernel once.  No more user-fn
                # accumulation loop or _partial/_w_slice/_b_zero scratch
                # — the kernel does the load-modify-store accumulation
                # internally and broadcasts bias once per c_out.
                lines.append(f"    arr_f32_fill(padded_multi, 0.0);")
                lines.append(f"    let _cin{idx}: i32 = 0;")
                lines.append(f"    while (_cin{idx} < {c_in}) {{")
                lines.append(f"        let _yc{idx}: i32 = 0;")
                lines.append(f"        while (_yc{idx} < 28) {{")
                lines.append(f"            arr_f32_copy_slice({src}, _cin{idx} * 784 + _yc{idx} * 28, padded_multi, _cin{idx} * 900 + (_yc{idx} + 1) * 30 + 1, 28);")
                lines.append(f"            _yc{idx} = _yc{idx} + 1;")
                lines.append(f"        }}")
                lines.append(f"        _cin{idx} = _cin{idx} + 1;")
                lines.append(f"    }}")
                if conv_scale is not None and multi_int8_runtime:
                    # int8 weights stay int8 in RAM — no startup dequant.
                    lines.append(f"    arr_i8_conv2d_3x3_p1_multi(padded_multi, Wc{idx}, bc{idx}, {dst}, {c_out}, {conv_scale!r});")
                else:
                    # f32 multi-channel kernel handles both the
                    # plain-f32 case and the int8-dequantised-at-startup
                    # case (Wc{idx} is f32 in RAM in both).
                    lines.append(f"    arr_f32_conv2d_3x3_p1_multi(padded_multi, {c_in}, Wc{idx}, bc{idx}, {dst}, {c_out});")
            cur_a += 1
        elif kind == "maxpool_2x2":
            c = args[0]
            src = f"a{cur_a}"
            dst = f"a{cur_a + 1}"
            argmax_buf = f"pool_a{pi}"
            # 2×2 max pooling kernel — inlined from conv2d_f32.ari.
            lines.append(f"    let _c{pi}: i32 = 0;")
            lines.append(f"    while (_c{pi} < {c}) {{")
            lines.append(f"        let _ino{pi}: i32 = _c{pi} * 784;")
            lines.append(f"        let _outo{pi}: i32 = _c{pi} * 196;")
            lines.append(f"        let _y{pi}: i32 = 0;")
            lines.append(f"        while (_y{pi} < 14) {{")
            lines.append(f"            let _x{pi}: i32 = 0;")
            lines.append(f"            while (_x{pi} < 14) {{")
            lines.append(f"                let _i0{pi}: i32 = _ino{pi} + (2 * _y{pi}) * 28 + 2 * _x{pi};")
            lines.append(f"                let _v0{pi}: f64 = arr_f32_get({src}, _i0{pi});")
            lines.append(f"                let _v1{pi}: f64 = arr_f32_get({src}, _i0{pi} + 1);")
            lines.append(f"                let _v2{pi}: f64 = arr_f32_get({src}, _i0{pi} + 28);")
            lines.append(f"                let _v3{pi}: f64 = arr_f32_get({src}, _i0{pi} + 29);")
            lines.append(f"                let _bv{pi}: f64 = _v0{pi};")
            lines.append(f"                if (_v1{pi} > _bv{pi}) {{ _bv{pi} = _v1{pi}; }}")
            lines.append(f"                if (_v2{pi} > _bv{pi}) {{ _bv{pi} = _v2{pi}; }}")
            lines.append(f"                if (_v3{pi} > _bv{pi}) {{ _bv{pi} = _v3{pi}; }}")
            lines.append(f"                arr_f32_set({dst}, _outo{pi} + _y{pi} * 14 + _x{pi}, _bv{pi});")
            lines.append(f"                _x{pi} = _x{pi} + 1;")
            lines.append(f"            }}")
            lines.append(f"            _y{pi} = _y{pi} + 1;")
            lines.append(f"        }}")
            lines.append(f"        _c{pi} = _c{pi} + 1;")
            lines.append(f"    }}")
            cur_a += 1
            pi += 1
        elif kind == "layernorm":
            # In-place affine layernorm over the last axis.  The kernel
            # normalises (mean=0, var=1 per group of `dim`); the helper
            # then applies the learnable γ and β.
            dim = args[0]
            src = f"a{cur_a}"
            lines.append(
                f"    layernorm_affine_f32({src}, {dim}, Wn{idx}, bn{idx});")
        elif kind == "rmsnorm":
            # In-place RMSNorm with affine γ (no β).  Llama / Mistral.
            dim = args[0]
            src = f"a{cur_a}"
            lines.append(
                f"    rmsnorm_affine_f32({src}, {dim}, Wr{idx});")
        elif kind == "swiglu_ffn":
            # Llama-style FFN: down(silu(gate(x)) * up(x)).  Three
            # matvecs + one element-wise silu*mul.  No biases.
            #
            # int8 path (per-tensor scale per projection — gate/up/down
            # have independently-scaled distributions; sharing one scale
            # across all three drifts noticeably on real Llama checkpoints):
            #   arr_i8_matvec_f32(W, x, b, y, m, n, scale)
            # f32 path:
            #   arr_f32_matvec(W, x, b, y, m, n)
            d_model, d_ffn = args
            src = f"a{cur_a}"
            dst = f"a{cur_a + 1}"
            sc_gate = scales.get(("swiglu_ffn", idx, "gate", "W")) if scales else None
            sc_up   = scales.get(("swiglu_ffn", idx, "up",   "W")) if scales else None
            sc_down = scales.get(("swiglu_ffn", idx, "down", "W")) if scales else None
            # arr_i8_matvec_f32_perrow(W, x, y, m, scales_buf) — 5 args
            # (n implicit from x's length header).  Per-row scales buffer
            # is allocated by gen_load as `{w_var}_scales` (one f32 per
            # output row of W).  No bias term: Llama's FFN omits them,
            # so the int8 path skips the zero-bias scratch entirely.
            lines.append(f"    // swiglu_ffn {idx}: d_model={d_model} d_ffn={d_ffn}"
                         f"{' [int8 per-row]' if sc_gate is not None else ''}")
            if sc_gate is not None:
                lines.append(
                    f"    arr_i8_matvec_f32_perrow(Wswgate{idx}, {src}, "
                    f"_swg_gate_{idx}, {d_ffn}, Wswgate{idx}_scales);")
            else:
                lines.append(
                    f"    arr_f32_matvec(Wswgate{idx}, {src}, _zb_swg, "
                    f"_swg_gate_{idx}, {d_ffn}, {d_model});")
            if sc_up is not None:
                lines.append(
                    f"    arr_i8_matvec_f32_perrow(Wswup{idx}, {src}, "
                    f"_swg_up_{idx}, {d_ffn}, Wswup{idx}_scales);")
            else:
                lines.append(
                    f"    arr_f32_matvec(Wswup{idx}, {src}, _zb_swg, "
                    f"_swg_up_{idx}, {d_ffn}, {d_model});")
            lines.append(
                f"    silu_mul_f32(_swg_gate_{idx}, _swg_up_{idx}, {d_ffn});")
            if sc_down is not None:
                lines.append(
                    f"    arr_i8_matvec_f32_perrow(Wswdown{idx}, _swg_up_{idx}, "
                    f"{dst}, {d_model}, Wswdown{idx}_scales);")
            else:
                lines.append(
                    f"    arr_f32_matvec(Wswdown{idx}, _swg_up_{idx}, _zb_swg, "
                    f"{dst}, {d_model}, {d_ffn});")
            cur_a += 1
        elif kind == "attention":
            # Single-head scaled dot-product attention.
            # The descriptor was allocated once in gen_scratch; here we
            # just point its Q/K/V/output slots at the current activation
            # buffers and the per-layer weight tensors, then run the
            # one-shot forward.  After the call, dst holds [seq, d_head]
            # row-major — same shape contract as the kernel returns.
            seq, d_in, d_head, causal = args
            src = f"a{cur_a}"
            dst = f"a{cur_a + 1}"
            d = f"attn_desc_{idx}"
            lines.append(f"    arr_set({d},  0, {src});")
            lines.append(f"    arr_set({d},  1, Wq{idx});")
            lines.append(f"    arr_set({d},  2, bq{idx});")
            lines.append(f"    arr_set({d},  3, Wk{idx});")
            lines.append(f"    arr_set({d},  4, bk{idx});")
            lines.append(f"    arr_set({d},  5, Wv{idx});")
            lines.append(f"    arr_set({d},  6, bv{idx});")
            lines.append(f"    arr_set({d}, 11, {dst});")
            lines.append(f"    attention_forward_f32({d});")
            cur_a += 1
        elif kind == "multi_head_attention":
            # Multi-head SDPA = single-head SDPA per head, with weights
            # sliced from (d_model, d_model) into per-head (d_head, d_model)
            # views, then heads concatenated and pushed through an output
            # projection.  We reuse the existing single-head kernel n_heads
            # times via a copy_slice + descriptor-pointer-swap scheme.
            seq, d_model, n_heads, causal = args
            d_head = d_model // n_heads
            src = f"a{cur_a}"
            dst = f"a{cur_a + 1}"
            d = f"mha_desc_{idx}"
            ho = f"mha_head_out_{idx}"
            cc = f"mha_concat_{idx}"
            lines.append(f"    // multi_head_attention {idx}: "
                         f"d_model={d_model} n_heads={n_heads} d_head={d_head}")
            lines.append(f"    let _mh{idx}: i32 = 0;")
            lines.append(f"    while (_mh{idx} < {n_heads}) {{")
            # Slice per-head Q/K/V W and b out of the full d_model tensors
            # into separate scratches; the project pass inside
            # attention_forward_f32 reads all three in the same loop, so
            # they need to be live simultaneously.  The full Q/K/V
            # weights stay in their (d_model, d_model) layout and we
            # copy the contiguous d_head-row chunk for the current head.
            for proj in ("q", "k", "v"):
                W_full = f"Wm{proj}{idx}"
                b_full = f"bm{proj}{idx}"
                ws = f"mha_W{proj}slice_{idx}"
                bs = f"mha_b{proj}slice_{idx}"
                slot_w = {"q": 1, "k": 3, "v": 5}[proj]
                slot_b = {"q": 2, "k": 4, "v": 6}[proj]
                lines.append(
                    f"        arr_f32_copy_slice({W_full}, "
                    f"_mh{idx} * {d_head * d_model}, {ws}, 0, {d_head * d_model});")
                lines.append(
                    f"        arr_f32_copy_slice({b_full}, "
                    f"_mh{idx} * {d_head}, {bs}, 0, {d_head});")
                lines.append(f"        arr_set({d}, {slot_w}, {ws});")
                lines.append(f"        arr_set({d}, {slot_b}, {bs});")
            lines.append(f"        arr_set({d}, 0, {src});")
            lines.append(f"        arr_set({d}, 11, {ho});")
            lines.append(f"        attention_forward_f32({d});")
            # Concat: per row, copy d_head elements from ho into cc at
            # offset row*d_model + h*d_head.
            lines.append(f"        let _mr{idx}: i32 = 0;")
            lines.append(f"        while (_mr{idx} < {seq}) {{")
            lines.append(
                f"            arr_f32_copy_slice({ho}, _mr{idx} * {d_head}, "
                f"{cc}, _mr{idx} * {d_model} + _mh{idx} * {d_head}, {d_head});")
            lines.append(f"            _mr{idx} = _mr{idx} + 1;")
            lines.append(f"        }}")
            lines.append(f"        _mh{idx} = _mh{idx} + 1;")
            lines.append(f"    }}")
            # Output projection: dst[r] = Wo · cc[r] + bo, per row.
            lines.append(f"    let _mp{idx}: i32 = 0;")
            lines.append(f"    while (_mp{idx} < {seq}) {{")
            lines.append(
                f"        arr_f32_copy_slice({cc}, _mp{idx} * {d_model}, "
                f"_row_in, 0, {d_model});")
            lines.append(
                f"        arr_f32_matvec(Wmo{idx}, _row_in, bmo{idx}, "
                f"_row_out, {d_model}, {d_model});")
            lines.append(
                f"        arr_f32_copy_slice(_row_out, 0, {dst}, "
                f"_mp{idx} * {d_model}, {d_model});")
            lines.append(f"        _mp{idx} = _mp{idx} + 1;")
            lines.append(f"    }}")
            cur_a += 1
        elif kind == "multi_head_attention_kv":
            # KV-cached MHA decode step.  All heads + Q/K/V/O projection
            # + cache append + scaled-softmax-against-cache + concat
            # happens inside attn_mh_kv_step_f32 against the persistent
            # mhkv_state_{idx} (allocated and weight-bound in gen_scratch).
            # Per token we just rebind input + output and call once.
            #
            # `idx` here is the SHARED multi_head_attention counter
            # (mi-equivalent), so the Wmq{idx} / mhkv_state_{idx} names
            # match what gen_scratch and weight_tensors emitted.
            max_seq, d_model, n_heads = args
            src = f"a{cur_a}"
            dst = f"a{cur_a + 1}"
            s = f"mhkv_state_{idx}"
            lines.append(f"    // multi_head_attention_kv {idx}: "
                         f"d_model={d_model} n_heads={n_heads} max_seq={max_seq}")
            lines.append(f"    arr_set({s}, 0, {src});")
            lines.append(f"    arr_set({s}, 9, {dst});")
            lines.append(f"    attn_mh_kv_step_f32({s});")
            cur_a += 1
        elif kind == "multi_head_attention_gqa_kv":
            # Llama-style GQA decode step.  RoPE + per-KV-head cache +
            # per-Q-head scoring + output projection all live inside
            # attn_gqa_kv_step_f32 against gqa_state_{idx} (alloc + weight
            # binding emitted in gen_scratch).  Per token we just bind
            # X + out and call once.
            max_seq, d_model, n_heads, n_kv_heads, theta = args
            src = f"a{cur_a}"
            dst = f"a{cur_a + 1}"
            s = f"gqa_state_{idx}"
            lines.append(f"    // multi_head_attention_gqa_kv {idx}: "
                         f"d_model={d_model} n_heads={n_heads} "
                         f"n_kv_heads={n_kv_heads} theta={theta!r} "
                         f"max_seq={max_seq}")
            lines.append(f"    arr_set({s}, 0, {src});")
            lines.append(f"    arr_set({s}, 9, {dst});")
            lines.append(f"    attn_gqa_kv_step_f32({s});")
            cur_a += 1
        elif kind == "flatten":
            # Reshape only — same buffer holds the data.  No code emitted.
            pass
        elif kind == "relu":
            lines += emit_relu(f"a{cur_a}")
        elif kind == "sigmoid":
            lines += emit_sigmoid(f"a{cur_a}")
        elif kind == "tanh":
            lines += emit_tanh(f"a{cur_a}")
        elif kind == "softmax":
            lines += emit_softmax(f"a{cur_a}")
        elif kind == "gelu":
            lines += emit_gelu(f"a{cur_a}")
        elif kind == "embedding":
            # Token-ID lookup.  Picks 1, 2, or 4 bytes per token at pack
            # time based on vocab_size, so real HF transformer vocabs
            # (30K+ → 2 bytes) just work.  The input loader emits a
            # raw byte stream into _toks_raw via embed_file_bytes;
            # this loop decodes the i-th token via byte_at and copies
            # d_model f32 elements from the matching row of W_emb.
            vocab_size, d_model, seq = args
            if cur_a != 0:
                raise SystemExit(
                    f"embedding #{idx} must be the first layer; today the "
                    "packer only emits a token-input loader for the "
                    "leading position.")
            tb = _embedding_token_bytes(vocab_size)
            dst = f"a{cur_a}"
            lines.append(
                f"    // embedding {idx}: vocab={vocab_size} d_model={d_model} "
                f"seq={seq} ({tb} byte/token)")
            lines.append(f"    let _ei{idx}: i32 = 0;")
            lines.append(f"    while (_ei{idx} < {seq}) {{")
            if tb == 1:
                lines.append(
                    f"        let _tok{idx}: i32 = byte_at(_toks_raw, _ei{idx});")
            elif tb == 2:
                lines.append(
                    f"        let _tok{idx}: i32 = "
                    f"byte_at(_toks_raw, _ei{idx} * 2) + "
                    f"byte_at(_toks_raw, _ei{idx} * 2 + 1) * 256;")
            else:   # tb == 4
                lines.append(
                    f"        let _tok{idx}: i32 = "
                    f"byte_at(_toks_raw, _ei{idx} * 4) + "
                    f"byte_at(_toks_raw, _ei{idx} * 4 + 1) * 256 + "
                    f"byte_at(_toks_raw, _ei{idx} * 4 + 2) * 65536 + "
                    f"byte_at(_toks_raw, _ei{idx} * 4 + 3) * 16777216;")
            lines.append(
                f"        arr_f32_copy_slice(Wemb{idx}, _tok{idx} * {d_model}, "
                f"{dst}, _ei{idx} * {d_model}, {d_model});")
            lines.append(f"        _ei{idx} = _ei{idx} + 1;")
            lines.append(f"    }}")
            # cur_a stays at 0; the embedding's output IS a0.
        elif kind == "positional_embedding":
            # Add the leading seq*d_model slice of W_pos in-place to
            # the current activation.  Slice once into pe_window_i so
            # arr_f32_add_scaled (whole-array) can fold it back.
            max_pos, d_model, seq = args
            lines.append(
                f"    arr_f32_copy_slice(Wpos{idx}, 0, "
                f"pe_window_{idx}, 0, {seq * d_model});")
            lines.append(
                f"    arr_f32_add_scaled(a{cur_a}, "
                f"pe_window_{idx}, 1.0);")
        elif kind == "save_residual":
            slot_idx = next_resid_slot
            next_resid_slot += 1
            resid_stack.append(slot_idx)
            lines.append(
                f"    arr_f32_copy_slice(a{cur_a}, 0, _resid_{slot_idx}, 0, "
                f"arr_len(a{cur_a}));")
        elif kind == "add_residual":
            slot_idx = resid_stack.pop()
            lines.append(
                f"    arr_f32_add_scaled(a{cur_a}, _resid_{slot_idx}, 1.0);")
        else:
            raise ValueError(f"unknown layer kind: {kind!r}")
    return lines, cur_a


def parse_sample_spec(spec):
    """Parse the --sample CLI spec into (mode, kwargs).

    Forms accepted:
      'greedy'                   → ("greedy", {})
      'temperature:T'            → ("temperature", {"temperature": float(T)})
      'topk:K:T'                 → ("topk", {"topk_k": int(K),
                                              "temperature": float(T)})
    """
    parts = spec.split(":")
    head = parts[0]
    if head == "greedy":
        if len(parts) != 1:
            raise SystemExit(
                f"--sample greedy takes no parameters; got {spec!r}")
        return ("greedy", {})
    if head == "temperature":
        if len(parts) != 2:
            raise SystemExit(
                f"--sample temperature:T (e.g. 'temperature:0.7'); "
                f"got {spec!r}")
        return ("temperature", {"temperature": float(parts[1])})
    if head == "topk":
        if len(parts) != 3:
            raise SystemExit(
                f"--sample topk:K:T (e.g. 'topk:50:0.9'); got {spec!r}")
        return ("topk", {"topk_k": int(parts[1]),
                         "temperature": float(parts[2])})
    raise SystemExit(
        f"--sample: unknown mode {head!r}.  Expected greedy, "
        f"temperature:T, or topk:K:T.")


def gen_decoder_main(arch, args, out_name, embed_dir, scales):
    """Decoder-mode main(): one persistent KV cache per attention layer,
    and a prefill+sample loop that runs the per-token forward once
    per prompt token (cache-warming) and once per generated token
    (sample, append, repeat).

    Required arch shape:
      arch[0]            ["embedding", vocab, d_model, 1]
      ...interior layers...   (LN, MH-KV, save/add residual, Linear, GELU)
      arch[-1]           ["linear", d_model, vocab]   (LM head)

    The embedding is consumed manually inside the loop (single-token
    lookup from cur_tok), so gen_forward sees `arch[1:]` and emits
    the rest of the per-token forward unchanged.

    --input-file is a raw byte stream of prompt token IDs (1, 2, or 4
    bytes per token, sized from vocab).  --max-new-tokens controls
    how many additional tokens to sample after the prompt.
    """
    if arch[0][0] != "embedding":
        raise SystemExit(
            "decoder mode: arch must start with `embedding`")
    if arch[-1][0] != "linear":
        raise SystemExit(
            "decoder mode: arch must end with `linear` (LM head)")
    vocab, d_model, _seq = arch[0][1:]
    lm_in, lm_out = arch[-1][1:]
    if lm_out != vocab:
        raise SystemExit(
            f"decoder mode: LM head out_features ({lm_out}) must match "
            f"embedding vocab ({vocab})")
    if lm_in != d_model:
        raise SystemExit(
            f"decoder mode: LM head in_features ({lm_in}) must match "
            f"embedding d_model ({d_model})")
    if args.input_format == "embedded" and not args.input_file:
        raise SystemExit(
            "decoder mode --input-format embedded requires --input-file "
            "<prompt.bin> (raw token IDs, byte-sized from vocab).")
    if args.input_format not in ("embedded", "stdin-tokens"):
        raise SystemExit(
            f"decoder mode supports --input-format embedded (baked-in "
            f"prompt) or stdin-tokens (dynamic prompt from stdin); got "
            f"{args.input_format!r}.")

    quant_active = (args.quantize == "int8")
    # int8 only quantises Linear/conv2d_3x3_p1 weights (see QUANTISABLE in
    # main()); embedding / positional_embedding / multi_head_attention_kv /
    # layernorm tensors stay f32, so the KV-cache attention path keeps its
    # f32 inputs.  The Linear int8 path flows through gen_load/gen_forward
    # via the existing --quantize int8 wiring.
    weight_decls = [] if quant_active else gen_weight_decls(arch, args.embed)
    act_decls, sizes = gen_act_decls(arch)
    load = gen_load(arch, embed_dir, out_name, args.embed,
                    args.quantize, scales, False,
                    int4_embedding=getattr(args, "int4_embedding", False),
                    int4_linear=getattr(args, "int4_linear", False))
    # If a positional_embedding immediately follows the embedding, the
    # decoder emits a per-token wpe add inside the loop (below) and the
    # forward pass skips the entry to avoid double-emission.
    has_pe = (len(arch) > 1 and arch[1][0] == "positional_embedding")
    fwd_arch = arch[2:] if has_pe else arch[1:]
    fwd, last = gen_forward(fwd_arch, scales if quant_active else None, False,
                            int4_linear=getattr(args, "int4_linear", False))

    tb = _embedding_token_bytes(vocab)

    body = []
    body.append("fn main() -> i32 {")
    body += weight_decls
    body += load
    body += act_decls
    body += gen_scratch(arch)
    body.append("")
    body.append(f"    // decoder loop: prefill prompt, then sample {args.max_new_tokens}")
    if args.input_format == "stdin-tokens":
        # Dynamic prompt from stdin.  Wire format:
        #   [u32 LE prompt_len_in_tokens]
        #   [u32 LE seed]
        #   [N · {tb}-byte token IDs LE]
        # Each token is tb bytes wide (1 / 2 / 4 derived from vocab).
        # Seed is consumed only in non-greedy modes; greedy still reads
        # and ignores it so the wire format is constant.  Buffer is
        # sized at pack time; --max-prompt-tokens caps the caller's
        # prompt length (default 512).
        max_prompt_bytes = args.max_prompt_tokens * tb
        # arr_new(N) allocates N i64 slots = 8N bytes; round up.
        toks_slots = (max_prompt_bytes + 7) // 8 + 1
        body.append(f"    let _toks: i32 = arr_new({toks_slots});")
        body.append(f"    let _len_buf: i32 = arr_new(1);")
        body.append(f"    file_read(0, _len_buf, 4);")
        body.append(f"    let prompt_len: i32 = byte_at(_len_buf, 0)")
        body.append(f"                       + byte_at(_len_buf, 1) * 256")
        body.append(f"                       + byte_at(_len_buf, 2) * 65536")
        body.append(f"                       + byte_at(_len_buf, 3) * 16777216;")
        body.append(f"    let _seed_buf: i32 = arr_new(1);")
        body.append(f"    file_read(0, _seed_buf, 4);")
        body.append(f"    let _stdin_seed: i32 = byte_at(_seed_buf, 0)")
        body.append(f"                        + byte_at(_seed_buf, 1) * 256")
        body.append(f"                        + byte_at(_seed_buf, 2) * 65536")
        body.append(f"                        + byte_at(_seed_buf, 3) * 16777216;")
        body.append(f"    file_read(0, _toks, prompt_len * {tb});")
    else:
        body.append(f"    let _toks: i32 = "
                    f"embed_file_bytes(\"{args.input_file}\");")
        body.append(f"    let _toks_len: i32 = arr_len(_toks);")
        if tb == 1:
            body.append(f"    let prompt_len: i32 = _toks_len;")
        else:
            body.append(f"    let prompt_len: i32 = _toks_len / {tb};")
    body.append(f"    let total: i32 = prompt_len + {args.max_new_tokens} - 1;")
    body.append(f"    let cur_tok: i32 = 0;")
    body.append(f"    let t: i32 = 0;")
    sample_mode, sample_kwargs = parse_sample_spec(args.sample)
    if sample_mode != "greedy":
        # 32-bit LCG (glibc rand-style constants); each step advances
        # rng_state and we feed the low 16 bits to sample_temperature_f32
        # as a uniform [0, 1) draw.  In stdin-tokens mode the caller
        # provides a fresh seed per turn (so repeated prompts diverge);
        # otherwise the seed is baked at pack time.
        if args.input_format == "stdin-tokens":
            body.append(f"    let rng_state: i32 = _stdin_seed;")
        else:
            body.append(f"    let rng_state: i32 = {args.rng_seed};")
    body.append(f"    while (t < total) {{")
    # Get current token: prompt or previous-step argmax.
    body.append(f"        if (t < prompt_len) {{")
    if tb == 1:
        body.append(f"            cur_tok = byte_at(_toks, t);")
    elif tb == 2:
        body.append(f"            cur_tok = byte_at(_toks, t * 2) + "
                    f"byte_at(_toks, t * 2 + 1) * 256;")
    else:   # tb == 4
        body.append(f"            cur_tok = byte_at(_toks, t * 4) + "
                    f"byte_at(_toks, t * 4 + 1) * 256 + "
                    f"byte_at(_toks, t * 4 + 2) * 65536 + "
                    f"byte_at(_toks, t * 4 + 3) * 16777216;")
    body.append(f"        }}")
    # Embedding lookup: a0[:] = Wemb0[cur_tok, :]
    # Three paths sharing the same scale sidecar:
    #   int4 (--int4-embedding): two nibbles per byte, branchless
    #     sign-extend via ((n + 8) % 16) - 8, walk d_model/2 bytes.
    #   int8 (--quantize int8):   one byte per element, arr_i8_get
    #     does the sign-extend in hardware.
    #   fp32 (no quantisation):   arr_f32_copy_slice as before.
    emb_quantised = ("embedding", 0, "", "W") in scales
    int4_emb = getattr(args, "int4_embedding", False)
    if emb_quantised and int4_emb:
        if d_model % 2 != 0:
            raise SystemExit(
                f"decoder int4 embedding requires even d_model; got {d_model}")
        half = d_model // 2
        body.append(
            f"        let _emb_scale: f64 = "
            f"arr_f32_get(Wemb0_scales, cur_tok);")
        body.append(f"        let _ei: i32 = 0;")
        body.append(f"        let _emb_off: i32 = cur_tok * {half};")
        body.append(f"        while (_ei < {half}) {{")
        body.append(
            f"            let _raw: i32 = byte_at(Wemb0, _emb_off + _ei);")
        body.append(f"            let _lo: i32 = _raw % 16;")
        body.append(f"            let _hi: i32 = _raw / 16;")
        # Sign-extend 4-bit values: ((n + 8) % 16) - 8 maps [0..15] to
        # [-8..7] without any branch.
        body.append(f"            let _los: i32 = (_lo + 8) % 16 - 8;")
        body.append(f"            let _his: i32 = (_hi + 8) % 16 - 8;")
        body.append(
            f"            arr_f32_set(a0, _ei * 2, "
            f"int_to_float(_los) * _emb_scale);")
        body.append(
            f"            arr_f32_set(a0, _ei * 2 + 1, "
            f"int_to_float(_his) * _emb_scale);")
        body.append(f"            _ei = _ei + 1;")
        body.append(f"        }}")
    elif emb_quantised:
        body.append(
            f"        let _emb_scale: f64 = "
            f"arr_f32_get(Wemb0_scales, cur_tok);")
        body.append(f"        let _ei: i32 = 0;")
        body.append(f"        let _emb_off: i32 = cur_tok * {d_model};")
        body.append(f"        while (_ei < {d_model}) {{")
        body.append(
            f"            arr_f32_set(a0, _ei, "
            f"int_to_float(arr_i8_get(Wemb0, _emb_off + _ei)) * _emb_scale);")
        body.append(f"            _ei = _ei + 1;")
        body.append(f"        }}")
    else:
        body.append(f"        arr_f32_copy_slice(Wemb0, cur_tok * {d_model}, "
                    f"a0, 0, {d_model});")
    # Optional positional embedding: a0 += Wpos0[t, :] (per absolute
    # position).  Inline scalar add since arr_f32_add_scaled would
    # need a separate sliced buffer for one row of d_model elements.
    if has_pe:
        max_pos_pe, d_model_pe, _seq_pe = arch[1][1:]
        body.append(f"        let _pi: i32 = 0;")
        body.append(f"        while (_pi < {d_model_pe}) {{")
        body.append(
            f"            arr_f32_set(a0, _pi, "
            f"arr_f32_get(a0, _pi) + "
            f"arr_f32_get(Wpos0, t * {d_model_pe} + _pi));")
        body.append(f"            _pi = _pi + 1;")
        body.append(f"        }}")
    # Inline the per-token forward (sized for indent 8 inside `while`).
    body += [("    " + l) for l in fwd]
    # Sample the next token from the LM head logits at a{last}.
    if sample_mode == "greedy":
        body.append(f"        let next_tok: i32 = sample_greedy_f32(a{last});")
    else:
        if sample_mode == "topk":
            # Truncate the distribution before softmax: any logit below
            # the k-th largest gets a -1e6 sentinel so softmax assigns
            # it ~0 probability.
            body.append(
                f"        sample_topk_filter_f32(a{last}, "
                f"{sample_kwargs['topk_k']});")
        # Advance the LCG, harvest 16 bits, scale to [0, 1).
        body.append(f"        rng_state = rng_state * 1103515245 + 12345;")
        body.append(f"        let _bits: i32 = rng_state & 65535;")
        body.append(
            f"        let _unit: f64 = int_to_float(_bits) / 65536.0;")
        body.append(
            f"        let next_tok: i32 = sample_temperature_f32("
            f"a{last}, {sample_kwargs['temperature']!r}, _unit);")
    # Optional EOS: stop generating BEFORE printing the EOS token so
    # the consumer sees only real content.  Triggered only past the
    # prefill window (t >= prompt_len - 1) — an EOS-shaped logit
    # mid-prompt is by definition impossible (we override cur_tok
    # from the prompt buffer there) but cheap to skip.
    if args.eos_token is not None:
        body.append(
            f"        if (t >= prompt_len - 1 && next_tok == "
            f"{args.eos_token}) {{ break; }}")
    # Print only past prompt (the very first emitted token follows the
    # last prompt position).
    body.append(f"        if (t >= prompt_len - 1) {{ print_int(next_tok); }}")
    body.append(f"        cur_tok = next_tok;")
    body.append(f"        t = t + 1;")
    body.append(f"    }}")
    body.append("    return 0;")
    body.append("}")
    return "\n".join(body)


def gen_main(arch, args, out_name, embed_dir, scales):
    # When --quantize int8 is in effect we don't pre-allocate W/b
    # tensors; the load step both allocates and fills them.
    quant_active = (args.quantize == "int8")
    weight_decls = [] if quant_active else gen_weight_decls(arch, args.embed)
    act_decls, sizes = gen_act_decls(arch)
    # Pick the multi-channel int8 conv strategy based on workload:
    # single-shot CLI wins from runtime dequant (no startup pass);
    # batch loaders amortise the startup dequant across all samples
    # so they prefer the f32 kernel.
    multi_int8_runtime = (quant_active and args.input_format == "stdin")
    load = gen_load(arch, embed_dir, out_name, args.embed,
                    args.quantize, scales, multi_int8_runtime)
    fwd, last = gen_forward(arch, scales if quant_active else None,
                            multi_int8_runtime)

    # n_in = the *size in elements* of the input buffer a0.  For an MLP
    # this is the first layer's in_features; for a CNN it's c_in*28*28;
    # for an attention-only model it's seq*d_in.
    n_in  = sizes[0]
    # n_out = element count of the final activation buffer.  Always
    # derived from sizes[-1] (gen_act_decls handles batched Linear,
    # attention output shape, classifier head, etc.) so the
    # no-argmax printing loop and the argmax search both walk the
    # whole final tensor.
    n_out = sizes[-1]

    body = []
    body.append(f"fn main() -> i32 {{")
    body += weight_decls
    body += load
    body += act_decls
    body += gen_scratch(arch)
    body.append("")

    if args.input_format == "stdin":
        # Single-shot CLI: read n_in raw bytes from stdin, run one forward
        # pass, print the predicted class to stdout.  No batch loop, no
        # external image files — the binary is fully self-contained.
        slots = (n_in + 7) // 8
        body.append(f"    let inbuf: i32 = arr_new({slots});")
        body.append(f"    file_read(0, inbuf, {n_in});")
        body.append("    let i: i32 = 0;")
        body.append(f"    while (i < {n_in}) {{")
        body.append("        let px: i32 = byte_at(inbuf, i);")
        body.append(f"        arr_f32_set(a0, i, (int_to_float(px) / 255.0 - {args.mean}) / {args.std});")
        body.append("        i = i + 1;")
        body.append("    }")
        body += [("    " + l.lstrip()) for l in fwd]
        if not args.no_argmax:
            body.append(f"    print_int(argmax_f32(a{last}, {n_out}));")
        else:
            body.append(f"    let i2: i32 = 0;")
            body.append(f"    while (i2 < {n_out}) {{")
            body.append(f"        print_f64(arr_f32_get(a{last}, i2), 6);")
            body.append("        i2 = i2 + 1;")
            body.append("    }")
        body.append("    return 0;")
        body.append("}")
        return "\n".join(body)

    if args.input_format == "embedded":
        # Embedded-input mode: a fixed input is baked into .text via
        # embed_file at pack time.  Use case: deterministic regression
        # tests, demo binaries, fixed-input batch evaluation against a
        # reference checkpoint.  The standard --no-argmax flag controls
        # whether the final activation is printed as argmax or as the
        # full element-by-element f32 dump (for attention output, etc.).
        if not args.input_file:
            raise SystemExit(
                "--input-format embedded requires --input-file <path>.")
        if arch[0][0] == "embedding":
            # Embedding-first arch: the file is a raw byte stream of
            # token IDs (1 byte per token).  The embedding layer's
            # forward decodes them into the activation buffer.
            body.append(
                f"    let _toks_raw: i32 = embed_file_bytes(\"{args.input_file}\");")
        else:
            # Standard f32-input case: copy n_in f32 elements straight
            # into a0.
            body.append(f"    let X_in: i32 = embed_file(\"{args.input_file}\");")
            body.append(f"    arr_f32_copy_slice(X_in, 0, a0, 0, {n_in});")
        body += [("    " + l.lstrip()) for l in fwd]
        if not args.no_argmax:
            body.append(f"    print_int(argmax_f32(a{last}, {n_out}));")
        else:
            body.append(f"    let i2: i32 = 0;")
            body.append(f"    while (i2 < {n_out}) {{")
            body.append(f"        print_f64(arr_f32_get(a{last}, i2), 6);")
            body.append("        i2 = i2 + 1;")
            body.append("    }")
        body.append("    return 0;")
        body.append("}")
        return "\n".join(body)

    # mnist mode: batch over the test set, report accuracy.
    if not args.input_images or not args.input_labels:
        raise SystemExit("--input-format mnist requires --input-images and --input-labels.")
    body.append(f"    let img_path: i32 = str_new(\"{args.input_images}\");")
    body.append(f"    let lbl_path: i32 = str_new(\"{args.input_labels}\");")
    body.append(f"    let imgs: i32 = load_byte_file(img_path, {args.n_test} * {n_in} + 16);")
    body.append(f"    let lbls: i32 = load_byte_file(lbl_path, {args.n_test} + 8);")
    body.append("")
    body.append(f"    print_str(\"aricode-pack inference, n={args.n_test}\");")
    body.append("    let correct: i32 = 0;")
    body.append("    let i: i32 = 0;")
    body.append(f"    while (i < {args.n_test}) {{")
    body.append(f"        load_image_at(imgs, 16 + i * {n_in}, a0);")
    body.append("        let lbl: i32 = byte_at(lbls, 8 + i);")
    body += [("        " + l.lstrip()) for l in fwd]
    if not args.no_argmax:
        body.append(f"        if (argmax_f32(a{last}, {n_out}) == lbl) {{ correct = correct + 1; }}")
    body.append("        i = i + 1;")
    body.append("    }")
    body.append("")
    body.append("    print_str(\"accuracy:\");")
    body.append(f"    print_f64(int_to_float(correct) / int_to_float({args.n_test}), 4);")
    body.append("    return 0;")
    body.append("}")
    return "\n".join(body)


# ──────────────────────────────────────────────────────────────────────
#  Top-level
# ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="aricode-pack")
    p.add_argument("--checkpoint", required=True,
                   help="PyTorch state_dict .pt or .pth file (or a model "
                        "with .state_dict() attribute).")
    p.add_argument("--arch", default=None,
                   help="Architecture spec: JSON list of [kind, ...args].  "
                        "Required for the pack flow; omitted with --infer-arch.")
    p.add_argument("--keys", default="fc{idx_plus_1}.{kind}",
                   help="state_dict key template.  {idx} = layer index "
                        "(0-based), {idx_plus_1} = 1-based, {kind} = "
                        "weight|bias.  Default: fc{idx_plus_1}.{kind}.")
    p.add_argument("--input-format",
                   choices=("mnist", "stdin", "embedded", "stdin-tokens"),
                   default="mnist",
                   help="`stdin-tokens` is the dynamic-prompt mode for "
                        "decoder LLMs: the binary reads a 4-byte uint32 "
                        "LE prompt-length prefix followed by that many "
                        "uint8/16/32 LE token IDs from stdin (byte width "
                        "auto-sized from vocab).  Prompts up to "
                        "--max-prompt-tokens long, default 512.")
    p.add_argument("--max-prompt-tokens", type=int, default=512,
                   help="Decoder mode + --input-format stdin-tokens: "
                        "upper bound on prompt length the binary will "
                        "accept.  Buffer is sized at pack time.")
    p.add_argument("--input-file",
                   help="path to .f32 input file (used by --input-format embedded)")
    p.add_argument("--input-images", default=None,
                   help="Path to raw MNIST images file (mnist mode only).")
    p.add_argument("--input-labels", default=None,
                   help="Path to raw MNIST labels file (mnist mode only).")
    p.add_argument("--n-test", type=int, default=10000,
                   help="Number of test samples to score in mnist mode.")
    p.add_argument("--mean", type=float, default=0.1307)
    p.add_argument("--std",  type=float, default=0.3081)
    p.add_argument("--no-argmax", action="store_true",
                   help="Skip argmax + accuracy reporting (e.g. when the "
                        "model isn't a classifier).")
    p.add_argument("--max-new-tokens", type=int, default=16,
                   help="Decoder mode only: number of tokens to sample "
                        "after consuming the prompt.  Default: 16.")
    p.add_argument("--sample", default="greedy",
                   help="Decoder mode sampling strategy.  Forms: "
                        "'greedy' (default, deterministic argmax); "
                        "'temperature:T' (e.g. 'temperature:0.7'); "
                        "'topk:K:T' (top-K filter then temperature, e.g. "
                        "'topk:50:0.9').  K=1 reduces to greedy.")
    p.add_argument("--eos-token", type=int, default=None,
                   help="Decoder mode only: stop generating when this "
                        "token id is sampled.  GPT-2's <|endoftext|> is "
                        "50256; Llama's </s> is 2.  Default: no early "
                        "stop, run full --max-new-tokens.")
    p.add_argument("--rng-seed", type=int, default=20260502,
                   help="Decoder mode + non-greedy sampling: initial "
                        "state for the per-step LCG that produces the "
                        "uniform draw passed to sample_temperature_f32.  "
                        "Default chosen for stable regression-test "
                        "reproducibility.")
    p.add_argument("--out", default=None,
                   help="Output base name.  <out>.ari is always written.  "
                        "Without --embed, a single <out>.f32 sidecar is "
                        "written too; with --embed, per-tensor "
                        "<out>_W{i}.f32 / <out>_b{i}.f32 staging files are "
                        "produced (the compiler bakes them into the binary "
                        "and they're no longer needed at runtime).  "
                        "Required for the pack flow; omitted with --infer-arch.")
    p.add_argument("--embed", action="store_true",
                   help="Emit a single self-contained binary: weights "
                        "embedded inline via embed_file, no runtime file "
                        "I/O for the model.  The MNIST images / labels "
                        "still come from disk.")
    p.add_argument("--quantize", choices=("none", "int8"), default="none",
                   help="Quantisation scheme for the weight blob.  int8 "
                        "stores each tensor as signed-byte values + a "
                        "single f32 scale per tensor; the binary shrinks "
                        "by ~4× and the .ari preamble dequantises back "
                        "to f32 at startup.  Implies --embed.")
    p.add_argument("--int4-linear", action="store_true",
                   help="Use int4 (instead of int8) for every Linear "
                        "layer's weight matrix.  On decoder models that "
                        "means the LM head — the second-largest tensor "
                        "after the embedding.  Per-row symmetric int4, "
                        "two nibbles per byte, dequantised by an inline "
                        "matmul (no SIMD kernel — slower than int8 but "
                        "saves ~7 MB on the LM head of a 32M model).  "
                        "Costs ~+4.5% perplexity at this scale.  "
                        "Requires --quantize int8.")
    p.add_argument("--int4-embedding", action="store_true",
                   help="Use int4 (instead of int8) for the token embedding "
                        "matrix only — matmul layers stay int8.  Each token "
                        "row is symmetrically quantised to [-8, 7] with its "
                        "own f32 scale; two nibbles pack into one byte for "
                        "storage.  Costs ~+0.5% perplexity on the validated "
                        "TinyLM (32M params, vocab 50257) and saves another "
                        "~6 MB binary on top of full int8.  Requires "
                        "--quantize int8.")
    p.add_argument("--infer-arch", action="store_true",
                   help="Walk the checkpoint and print a starter arch.json "
                        "for sequential MLP / CNN architectures.  Inserts "
                        "ReLU between weight layers as the default "
                        "activation; you can edit the output if your model "
                        "uses different activations or layout.  Skips the "
                        "pack step entirely (no .ari emitted).  Useful for "
                        "models from HuggingFace where you don't want to "
                        "hand-write the architecture descriptor.")
    return p.parse_args()


def expand_keys(template):
    """Adapt the template to a callable: keys(idx, kind)."""
    def keys(idx: int, kind: str):
        return template.format(idx=idx, idx_plus_1=idx + 1, kind=kind)
    return keys


def sd_lookup(sd, candidates):
    """Return the first key in `candidates` that's present in `sd`,
    or None if none match.  Used for conv weights where the natural
    state_dict key (conv.weight) doesn't share the linear naming
    template."""
    for k in candidates:
        if k in sd:
            return k
    return None


def infer_arch_from_state_dict(sd):
    """Best-effort architecture inference from a flat state_dict.

    Heuristics:
    - Group keys by their stem (the part before .weight / .bias).
    - Walk stems in the order they appear in the dict (PyTorch /
      safetensors preserve insertion order, which usually mirrors
      forward-pass order for sequential models).
    - 4-D weight tensor (C_out, C_in, kH, kW) → conv2d_3x3_p1 (we only
      support 3×3 pad 1 today; assert kH = kW = 3).
    - 2-D weight tensor (out, in) → linear.
    - Insert ReLU between consecutive weight layers — almost always the
      right default for the architectures the pack tool can deploy
      today; users edit if they have something else.
    - When a conv-output's flattened size doesn't equal the next
      linear's in_features, insert maxpool_2x2 + flatten before it.
      We only know how to suggest the 28×28 → 14×14 maxpool we have a
      builtin for, so the inference is correct only when the
      architecture matches the MNIST CNN family.

    Returns the inferred arch as a list of [kind, ...args] entries.
    Caller is responsible for sanity-checking the result; this is a
    starting point, not a contract.
    """
    # Walk stems in insertion order.
    stems = []
    seen = set()
    for k in sd.keys():
        for suf in (".weight", ".bias"):
            if k.endswith(suf):
                stem = k[:-len(suf)]
                if stem not in seen:
                    stems.append(stem)
                    seen.add(stem)
                break

    # Pre-pass: detect contiguous q_proj / k_proj / v_proj triples and
    # rewrite the stem list so they emit one attention entry instead of
    # three Linears.  HuggingFace convention; PyTorch nn.MultiheadAttention's
    # combined `in_proj_weight` would need a separate splitter.
    def _is_qkv_triple(s_q, s_k, s_v):
        return (s_q.endswith("q_proj") and s_k.endswith("k_proj")
                and s_v.endswith("v_proj")
                and s_q[:-6] == s_k[:-6] == s_v[:-6])

    rewritten = []
    skip = 0
    for i, stem in enumerate(stems):
        if skip:
            skip -= 1
            continue
        if (i + 2 < len(stems)
                and _is_qkv_triple(stem, stems[i+1], stems[i+2])):
            wq = sd[stem + ".weight"]
            shape = tuple(wq.shape) if hasattr(wq, "shape") else tuple(wq.size())
            if len(shape) == 2:
                d_head, d_in = int(shape[0]), int(shape[1])
                # seq is a runtime parameter; the state_dict doesn't carry
                # it.  Emit a placeholder the user fills in (or pass it
                # via a future --seq-len flag).  Default 128 ≈ typical
                # sentence-transformer max_seq.
                rewritten.append(("__attention__", d_in, d_head, 128, 0))
                skip = 2
                continue
        rewritten.append(stem)

    arch = []
    last_spatial = None   # (C_out, 28, 28) after a conv layer; None otherwise
    for i, stem in enumerate(rewritten):
        if isinstance(stem, tuple) and stem[0] == "__attention__":
            _, d_in, d_head, seq, causal = stem
            arch.append(["attention", seq, d_in, d_head, causal])
            # Reset spatial tracking — attention output is [seq, d_head] flat,
            # not spatial; if a Linear follows, it'll connect by element count.
            last_spatial = None
            if i < len(rewritten) - 1:
                arch.append(["relu"])
            continue
        wkey = stem + ".weight"
        if wkey not in sd:
            continue
        W = sd[wkey]
        shape = tuple(W.shape) if hasattr(W, "shape") else tuple(W.size())

        if len(shape) == 4:
            # Conv2d (C_out, C_in, kH, kW)
            c_out, c_in, kh, kw = shape
            if (kh, kw) != (3, 3):
                raise SystemExit(
                    f"infer-arch: conv kernel {kh}×{kw} for layer "
                    f"{stem!r} not supported (only 3×3 pad-1 today).")
            arch.append(["conv2d_3x3_p1", int(c_in), int(c_out)])
            last_spatial = (int(c_out), 28, 28)
        elif len(shape) == 2:
            out_f, in_f = shape
            # If we just left a spatial layer, insert pool + flatten
            # before this Linear when the dims look right.
            if last_spatial is not None:
                C, H, W_ = last_spatial
                # Try maxpool 2×2 → 14×14: pooled flat = C·14·14
                pooled = C * (H // 2) * (W_ // 2)
                if pooled == in_f:
                    arch.append(["maxpool_2x2", C])
                    arch.append(["flatten"])
                elif C * H * W_ == in_f:
                    arch.append(["flatten"])
                else:
                    raise SystemExit(
                        f"infer-arch: can't bridge spatial output "
                        f"{last_spatial} to linear in_features={in_f}.  "
                        "Add a manual maxpool/flatten/etc. step.")
                last_spatial = None
            arch.append(["linear", int(in_f), int(out_f)])
        else:
            raise SystemExit(
                f"infer-arch: layer {stem!r} has weight shape {shape}; "
                "only 2-D (linear) and 4-D (conv) are supported.")

        # Insert ReLU between layers, except after the last one (the
        # head's logits aren't activation-followed in classifiers).
        if i < len(rewritten) - 1:
            arch.append(["relu"])

    return arch


def emit_arch_json(arch):
    """Pretty-print an inferred arch in the format aricode-pack reads."""
    import json
    lines = ["["]
    for i, layer in enumerate(arch):
        suffix = "," if i + 1 < len(arch) else ""
        lines.append(f"    {json.dumps(layer)}{suffix}")
    lines.append("]")
    return "\n".join(lines)


def main():
    args = parse_args()

    if args.infer_arch:
        # --infer-arch short-circuits the pack pipeline: we only need
        # the checkpoint, walk it, print a starter arch.json, exit.
        sd = load_state_dict(args.checkpoint)
        arch = infer_arch_from_state_dict(sd)
        print(f"// Inferred from {args.checkpoint}")
        print(f"// {len(sd)} tensors, {sum(1 for k in sd if k.endswith('.weight'))} weight tensors")
        print(f"// Insert manual edits if your model uses a non-ReLU activation,")
        print(f"// has different layer ordering, or needs additional pool/flatten.")
        print(emit_arch_json(arch))
        return

    if args.arch is None or args.out is None:
        missing = [a for a, v in (("--arch", args.arch), ("--out", args.out))
                   if v is None]
        raise SystemExit(f"{', '.join(missing)} required for the pack flow.  "
                         "Run with --infer-arch to get a starter arch.json.")
    arch = json.loads(Path(args.arch).read_text())
    if not isinstance(arch, list):
        raise SystemExit("--arch must contain a JSON list of [kind, ...args].")
    arch = [tuple(layer) for layer in arch]

    sd = load_state_dict(args.checkpoint)

    keys = expand_keys(args.keys)
    # Adapter to fit collect_weights' template-arg expectation.
    def keypat(idx, kind):
        return keys(idx, kind)

    # collect_weights expects a key template via .format with {idx, kind}
    # — simpler to pre-resolve to a dict.
    resolved = {}
    li = 0
    for kind, *largs in arch:
        if kind == "linear":
            resolved[("weight", li)] = keys(li, "weight")
            resolved[("bias",   li)] = keys(li, "bias")
            li += 1

    # Collect (kind, idx, W, b) tuples in arch order so the staging-file
    # writer can name them consistently.  `arch_walk` supplies the
    # shared per-kind index — the same one weight_tensors / gen_scratch
    # / gen_forward consume, so naming stays consistent across the four
    # walkers without needing five hand-rolled `<op>i += 1` blocks.
    collected = []   # list of (kind, idx, suffix, W_array, b_array)
    import numpy as _np
    for kind, idx, largs in arch_walk(arch):
        if idx is None:
            continue   # pseudo-op (residual / activation / flatten) — no weights.
        if kind == "linear":
            in_f, out_f = largs
            wkey = keys(idx, "weight")
            bkey = keys(idx, "bias")
            if wkey not in sd or bkey not in sd:
                raise SystemExit(
                    f"linear #{idx}: missing '{wkey}' or '{bkey}' in checkpoint. "
                    f"Available keys: {sorted(sd.keys())}"
                )
            W = sd[wkey].detach().float().cpu().numpy()
            b = sd[bkey].detach().float().cpu().numpy()
            if W.shape != (out_f, in_f) or b.shape != (out_f,):
                raise SystemExit(
                    f"linear #{idx}: shape mismatch.  expected ({out_f},{in_f}) "
                    f"+ ({out_f},); got {tuple(W.shape)} + {tuple(b.shape)}."
                )
            collected.append(("linear", idx, "", W, b))
        elif kind == "conv2d_3x3_p1":
            c_in, c_out = largs
            # Conv keys default to {idx_plus_1}.weight; if the user kept
            # PyTorch's natural `conv.weight` naming, --keys can be
            # remapped per call.  By default the same template is shared
            # — works for the demo where the conv's index_plus_1 = 1
            # (i.e. "1.weight") doesn't clash with fc names.  Real users
            # can pass --conv-keys later; v0.4 supports the simple case.
            wkey = sd_lookup(sd, ["conv.weight",
                                  f"conv{idx + 1}.weight",
                                  f"layers.{idx}.weight",
                                  keys(idx, "weight")])
            bkey = sd_lookup(sd, ["conv.bias",
                                  f"conv{idx + 1}.bias",
                                  f"layers.{idx}.bias",
                                  keys(idx, "bias")])
            if wkey is None or bkey is None:
                raise SystemExit(
                    f"conv2d_3x3_p1 #{idx}: cannot find weight/bias in "
                    f"checkpoint.  Tried: conv.weight / layers.{idx}.weight / "
                    f"{keys(idx, 'weight')}.  Available: {sorted(sd.keys())}"
                )
            W = sd[wkey].detach().float().cpu().numpy()
            b = sd[bkey].detach().float().cpu().numpy()
            if W.shape != (c_out, c_in, 3, 3) or b.shape != (c_out,):
                raise SystemExit(
                    f"conv2d_3x3_p1 #{idx}: shape mismatch.  expected "
                    f"({c_out},{c_in},3,3) + ({c_out},); got "
                    f"{tuple(W.shape)} + {tuple(b.shape)}."
                )
            # Flatten (C_out, C_in, 3, 3) → (C_out, C_in*9) row-major to
            # match arr_f32_conv2d_3x3_p1's expected weight layout.
            W = W.reshape(c_out, c_in * 9)
            collected.append(("conv2d_3x3_p1", idx, "", W, b))
        elif kind == "attention":
            # ["attention", seq, d_in, d_head, causal]
            # Three projections — Q, K, V — each loaded as a separate
            # (W, b) tensor.  Standard HuggingFace / PyTorch convention
            # is q_proj.weight / k_proj.weight / v_proj.weight; we also
            # accept attn{idx}.q_proj.weight for stacked-block models.
            seq, d_in, d_head, causal = largs
            for proj in ("q", "k", "v"):
                candidates_w = [
                    f"{proj}_proj.weight",
                    f"attn.{proj}_proj.weight",
                    f"attention.{proj}_proj.weight",
                    f"layers.{idx}.attn.{proj}_proj.weight",
                    f"layers.{idx}.{proj}_proj.weight",
                ]
                candidates_b = [c.replace(".weight", ".bias") for c in candidates_w]
                wkey = sd_lookup(sd, candidates_w)
                bkey = sd_lookup(sd, candidates_b)
                if wkey is None or bkey is None:
                    raise SystemExit(
                        f"attention #{idx}: missing {proj}_proj weight or bias.  "
                        f"Tried: {candidates_w[0]} / "
                        f"{candidates_w[3]}.  "
                        f"Available: {sorted(sd.keys())}"
                    )
                W = sd[wkey].detach().float().cpu().numpy()
                b = sd[bkey].detach().float().cpu().numpy()
                if W.shape != (d_head, d_in) or b.shape != (d_head,):
                    raise SystemExit(
                        f"attention #{idx} {proj}_proj: shape mismatch. "
                        f"expected ({d_head},{d_in}) + ({d_head},); got "
                        f"{tuple(W.shape)} + {tuple(b.shape)}."
                    )
                collected.append(("attention", idx, proj, W, b))
        elif kind == "multi_head_attention":
            seq, d_model, n_heads, causal = largs
            # Q/K/V/O projections — same key-pattern set as single-head
            # attention, plus an out_proj.  All four shaped (d_model, d_model).
            # We accept both the bare HuggingFace `q_proj` convention
            # and distilbert's nested `transformer.layer.{i}.attention.q_lin`
            # form (note: distilbert uses `_lin` not `_proj`).
            proj_to_keys = {
                "q": ("q_proj", "q_lin"),
                "k": ("k_proj", "k_lin"),
                "v": ("v_proj", "v_lin"),
                "o": ("out_proj", "out_lin"),
            }
            for proj in ("q", "k", "v", "o"):
                hf_key, dbert_key = proj_to_keys[proj]
                candidates_w = [
                    f"{hf_key}.weight",
                    f"attn.{hf_key}.weight",
                    f"attention.{hf_key}.weight",
                    f"layers.{idx}.attn.{hf_key}.weight",
                    f"layers.{idx}.{hf_key}.weight",
                    # Llama prefill mode (mirrors the GQA-KV path).
                    f"model.layers.{idx}.self_attn.{hf_key}.weight",
                    # distilbert: transformer.layer.{i}.attention.q_lin.weight
                    f"transformer.layer.{idx}.attention.{dbert_key}.weight",
                    f"layer.{idx}.attention.{dbert_key}.weight",
                ]
                candidates_b = [c.replace(".weight", ".bias") for c in candidates_w]
                wkey = sd_lookup(sd, candidates_w)
                bkey = sd_lookup(sd, candidates_b)
                if wkey is None or bkey is None:
                    raise SystemExit(
                        f"multi_head_attention #{idx}: missing {proj} "
                        f"weight or bias.  Tried: {candidates_w[0]} / "
                        f"{candidates_w[3]}.  Available: {sorted(sd.keys())}"
                    )
                W = sd[wkey].detach().float().cpu().numpy()
                b = sd[bkey].detach().float().cpu().numpy()
                if W.shape != (d_model, d_model) or b.shape != (d_model,):
                    raise SystemExit(
                        f"multi_head_attention #{idx} {proj}: shape "
                        f"mismatch.  expected ({d_model},{d_model}) + "
                        f"({d_model},); got {tuple(W.shape)} + "
                        f"{tuple(b.shape)}."
                    )
                collected.append(("multi_head_attention", idx, proj, W, b))
        elif kind == "multi_head_attention_kv":
            # Same weight layout as the one-shot MHA — Q/K/V/O each
            # (d_model, d_model) — so the same key candidate set works.
            # Tagged kv in `collected` so tensor_names yields the same
            # Wm{suffix}{i}/bm{suffix}{i} names (decoder-mode forward
            # already references those).  `idx` is the SHARED
            # multi_head_attention counter (see _ARCH_WALK_COUNTER_KEY)
            # so prefill and decode forms can swap state dicts cleanly.
            max_seq, d_model, n_heads = largs
            proj_to_keys = {
                "q": ("q_proj", "q_lin"),
                "k": ("k_proj", "k_lin"),
                "v": ("v_proj", "v_lin"),
                "o": ("out_proj", "out_lin"),
            }
            for proj in ("q", "k", "v", "o"):
                hf_key, dbert_key = proj_to_keys[proj]
                candidates_w = [
                    f"{hf_key}.weight",
                    f"attn.{hf_key}.weight",
                    f"attention.{hf_key}.weight",
                    f"layers.{idx}.attn.{hf_key}.weight",
                    f"layers.{idx}.{hf_key}.weight",
                    # Llama-style decoder (mirrors GQA-KV path).
                    f"model.layers.{idx}.self_attn.{hf_key}.weight",
                    f"transformer.layer.{idx}.attention.{dbert_key}.weight",
                    f"layer.{idx}.attention.{dbert_key}.weight",
                ]
                candidates_b = [c.replace(".weight", ".bias") for c in candidates_w]
                wkey = sd_lookup(sd, candidates_w)
                bkey = sd_lookup(sd, candidates_b)
                if wkey is None or bkey is None:
                    raise SystemExit(
                        f"multi_head_attention_kv #{idx}: missing {proj} "
                        f"weight or bias.  Tried: {candidates_w[0]} / "
                        f"{candidates_w[3]}.  Available: {sorted(sd.keys())}"
                    )
                W = sd[wkey].detach().float().cpu().numpy()
                b = sd[bkey].detach().float().cpu().numpy()
                if W.shape != (d_model, d_model) or b.shape != (d_model,):
                    raise SystemExit(
                        f"multi_head_attention_kv #{idx} {proj}: shape "
                        f"mismatch.  expected ({d_model},{d_model}) + "
                        f"({d_model},); got {tuple(W.shape)} + "
                        f"{tuple(b.shape)}."
                    )
                collected.append(("multi_head_attention_kv", idx, proj, W, b))
        elif kind == "multi_head_attention_gqa_kv":
            # Llama / TinyLlama / Mistral attention.  Q is (d_model,
            # d_model); K/V are (n_kv_heads*d_head, d_model); O is
            # (d_model, d_model).  No biases.
            max_seq, d_model, n_heads, n_kv_heads, theta = largs
            d_head = d_model // n_heads
            kv_dim = n_kv_heads * d_head
            shape_for = {
                "q": (d_model, d_model),
                "k": (kv_dim,  d_model),
                "v": (kv_dim,  d_model),
                "o": (d_model, d_model),
            }
            # HuggingFace's Llama state_dict naming:
            #   model.layers.{i}.self_attn.q_proj.weight ...
            for proj in ("q", "k", "v", "o"):
                hf = "o_proj" if proj == "o" else f"{proj}_proj"
                candidates_w = [
                    f"model.layers.{idx}.self_attn.{hf}.weight",
                    f"layers.{idx}.self_attn.{hf}.weight",
                    f"transformer.h.{idx}.self_attn.{hf}.weight",
                    # Plain fallback for synthetic test rigs.
                    f"gqa{idx}.{proj}.weight",
                ]
                wkey = sd_lookup(sd, candidates_w)
                if wkey is None:
                    raise SystemExit(
                        f"multi_head_attention_gqa_kv #{idx}: missing "
                        f"{hf} weight.  Tried: "
                        f"{candidates_w[0]} / {candidates_w[3]}.  "
                        f"Available: {sorted(sd.keys())}"
                    )
                W = sd[wkey].detach().float().cpu().numpy()
                if W.shape != shape_for[proj]:
                    raise SystemExit(
                        f"multi_head_attention_gqa_kv #{idx} {proj}: "
                        f"shape mismatch.  expected {shape_for[proj]}; "
                        f"got {tuple(W.shape)}."
                    )
                b = _np.zeros(0, dtype=_np.float32)
                collected.append(("multi_head_attention_gqa_kv", idx, proj, W, b))
        elif kind == "layernorm":
            # ["layernorm", dim] — γ ("LayerNorm.weight") and β ("LayerNorm.bias")
            # both shape (dim,).  HuggingFace naming.  The packer's per-LN
            # counter walks LN entries in arch order, so for typical
            # encoder stacks the embedding-LN is idx=0, then per-block
            # sa_layer_norm and output_layer_norm interleave.  Distilbert
            # / BERT layouts:
            #   idx=0   embeddings.LayerNorm
            #   idx=1   transformer.layer.0.sa_layer_norm
            #   idx=2   transformer.layer.0.output_layer_norm
            #   idx=3   transformer.layer.1.sa_layer_norm
            #   ...
            # We try all of these patterns; first hit wins.
            dim = largs[0]
            if idx == 0:
                # Initial / embedding-level LN.  Prefer HF embedding LN
                # over plain `LayerNorm.weight` so a checkpoint that
                # has both an embedding LN and per-block LNs disambiguates
                # correctly (an arch with idx=0 ALWAYS expects the
                # embedding-level LN to match first).
                candidates_w = [
                    "embeddings.LayerNorm.weight",
                    "LayerNorm.weight",
                    "layers.0.LayerNorm.weight",
                    "layer_norm0.weight",
                    "ln_0.weight",
                ]
            else:
                # Per-block LN.  Decode (block_idx, sa|output) from idx:
                # blocks contribute 2 LNs each after the embedding LN
                # at idx=0, so block_idx = (idx - 1) // 2 and the SA flag
                # is is_sa = idx is odd.  Tries distilbert and BERT
                # naming patterns FIRST so embedding-level keys can't
                # accidentally match a per-block idx>0 slot.
                block_idx = (idx - 1) // 2
                is_sa = (idx % 2 == 1)
                sa_or_out = "sa_layer_norm" if is_sa else "output_layer_norm"
                candidates_w = [
                    # distilbert: transformer.layer.{i}.{sa|output}_layer_norm
                    f"transformer.layer.{block_idx}.{sa_or_out}.weight",
                    f"layer.{block_idx}.{sa_or_out}.weight",
                    # BERT: encoder.layer.{i}.attention.output.LayerNorm /
                    #       encoder.layer.{i}.output.LayerNorm
                    f"encoder.layer.{block_idx}.attention.output.LayerNorm.weight"
                        if is_sa else
                        f"encoder.layer.{block_idx}.output.LayerNorm.weight",
                    # Plain layered fallbacks (the existing convention)
                    f"layers.{idx}.LayerNorm.weight",
                    f"layer_norm{idx}.weight",
                    f"ln_{idx}.weight",
                ]
            candidates_b = [c.replace(".weight", ".bias") for c in candidates_w]
            wkey = sd_lookup(sd, candidates_w)
            bkey = sd_lookup(sd, candidates_b)
            if wkey is None or bkey is None:
                raise SystemExit(
                    f"layernorm #{idx}: missing γ / β.  Tried: "
                    f"{candidates_w[0]} / {candidates_w[1]}.  "
                    f"Available: {sorted(sd.keys())}"
                )
            W = sd[wkey].detach().float().cpu().numpy()
            b = sd[bkey].detach().float().cpu().numpy()
            if W.shape != (dim,) or b.shape != (dim,):
                raise SystemExit(
                    f"layernorm #{idx}: shape mismatch.  expected "
                    f"({dim},) + ({dim},); got {tuple(W.shape)} + "
                    f"{tuple(b.shape)}."
                )
            collected.append(("layernorm", idx, "", W, b))
        elif kind == "rmsnorm":
            # ["rmsnorm", dim] — γ only, no β.  Llama state_dict layout:
            #   model.layers.{i}.input_layernorm.weight        (pre-attn)
            #   model.layers.{i}.post_attention_layernorm.weight (pre-FFN)
            #   model.norm.weight                              (final)
            # `idx` walks RMSNorm entries in arch order.  For a decoder
            # with K blocks the layout is:
            #   idx=0      block 0 input_layernorm
            #   idx=1      block 0 post_attention_layernorm
            #   idx=2      block 1 input_layernorm
            #   ...
            #   idx=2K     final model.norm
            dim = largs[0]
            block_idx = idx // 2
            is_input = (idx % 2 == 0)
            sa_or_post = "input_layernorm" if is_input else "post_attention_layernorm"
            candidates_w = [
                f"model.layers.{block_idx}.{sa_or_post}.weight",
                f"layers.{block_idx}.{sa_or_post}.weight",
                "model.norm.weight",   # final RMSNorm; matches when idx == 2*K
                "norm.weight",
                f"rmsnorm{idx}.weight",
                f"rn_{idx}.weight",
            ]
            wkey = sd_lookup(sd, candidates_w)
            if wkey is None:
                raise SystemExit(
                    f"rmsnorm #{idx}: missing γ.  Tried: "
                    f"{candidates_w[0]} / {candidates_w[1]} / "
                    f"{candidates_w[2]}.  Available: {sorted(sd.keys())}"
                )
            W = sd[wkey].detach().float().cpu().numpy()
            if W.shape != (dim,):
                raise SystemExit(
                    f"rmsnorm #{idx}: shape mismatch.  expected ({dim},); "
                    f"got {tuple(W.shape)}."
                )
            b = _np.zeros(0, dtype=_np.float32)
            collected.append(("rmsnorm", idx, "", W, b))
        elif kind == "swiglu_ffn":
            # ["swiglu_ffn", d_model, d_ffn] — three weight tensors,
            # NO biases (Llama convention).  HF state_dict layout:
            #   model.layers.{i}.mlp.gate_proj.weight   (d_ffn, d_model)
            #   model.layers.{i}.mlp.up_proj.weight     (d_ffn, d_model)
            #   model.layers.{i}.mlp.down_proj.weight   (d_model, d_ffn)
            d_model, d_ffn = largs
            shape_for = {
                "gate": (d_ffn, d_model),
                "up":   (d_ffn, d_model),
                "down": (d_model, d_ffn),
            }
            for proj in ("gate", "up", "down"):
                hf = f"{proj}_proj"
                candidates_w = [
                    f"model.layers.{idx}.mlp.{hf}.weight",
                    f"layers.{idx}.mlp.{hf}.weight",
                    # Plain fallback — useful for synthetic test rigs
                    # that don't follow the HF block-tree naming.
                    f"swiglu{idx}.{proj}.weight",
                ]
                wkey = sd_lookup(sd, candidates_w)
                if wkey is None:
                    raise SystemExit(
                        f"swiglu_ffn #{idx}: missing {proj}_proj weight.  "
                        f"Tried: {candidates_w[0]} / {candidates_w[1]} / "
                        f"{candidates_w[2]}.  "
                        f"Available: {sorted(sd.keys())}"
                    )
                W = sd[wkey].detach().float().cpu().numpy()
                if W.shape != shape_for[proj]:
                    raise SystemExit(
                        f"swiglu_ffn #{idx} {proj}: shape mismatch.  "
                        f"expected {shape_for[proj]}; got {tuple(W.shape)}."
                    )
                b = _np.zeros(0, dtype=_np.float32)
                collected.append(("swiglu_ffn", idx, proj, W, b))
        elif kind == "embedding":
            # ["embedding", vocab_size, d_model, seq] — only a weight
            # tensor (no bias).  HuggingFace canonical names cover BERT/
            # distilbert (embeddings.word_embeddings.weight) and GPT-style
            # (wte.weight); also accept bare ".weight" stems for one-off
            # checkpoints saved with state_dict[token_emb] = ...
            vocab_size, d_model, _seq = largs
            candidates = [
                "embeddings.word_embeddings.weight",
                "wte.weight",
                "embed.weight",
                "tok_embeddings.weight",
                "embeddings.weight",
                f"layers.{idx}.embed.weight",
            ]
            wkey = sd_lookup(sd, candidates)
            if wkey is None:
                raise SystemExit(
                    f"embedding #{idx}: cannot find weight in checkpoint.  "
                    f"Tried: {candidates[0]} / {candidates[1]} / ...  "
                    f"Available: {sorted(sd.keys())}"
                )
            W = sd[wkey].detach().float().cpu().numpy()
            if W.shape != (vocab_size, d_model):
                raise SystemExit(
                    f"embedding #{idx}: shape mismatch.  expected "
                    f"({vocab_size},{d_model}); got {tuple(W.shape)}."
                )
            # Synthesise a zero-byte placeholder for `b` so the shared
            # 5-tuple shape stays consistent.  Downstream consumers
            # already gate on size==0 to skip bias staging.
            b = _np.zeros(0, dtype=_np.float32)
            collected.append(("embedding", idx, "", W, b))
        elif kind == "positional_embedding":
            # ["positional_embedding", max_pos, d_model, seq] — only a
            # weight tensor (no bias).  HuggingFace canonical names cover
            # BERT/distilbert (embeddings.position_embeddings.weight)
            # and GPT-2 style (wpe.weight).
            max_pos, d_model, _seq = largs
            candidates = [
                "embeddings.position_embeddings.weight",
                "position_embeddings.weight",
                "wpe.weight",
                "pos_embed.weight",
                f"layers.{idx}.pos_embed.weight",
            ]
            wkey = sd_lookup(sd, candidates)
            if wkey is None:
                raise SystemExit(
                    f"positional_embedding #{idx}: cannot find weight "
                    f"in checkpoint.  Tried: {candidates[0]} / "
                    f"{candidates[1]} / ...  Available: {sorted(sd.keys())}"
                )
            W = sd[wkey].detach().float().cpu().numpy()
            if W.shape != (max_pos, d_model):
                raise SystemExit(
                    f"positional_embedding #{idx}: shape mismatch.  "
                    f"expected ({max_pos},{d_model}); got {tuple(W.shape)}."
                )
            b = _np.zeros(0, dtype=_np.float32)
            collected.append(("positional_embedding", idx, "", W, b))
    # Skip zero-size tensors when accumulating the f32 blob (no bias for
    # nn.Embedding); same condition gate the staging-file writer uses.
    weights_blob = [a for (_, _, _, W, b) in collected
                    for a in ((W,) if b.size == 0 else (W, b))]

    out_base = Path(args.out)
    src_path  = out_base.with_suffix(".ari")
    embed_dir = str(out_base.parent.resolve())

    scales = {}   # (kind, idx, suffix, "W"|"b") → f32 scale, populated
                  #   for int8.  Suffix is "" for single-tensor layers
                  #   (linear, conv2d_3x3_p1) and "gate"/"up"/"down" for
                  #   swiglu_ffn (which carries 3 weight tensors per layer
                  #   that need INDEPENDENT scales since their value
                  #   distributions differ — gate/up grow ~3-5× wider
                  #   than down on real Llama checkpoints).

    if args.quantize == "int8" and not args.embed:
        # int8 only makes sense as a single-binary deploy; the
        # alternative (separate sidecar) would require a different
        # runtime loader path we don't currently emit.
        args.embed = True

    if args.embed and args.quantize == "int8":
        # int8 staging: LINEAR, CONV, SwiGLU, and EMBEDDING get
        # quantised.  Other kinds (attention, multi_head_attention,
        # positional_embedding, layernorm) keep f32 staging because
        # their forward paths consume f32 weights (no int8 kernel
        # exists for them).
        # Embedding was added 2026-05-12 after measure_quant.py showed
        # that per-row int8 of the embedding row costs ~0.01% ppl
        # while the matrix occupies ~40% of the binary on small
        # vocab=50K models; it's the single biggest binary-shrink
        # opportunity per quality-cost dollar.
        QUANTISABLE = ("linear", "conv2d_3x3_p1", "swiglu_ffn", "embedding")
        # Matvec/embedding layers use PER-ROW scales (one f32 per
        # output row of W); conv2d_3x3_p1 stays per-tensor (its kernel,
        # arr_i8_conv2d_3x3_p1*, has no per-row variant yet).
        # Per-row dramatically improves quantisation fidelity on large
        # matrices — gate/up/down on TinyLlama (5632 × 2048) collapse
        # from "model diverges from f32 greedy after 0 tokens" to ~5/8
        # prefix match because each row no longer competes for a
        # shared dynamic range.  For embeddings each token row has its
        # own scale so high-magnitude rare tokens don't waste the
        # dynamic range of the more common ones.
        PER_ROW = ("linear", "swiglu_ffn", "embedding")
        import numpy as _np
        wrote = []
        for kind, idx, suffix, W, b in collected:
            wname, bname = tensor_names(kind, idx, suffix)
            # Special path: int4 embedding (per-row symmetric int4, two
            # nibbles packed into one byte).  Only kicks in when both
            # --quantize int8 and --int4-embedding are set; everything
            # else stays on the int8 path.
            if (args.int4_linear and kind == "linear"):
                out_f, in_f = W.shape    # (out_features, in_features)
                if in_f % 2 != 0:
                    raise SystemExit(
                        f"--int4-linear requires even in_features; "
                        f"linear #{idx} has in_f={in_f}")
                if W.size:
                    abs_max_row = _np.abs(W).max(axis=1)
                else:
                    abs_max_row = _np.zeros((0,), dtype=_np.float32)
                scales_arr = (abs_max_row / 7.0).astype(_np.float32)
                safe = _np.where(scales_arr > 0, scales_arr, 1.0)
                q4 = _np.round(W / safe[:, None]).clip(-8, 7).astype(_np.int8)
                q4_u = (q4 & 0x0F).astype(_np.uint8)
                lo = q4_u[:, 0::2]
                hi = q4_u[:, 1::2]
                packed = (lo | (hi << 4)).astype(_np.uint8)   # (out_f, in_f/2)
                scales[(kind, idx, suffix, "W")] = scales_arr
                wp = out_base.parent / f"{out_base.name}_{wname}.i4"
                packed.tofile(wp)
                wrote.append(wp)
                sp = out_base.parent / f"{out_base.name}_{wname}_scales.f32"
                scales_arr.tofile(sp)
                wrote.append(sp)
            elif (args.int4_embedding and kind == "embedding"):
                vocab, d_model = W.shape
                if d_model % 2 != 0:
                    raise SystemExit(
                        f"--int4-embedding requires even d_model; "
                        f"got {d_model}")
                if W.size:
                    abs_max_row = _np.abs(W).max(axis=1)
                else:
                    abs_max_row = _np.zeros((0,), dtype=_np.float32)
                # int4 symmetric: representable range [-7, 7], so divide
                # by max/7 (we leave -8 unused to keep symmetry — 0.5
                # bits of range loss is a non-issue here since per-row
                # scaling already concentrates dynamic range).
                scales_arr = (abs_max_row / 7.0).astype(_np.float32)
                safe = _np.where(scales_arr > 0, scales_arr, 1.0)
                q4 = _np.round(W / safe[:, None]).clip(-8, 7).astype(_np.int8)
                # Pack two int4 values per byte.  Low nibble = even
                # column, high nibble = odd column.  Convert signed
                # nibbles to unsigned (-8 → 8 … 7 → 7 in unsigned form)
                # via & 0x0F so byte_at gives us back the same unsigned
                # int we can then sign-extend in the .ari loop.
                q4_u = (q4 & 0x0F).astype(_np.uint8)        # (vocab, d_model)
                lo = q4_u[:, 0::2]
                hi = q4_u[:, 1::2]
                packed = (lo | (hi << 4)).astype(_np.uint8) # (vocab, d_model/2)
                scales[(kind, idx, suffix, "W")] = scales_arr
                wp = out_base.parent / f"{out_base.name}_{wname}.i4"
                packed.tofile(wp)
                wrote.append(wp)
                sp = out_base.parent / f"{out_base.name}_{wname}_scales.f32"
                scales_arr.tofile(sp)
                wrote.append(sp)
            elif kind in PER_ROW:
                # Per-row symmetric int8: scales[i] = max(|W[i,:]|) / 127.
                # Each output row gets its own scale, so fat rows don't
                # waste the dynamic range of skinny rows.
                if W.size:
                    abs_max_row = _np.abs(W).max(axis=1)            # (m,)
                else:
                    abs_max_row = _np.zeros((0,), dtype=_np.float32)
                scales_arr = abs_max_row / 127.0
                # Empty rows or all-zero rows would divide by zero;
                # substitute 1.0 (q values stay zero, so the scale is a
                # no-op).
                safe = _np.where(scales_arr > 0, scales_arr, 1.0)
                q = _np.round(W / safe[:, None]).clip(-128, 127).astype(_np.int8)
                scales_arr = scales_arr.astype(_np.float32)
                scales[(kind, idx, suffix, "W")] = scales_arr
                wp = out_base.parent / f"{out_base.name}_{wname}.i8"
                q.tofile(wp)
                wrote.append(wp)
                # Per-row scale sidecar.  gen_load embeds it via
                # embed_file (f32 path) alongside the .i8 weight.
                sp = out_base.parent / f"{out_base.name}_{wname}_scales.f32"
                scales_arr.tofile(sp)
                wrote.append(sp)
            elif kind in QUANTISABLE:
                # conv2d_3x3_p1 — per-tensor.  Same staging shape as
                # before per-row landed.
                abs_max = float(_np.abs(W).max()) if W.size else 0.0
                scale = abs_max / 127.0 if abs_max > 0 else 0.0
                if scale == 0.0:
                    q = _np.zeros(W.shape, dtype=_np.int8)
                else:
                    q = _np.round(W / scale).clip(-128, 127).astype(_np.int8)
                scales[(kind, idx, suffix, "W")] = scale
                wp = out_base.parent / f"{out_base.name}_{wname}.i8"
                q.tofile(wp)
                wrote.append(wp)
            else:
                # Stays f32 — gen_load embeds via embed_file (not
                # embed_file_bytes) and the runtime kernel reads
                # straight f32.
                wp = out_base.parent / f"{out_base.name}_{wname}.f32"
                W.tofile(wp)
                wrote.append(wp)
            # Bias (when present): always f32.  Tiny tensors so
            # quantisation buys ~10 bytes per layer at the cost of
            # a less-accurate constant offset.
            if b.size > 0:
                bp = out_base.parent / f"{out_base.name}_{bname}.f32"
                b.tofile(bp)
                wrote.append(bp)
        print("wrote per-tensor staging files:")
        total = 0
        for p in wrote:
            sz = p.stat().st_size
            total += sz
            print(f"  {p}  ({sz} bytes)")
        f32_total = sum(a.size * 4 for (_, _, _, W, b) in collected for a in (W, b))
        print(f"weights+biases:  {f32_total} bytes (all-f32) → {total} bytes "
              f"(int8 W on Linear/Conv only + f32 elsewhere) — "
              f"{f32_total / max(total, 1):.2f}× smaller")
    elif args.embed:
        wrote = []
        for kind, idx, suffix, W, b in collected:
            wname, bname = tensor_names(kind, idx, suffix)
            wp = out_base.parent / f"{out_base.name}_{wname}.f32"
            with open(wp, "wb") as f: W.tofile(f)
            wrote.append(wp)
            if b.size > 0:
                bp = out_base.parent / f"{out_base.name}_{bname}.f32"
                with open(bp, "wb") as f: b.tofile(f)
                wrote.append(bp)
        print("wrote per-tensor staging files:")
        for p in wrote:
            print(f"  {p}")
    else:
        blob_path = out_base.with_suffix(".f32")
        with open(blob_path, "wb") as f:
            for arr in weights_blob:
                arr.tofile(f)
        print(f"wrote {blob_path}")

    # The activation buffer a0 is sized in elements; that's also the
    # number of input bytes per MNIST sample (1 byte per pixel).
    _, sizes = gen_act_decls(arch)
    n_in_elements = sizes[0]
    prologue_parts = [
        PROLOGUE.format(model_name=out_base.name,
                        arch_repr=arch,
                        loader=args.input_format,
                        quantize=args.quantize),
    ]
    if needs_dequant_helper(arch, args.quantize, args.input_format):
        prologue_parts.append(DEQUANT_HELPER)
    if needs_attention_lib(arch):
        prologue_parts.append(ATTENTION_LIB)
    if needs_kv_attention_lib(arch):
        prologue_parts.append(_load_kv_attention_lib())
    if needs_layernorm_helper(arch):
        prologue_parts.append(LAYERNORM_HELPER)
    if needs_rmsnorm_helper(arch):
        prologue_parts.append(RMSNORM_HELPER)
    if needs_gelu_helper(arch):
        prologue_parts.append(GELU_HELPER)
    if needs_swiglu_helper(arch):
        prologue_parts.append(SWIGLU_HELPER)
    prologue_parts.append(PROLOGUE_TAIL)
    main_emitter = (gen_decoder_main if is_decoder_arch(arch)
                    else gen_main)
    src = (
        "".join(prologue_parts)
        + "\n"
        + MNIST_LOADER.format(n_in=n_in_elements,
                              mean=args.mean,
                              std=args.std)
        + "\n"
        + main_emitter(arch, args, out_base.name, embed_dir, scales)
        + "\n"
    )
    src_path.write_text(src)

    n_floats = sum(a.size for a in weights_blob)
    print(f"wrote {src_path}  ({n_floats} f32 of weights total, {n_floats * 4} bytes)")
    print(f"build:  aric {src_path.name} -o {out_base.name}")
    if args.embed:
        print(f"        (after build, the per-tensor .f32 staging files can be deleted)")
    print(f"run:    ./{out_base.name}")


if __name__ == "__main__":
    main()
