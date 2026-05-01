#!/usr/bin/env python3
"""
aricode-pack v0.1 — model → static binary compiler.

Given a PyTorch state_dict and an explicit architecture spec, emit:
    <out>.ari   aricode source for the forward pass
    <out>.f32   raw little-endian f32 weights blob (in arch order)

The `aric` compiler then turns <out>.ari into a static x86_64+AVX2
binary that loads <out>.f32 at startup and serves inference.  No
PyTorch, no Python, no CUDA — a few hundred KB total.

This v0.1 supports a deliberately small layer vocabulary: Linear,
ReLU, Sigmoid, Tanh, Softmax (output).  CNN + attention land in
v0.2 once the layer template family stabilises.

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
    W = sd[w_key].detach().cpu().numpy().astype("float32")
    b = sd[b_key].detach().cpu().numpy().astype("float32")
    if W.shape != (out_f, in_f) or b.shape != (out_f,):
        raise ValueError(
            f"linear #{idx}: shape mismatch.  expected W=({out_f},{in_f}) "
            f"b=({out_f},), got W={tuple(W.shape)} b={tuple(b.shape)}."
        )
    return [W, b]


def emit_linear(idx, in_f, out_f, src_var, dst_var, w_var, b_var,
                quant_scale=None):
    """Emit a Linear-layer forward.

    `quant_scale` is None for the f32 path (default, single arr_f32_matvec
    call) or a Python float when the weights are int8.  In the int8 case
    we use the native arr_i8_matvec_f32 builtin, then arr_f32_add_scaled
    to fuse in the f32 bias — two calls vs one on the f32 path, but the
    weights stay int8 in RAM so RAM usage on a 200 KB FC layer drops to
    50 KB and we skip the startup dequant pass entirely."""
    if quant_scale is None:
        return [
            f"    arr_f32_matvec({w_var}, {src_var}, {b_var}, {dst_var}, {out_f}, {in_f});",
        ]
    return [
        f"    arr_i8_matvec_f32({w_var}, {src_var}, {dst_var}, {out_f}, {quant_scale!r});",
        f"    arr_f32_add_scaled({dst_var}, {b_var}, 1.0);",
    ]


def emit_relu(var):
    return [f"    arr_f32_relu({var});"]


def emit_sigmoid(var):
    return [f"    arr_f32_sigmoid({var});"]   # f64 builtin; user must promote


def emit_tanh(var):
    return [f"    arr_f32_tanh({var});"]


def emit_softmax(var):
    return [f"    arr_f32_softmax({var});"]


# ──────────────────────────────────────────────────────────────────────
#  Code generator: walks the arch list and emits a self-contained
#  aricode source that loads the weight blob, runs the test loop, and
#  reports accuracy.
# ──────────────────────────────────────────────────────────────────────


PROLOGUE = """\
// Generated by aricode-pack v0.1 — DO NOT EDIT.
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
    raise ValueError(f"tensor_names: no weights for {kind!r}")


def weight_tensors(arch):
    """Yield (kind, idx, suffix, nw, nb, wname, bname) per (W, b) tensor
    pair across the whole arch, in source order.  This is the iteration
    shape every weight-emission site should use — handles the multi-
    tensor case transparently (attention yields three entries per layer
    with q/k/v suffixes) and exposes the canonical names without each
    callsite having to reinvent them."""
    li = 0
    ci = 0
    for kind, *args in arch:
        if kind == "linear":
            for suffix, nw, nb in tensor_specs(kind, *args):
                wname, bname = tensor_names(kind, li, suffix)
                yield (kind, li, suffix, nw, nb, wname, bname)
            li += 1
        elif kind == "conv2d_3x3_p1":
            for suffix, nw, nb in tensor_specs(kind, *args):
                wname, bname = tensor_names(kind, ci, suffix)
                yield (kind, ci, suffix, nw, nb, wname, bname)
            ci += 1


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
    nothing (embed_file inside gen_load declares the let bindings)."""
    if embed:
        return []
    lines = []
    for kind, idx, suffix, nw, nb, wname, bname in weight_tensors(arch):
        lines.append(f"    let {wname}: i32 = arr_f32_new({nw});")
        lines.append(f"    let {bname}: i32 = arr_f32_new({nb});")
    return lines


def gen_act_decls(arch):
    """One activation buffer per shape-changing step.  Tracks size in
    elements; spatial vs flat is implicit in how the next layer reads
    it.  Returns (decl_lines, sizes_list)."""
    if not arch:
        raise ValueError("empty arch")
    first = arch[0]
    if first[0] == "linear":
        sizes = [first[1]]
    elif first[0] == "conv2d_3x3_p1":
        c_in, _ = first[1], first[2]
        sizes = [c_in * 28 * 28]
    else:
        raise ValueError(f"first layer must be linear or conv2d_3x3_p1, got {first[0]!r}")

    for kind, *args in arch:
        if kind == "linear":
            sizes.append(args[1])
        elif kind == "conv2d_3x3_p1":
            c_in, c_out = args
            sizes.append(c_out * 28 * 28)
        elif kind == "maxpool_2x2":
            c = args[0]
            sizes.append(c * 14 * 14)
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
    pi = 0
    for kind, *args in arch:
        if kind == "maxpool_2x2":
            c = args[0]
            lines.append(f"    let pool_a{pi}: i32 = arr_f32_new({c * 14 * 14});")
            pi += 1
    return lines


def names_for(kind, idx):
    """Legacy single-tensor accessor.  Equivalent to tensor_names(kind, idx)
    with the default empty suffix; kept for callers that don't yet use
    weight_tensors()."""
    return tensor_names(kind, idx)


def gen_load(arch, embed_dir, out_name, embed, quantize, scales,
             multi_int8_runtime=False):
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
        lines = []
        for kind, idx, suffix, nw, nb, wname, bname in weight_tensors(arch):
            scale_w = scales[(kind, idx, "W")]
            c_in = arch_by_idx.get((kind, idx))
            if kind == "linear":
                lines.append(f"    let {wname}: i32 = embed_file_bytes(\"{embed_dir}/{out_name}_{wname}.i8\");")
                lines.append(f"    let {bname}: i32 = embed_file(\"{embed_dir}/{out_name}_{bname}.f32\");")
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
            lines.append(f"    let {bname}: i32 = embed_file(\"{embed_dir}/{out_name}_{bname}.f32\");")
        return lines
    lines = [f"    let weight_path: i32 = str_new(\"{out_name}.f32\");",
             "    let _wfd: i32 = file_open(weight_path, 0);"]
    for kind, idx, suffix, nw, nb, wname, bname in weight_tensors(arch):
        lines.append(f"    file_read(_wfd, {wname}, {nw * 4});")
        lines.append(f"    file_read(_wfd, {bname}, {nb * 4});")
    lines.append("    file_close(_wfd);")
    return lines


def gen_forward(arch, scales=None, multi_int8_runtime=False):
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
    li = 0
    ci = 0
    pi = 0
    for kind, *args in arch:
        if kind == "linear":
            in_f, out_f = args
            src = f"a{cur_a}"
            dst = f"a{cur_a + 1}"
            sc = scales.get(("linear", li, "W")) if scales else None
            lines += emit_linear(li, in_f, out_f, src, dst,
                                 f"W{li}", f"b{li}", quant_scale=sc)
            cur_a += 1
            li += 1
        elif kind == "conv2d_3x3_p1":
            c_in, c_out = args
            src = f"a{cur_a}"
            dst = f"a{cur_a + 1}"
            conv_scale = scales.get(("conv2d_3x3_p1", ci, "W")) if scales else None
            if c_in == 1:
                # Fast path: single-channel input goes straight into the
                # AVX2 builtin.  Pad once, call once.  The int8 variant
                # carries the per-tensor scale and reads weights as i8;
                # otherwise fall back to the f32 conv builtin.
                lines.append(f"    arr_f32_fill(padded, 0.0);")
                lines.append(f"    let _y{ci}: i32 = 0;")
                lines.append(f"    while (_y{ci} < 28) {{")
                lines.append(f"        arr_f32_copy_slice({src}, _y{ci} * 28, padded, (_y{ci} + 1) * 30 + 1, 28);")
                lines.append(f"        _y{ci} = _y{ci} + 1;")
                lines.append(f"    }}")
                if conv_scale is not None:
                    lines.append(f"    arr_i8_conv2d_3x3_p1(padded, Wc{ci}, bc{ci}, {dst}, {c_out}, {conv_scale!r});")
                else:
                    lines.append(f"    arr_f32_conv2d_3x3_p1(padded, Wc{ci}, bc{ci}, {dst}, {c_out});")
            else:
                # Multi-channel input: pad each input channel into the
                # shared padded_multi [C_in, 30, 30] buffer and call the
                # native AVX2 multi-channel kernel once.  No more user-fn
                # accumulation loop or _partial/_w_slice/_b_zero scratch
                # — the kernel does the load-modify-store accumulation
                # internally and broadcasts bias once per c_out.
                lines.append(f"    arr_f32_fill(padded_multi, 0.0);")
                lines.append(f"    let _cin{ci}: i32 = 0;")
                lines.append(f"    while (_cin{ci} < {c_in}) {{")
                lines.append(f"        let _yc{ci}: i32 = 0;")
                lines.append(f"        while (_yc{ci} < 28) {{")
                lines.append(f"            arr_f32_copy_slice({src}, _cin{ci} * 784 + _yc{ci} * 28, padded_multi, _cin{ci} * 900 + (_yc{ci} + 1) * 30 + 1, 28);")
                lines.append(f"            _yc{ci} = _yc{ci} + 1;")
                lines.append(f"        }}")
                lines.append(f"        _cin{ci} = _cin{ci} + 1;")
                lines.append(f"    }}")
                if conv_scale is not None and multi_int8_runtime:
                    # int8 weights stay int8 in RAM — no startup dequant.
                    lines.append(f"    arr_i8_conv2d_3x3_p1_multi(padded_multi, Wc{ci}, bc{ci}, {dst}, {c_out}, {conv_scale!r});")
                else:
                    # f32 multi-channel kernel handles both the
                    # plain-f32 case and the int8-dequantised-at-startup
                    # case (Wc{ci} is f32 in RAM in both).
                    lines.append(f"    arr_f32_conv2d_3x3_p1_multi(padded_multi, {c_in}, Wc{ci}, bc{ci}, {dst}, {c_out});")
            cur_a += 1
            ci += 1
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
        else:
            raise ValueError(f"unknown layer kind: {kind!r}")
    return lines, cur_a


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
    # this is the first layer's in_features; for a CNN it's c_in*28*28.
    n_in  = sizes[0]
    n_out = next(i for i in reversed(arch) if i[0] == "linear")[2]

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
    p = argparse.ArgumentParser(description="aricode-pack v0.1")
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
    p.add_argument("--input-format", choices=("mnist", "stdin"), default="mnist")
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

    arch = []
    last_spatial = None   # (C_out, 28, 28) after a conv layer; None otherwise
    for i, stem in enumerate(stems):
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
        if i < len(stems) - 1:
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
    # writer can name them consistently.
    collected = []   # list of (kind, idx, W_array, b_array)
    li = 0
    ci = 0
    for kind, *largs in arch:
        if kind == "linear":
            in_f, out_f = largs
            wkey = keys(li, "weight")
            bkey = keys(li, "bias")
            if wkey not in sd or bkey not in sd:
                raise SystemExit(
                    f"linear #{li}: missing '{wkey}' or '{bkey}' in checkpoint. "
                    f"Available keys: {sorted(sd.keys())}"
                )
            W = sd[wkey].detach().cpu().numpy().astype("float32")
            b = sd[bkey].detach().cpu().numpy().astype("float32")
            if W.shape != (out_f, in_f) or b.shape != (out_f,):
                raise SystemExit(
                    f"linear #{li}: shape mismatch.  expected ({out_f},{in_f}) "
                    f"+ ({out_f},); got {tuple(W.shape)} + {tuple(b.shape)}."
                )
            collected.append(("linear", li, "", W, b))
            li += 1
        elif kind == "conv2d_3x3_p1":
            c_in, c_out = largs
            # Conv keys default to {idx_plus_1}.weight; if the user kept
            # PyTorch's natural `conv.weight` naming, --keys can be
            # remapped per call.  By default the same template is shared
            # — works for the demo where the conv's index_plus_1 = 1
            # (i.e. "1.weight") doesn't clash with fc names.  Real users
            # can pass --conv-keys later; v0.4 supports the simple case.
            wkey = sd_lookup(sd, ["conv.weight",
                                  f"conv{ci + 1}.weight",
                                  f"layers.{ci}.weight",
                                  keys(ci, "weight")])
            bkey = sd_lookup(sd, ["conv.bias",
                                  f"conv{ci + 1}.bias",
                                  f"layers.{ci}.bias",
                                  keys(ci, "bias")])
            if wkey is None or bkey is None:
                raise SystemExit(
                    f"conv2d_3x3_p1 #{ci}: cannot find weight/bias in "
                    f"checkpoint.  Tried: conv.weight / layers.{ci}.weight / "
                    f"{keys(ci, 'weight')}.  Available: {sorted(sd.keys())}"
                )
            W = sd[wkey].detach().cpu().numpy().astype("float32")
            b = sd[bkey].detach().cpu().numpy().astype("float32")
            if W.shape != (c_out, c_in, 3, 3) or b.shape != (c_out,):
                raise SystemExit(
                    f"conv2d_3x3_p1 #{ci}: shape mismatch.  expected "
                    f"({c_out},{c_in},3,3) + ({c_out},); got "
                    f"{tuple(W.shape)} + {tuple(b.shape)}."
                )
            # Flatten (C_out, C_in, 3, 3) → (C_out, C_in*9) row-major to
            # match arr_f32_conv2d_3x3_p1's expected weight layout.
            W = W.reshape(c_out, c_in * 9)
            collected.append(("conv2d_3x3_p1", ci, "", W, b))
            ci += 1
    weights_blob = [a for (_, _, _, W, b) in collected for a in (W, b)]

    out_base = Path(args.out)
    src_path  = out_base.with_suffix(".ari")
    embed_dir = str(out_base.parent.resolve())

    scales = {}   # (kind, idx, "W"|"b") → f32 scale, populated for int8

    if args.quantize == "int8" and not args.embed:
        # int8 only makes sense as a single-binary deploy; the
        # alternative (separate sidecar) would require a different
        # runtime loader path we don't currently emit.
        args.embed = True

    if args.embed and args.quantize == "int8":
        # Quantise weights to int8; keep biases as f32 (always tiny —
        # quantising them buys ~10 bytes per layer at the cost of a
        # less-accurate constant offset).
        import numpy as _np
        wrote = []
        for kind, idx, suffix, W, b in collected:
            wname, bname = tensor_names(kind, idx, suffix)
            # Weight: int8.
            abs_max = float(_np.abs(W).max()) if W.size else 0.0
            scale = abs_max / 127.0 if abs_max > 0 else 0.0
            if scale == 0.0:
                q = _np.zeros(W.shape, dtype=_np.int8)
            else:
                q = _np.round(W / scale).clip(-128, 127).astype(_np.int8)
            scales[(kind, idx, "W")] = scale
            wp = out_base.parent / f"{out_base.name}_{wname}.i8"
            q.tofile(wp)
            wrote.append(wp)
            # Bias: f32.
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
              f"(int8 W + f32 b) — {f32_total / max(total, 1):.2f}× smaller")
    elif args.embed:
        wrote = []
        for kind, idx, suffix, W, b in collected:
            wname, bname = tensor_names(kind, idx, suffix)
            wp = out_base.parent / f"{out_base.name}_{wname}.f32"
            bp = out_base.parent / f"{out_base.name}_{bname}.f32"
            with open(wp, "wb") as f: W.tofile(f)
            with open(bp, "wb") as f: b.tofile(f)
            wrote.extend([wp, bp])
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
    prologue_parts.append(PROLOGUE_TAIL)
    src = (
        "".join(prologue_parts)
        + "\n"
        + MNIST_LOADER.format(n_in=n_in_elements,
                              mean=args.mean,
                              std=args.std)
        + "\n"
        + gen_main(arch, args, out_base.name, embed_dir, scales)
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
