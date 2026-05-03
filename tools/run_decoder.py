#!/usr/bin/env python3
"""run_decoder — text-in / text-out wrapper for an aricode-pack decoder
binary built with --input-format stdin-tokens.

Usage:
    python tools/run_decoder.py <binary> <hf_model_id> "<prompt>"

The wrapper:
  1. Loads the HF tokenizer for the given model id.
  2. Tokenizes the text prompt (BOS auto-added by transformers if the
     tokenizer config requests it — Llama / GPT-2 differ).
  3. Sends `struct.pack("<I", N) + tokens_as_uint{1,2,4}` to the binary's
     stdin.  Token byte width is auto-derived from vocab size:
       vocab ≤ 256       → uint8     (MNIST / very small classifiers)
       vocab ≤ 65536     → uint16 LE (GPT-2 50257, Llama 32000, BERT 30522)
       vocab > 65536     → uint32 LE (Llama-3 128k, etc.)
  4. Streams stdout: each generated token appears as a decimal integer
     on its own line.  As soon as the binary prints a token, the wrapper
     decodes it (incremental) and prints the resulting text fragment.
  5. Exits when the binary exits (which happens when --max-new-tokens
     have been emitted OR the binary's --eos-token was sampled).

Example:
    python tools/run_decoder.py \
        examples/llama2_7b/llama2_7b_chat \
        NousResearch/Llama-2-7b-hf \
        "The capital of France is"
"""

import os
import struct
import subprocess
import sys

from transformers import AutoTokenizer


def _token_byte_width(vocab_size: int) -> int:
    if vocab_size <= 256:
        return 1
    if vocab_size <= 65536:
        return 2
    return 4


def _pack_tokens(tokens: list[int], tb: int) -> bytes:
    """Pack a token-id list into the wire format the binary reads:
    4-byte uint32 LE length prefix, then N · tb-byte little-endian
    token IDs."""
    if tb == 1:
        body = bytes(tokens)
    elif tb == 2:
        body = b"".join(struct.pack("<H", t) for t in tokens)
    else:
        body = b"".join(struct.pack("<I", t) for t in tokens)
    return struct.pack("<I", len(tokens)) + body


def main():
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <binary> <hf_model_id> \"<prompt>\"",
              file=sys.stderr)
        sys.exit(2)

    binary, model_id, prompt = sys.argv[1], sys.argv[2], sys.argv[3]

    print(f"  loading tokenizer for {model_id}...", file=sys.stderr,
          flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    vocab = tok.vocab_size
    tb = _token_byte_width(vocab)

    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    print(f"  prompt: {prompt!r}", file=sys.stderr, flush=True)
    print(f"  → {len(prompt_ids)} tokens (tb={tb}): {prompt_ids[:8]}"
          f"{'...' if len(prompt_ids) > 8 else ''}", file=sys.stderr,
          flush=True)

    payload = _pack_tokens(prompt_ids, tb)
    print(f"  payload: {len(payload)} bytes", file=sys.stderr, flush=True)
    print(f"  launching {binary}...", file=sys.stderr, flush=True)
    print(f"---", file=sys.stderr, flush=True)
    print(f"{prompt}", end="", flush=True)

    # The binary opens its weight sidecar (e.g. llama2_7b_chat.f32) via
    # a RELATIVE path — pack.py's gen_load emits `str_new("…f32")`, and
    # file_open resolves that against cwd.  Run the binary with cwd =
    # binary's directory so the sidecar resolves correctly even when
    # the wrapper is invoked from elsewhere.  Otherwise file_open
    # silently fails, the KV cache stays uninitialised, and the
    # binary emits all-zero tokens.
    bin_abs = os.path.abspath(binary)
    bin_dir = os.path.dirname(bin_abs) or "."
    bin_name = "./" + os.path.basename(bin_abs)

    proc = subprocess.Popen(
        [bin_name],
        cwd=bin_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )

    # Send prompt all at once.  Close stdin so the binary doesn't block
    # on additional file_read calls (it shouldn't make any after the
    # prompt is read, but closing makes it explicit).
    proc.stdin.write(payload)
    proc.stdin.close()

    # Stream tokens as they appear.  The binary prints `print_int(tok)`
    # which writes "<int>\n" to stdout.  Decode each as soon as it
    # arrives — token-streaming UX.
    generated = []
    pending_buf = b""
    while True:
        chunk = proc.stdout.read(64)
        if not chunk:
            break
        pending_buf += chunk
        while b"\n" in pending_buf:
            line, _, pending_buf = pending_buf.partition(b"\n")
            line = line.decode("ascii", errors="ignore").strip()
            if not line:
                continue
            try:
                tok_id = int(line)
            except ValueError:
                continue
            generated.append(tok_id)
            # Decode incrementally — re-decode the full generated list
            # so multi-byte UTF-8 sequences from Llama's BPE land
            # correctly across token boundaries.
            full = tok.decode(generated, skip_special_tokens=False)
            # Diff vs what we've printed so far.  Track the printed
            # length and append the new suffix.
            full_text = full
            if not hasattr(main, "_printed_len"):
                main._printed_len = 0
            new_text = full_text[main._printed_len:]
            print(new_text, end="", flush=True)
            main._printed_len = len(full_text)

    proc.wait()
    print(f"\n", flush=True)
    print(f"---  generated {len(generated)} token(s) "
          f"({proc.returncode=})", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
