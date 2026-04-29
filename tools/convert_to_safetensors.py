#!/usr/bin/env python3
"""
Convert a PyTorch state_dict .pt/.pth into the HuggingFace
.safetensors format aricode-pack reads natively.

Use case: you have a model trained with PyTorch and want to drop it
into the HF deploy ecosystem (or just share a safer-to-load file —
.pt is pickle-based and can execute arbitrary code on load).
aricode-pack accepts either format directly via --checkpoint, but
.safetensors is the de-facto standard on the Hub and worth
converting toward.

Usage:
    python convert_to_safetensors.py model.pt model.safetensors
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("error: PyTorch is required for the load step.", file=sys.stderr)
    sys.exit(2)

try:
    from safetensors.torch import save_file
except ImportError:
    print("error: install `safetensors` (pip install safetensors).",
          file=sys.stderr)
    sys.exit(2)


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    src = Path(argv[1])
    dst = Path(argv[2])

    sd = torch.load(src, map_location="cpu", weights_only=True)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    # safetensors requires contiguous f32 tensors; cast where needed.
    sd = {k: v.detach().cpu().contiguous() for k, v in sd.items()}
    save_file(sd, dst)

    src_sz = src.stat().st_size
    dst_sz = dst.stat().st_size
    print(f"wrote {dst}")
    print(f"  source ({src.suffix}): {src_sz} bytes")
    print(f"  output (.safetensors): {dst_sz} bytes")
    print("\ntensor inventory:")
    for k, v in sd.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main(sys.argv)
