"""Synthetic embedding with a real-HF-class vocab (30522 = distilbert).

vocab=30522 forces the 2-byte-per-token loader path.  Token IDs
include both small values (< 256) and large ones (> 65000) to make
sure the little-endian decode is bit-correct across the full range.

torch.manual_seed(31).
"""

import numpy as np
import torch

torch.manual_seed(31)

vocab_size, d_model, seq = 30522, 32, 6

emb_table = torch.randn(vocab_size, d_model)

# Mix of low (< 256), mid (256..30521), and edge token IDs across
# the full vocab range.  Stays within vocab_size to be a valid
# index, but spans both byte ranges so the LE decode is exercised.
token_ids = [101, 7592, 30521, 0, 256, 1024]
expected = emb_table[token_ids].view(-1)   # [seq * d_model]

sd = {"embeddings.word_embeddings.weight": emb_table}
torch.save(sd, "synth.pt")

# Token IDs as little-endian uint16 — 2 bytes per token.
np.array(token_ids, dtype=np.uint16).tofile("tokens.bin")
expected.detach().numpy().tofile("expected.f32")

print(f"vocab_size={vocab_size} d_model={d_model} seq={seq}")
print(f"tokens={token_ids}")
print(f"output[:4] = {expected[:4].tolist()}")
