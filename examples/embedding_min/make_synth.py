"""Synthetic Embedding-only checkpoint.

vocab=10, d_model=16, seq=4.  Token IDs [3, 1, 4, 2] should produce
the corresponding 4 rows of the embedding matrix concatenated.

torch.manual_seed(29).
"""

import numpy as np
import torch

torch.manual_seed(29)

vocab_size, d_model, seq = 10, 16, 4

emb_table = torch.randn(vocab_size, d_model)

token_ids = [3, 1, 4, 2]
expected = emb_table[token_ids].view(-1)   # [seq * d_model]

sd = {"embeddings.word_embeddings.weight": emb_table}
torch.save(sd, "synth.pt")

# Token IDs as raw bytes (1 byte per token; vocab < 256).
np.array(token_ids, dtype=np.uint8).tofile("tokens.bin")
expected.detach().numpy().tofile("expected.f32")

print(f"vocab_size={vocab_size} d_model={d_model} seq={seq}")
print(f"tokens={token_ids}")
print(f"output[:4] = {expected[:4].tolist()}")
