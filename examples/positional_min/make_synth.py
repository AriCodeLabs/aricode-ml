"""Synthetic Embedding + PositionalEmbedding regression.

The standard HuggingFace BERT-class input pipeline:
    token_emb[token_ids] + position_emb[0..seq]

Validates that the pos-embed table is loaded correctly and added
in-place to the token embedding output.

torch.manual_seed(37).
"""

import numpy as np
import torch

torch.manual_seed(37)

vocab_size, max_pos, d_model, seq = 100, 32, 16, 4

tok_emb = torch.randn(vocab_size, d_model)
pos_emb = torch.randn(max_pos, d_model)

token_ids = [7, 13, 42, 99]
expected = (tok_emb[token_ids] + pos_emb[:seq]).view(-1)

sd = {
    "embeddings.word_embeddings.weight":     tok_emb,
    "embeddings.position_embeddings.weight": pos_emb,
}
torch.save(sd, "synth.pt")

# Token IDs as 1-byte (vocab=100 ≤ 256).
np.array(token_ids, dtype=np.uint8).tofile("tokens.bin")
expected.detach().numpy().tofile("expected.f32")

print(f"vocab_size={vocab_size} max_pos={max_pos} d_model={d_model} seq={seq}")
print(f"tokens={token_ids}")
print(f"output[:4] = {expected[:4].tolist()}")
