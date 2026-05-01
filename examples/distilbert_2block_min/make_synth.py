"""1-block synthetic distilbert (Post-LN)."""
import math, numpy as np, torch
torch.manual_seed(43)
vocab, max_pos, d_model, seq, d_ff, n_classes = 200, 32, 32, 4, 128, 8
n_heads = 4; d_head = d_model // n_heads
tok_emb = torch.randn(vocab, d_model); pos_emb = torch.randn(max_pos, d_model)
ln_emb_g = torch.randn(d_model); ln_emb_b = torch.randn(d_model)
W_Q = torch.randn(d_model, d_model); b_Q = torch.randn(d_model)
W_K = torch.randn(d_model, d_model); b_K = torch.randn(d_model)
W_V = torch.randn(d_model, d_model); b_V = torch.randn(d_model)
W_O = torch.randn(d_model, d_model); b_O = torch.randn(d_model)
ln_sa_g = torch.randn(d_model); ln_sa_b = torch.randn(d_model)
W_ff1 = torch.randn(d_ff, d_model); b_ff1 = torch.randn(d_ff)
W_ff2 = torch.randn(d_model, d_ff); b_ff2 = torch.randn(d_model)
ln_out_g = torch.randn(d_model); ln_out_b = torch.randn(d_model)
W_cls = torch.randn(n_classes, d_model); b_cls = torch.randn(n_classes)
gelu = torch.nn.GELU(approximate='tanh')
def lnorm(x, g, b, eps=1e-5):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight=g, bias=b, eps=eps)
token_ids = [11, 73, 142, 199]
x = tok_emb[token_ids] + pos_emb[:seq]
x = lnorm(x, ln_emb_g, ln_emb_b)
saved = x.clone()
Q = x @ W_Q.t() + b_Q; K = x @ W_K.t() + b_K; V = x @ W_V.t() + b_V
Q_h = Q.view(seq, n_heads, d_head); K_h = K.view(seq, n_heads, d_head); V_h = V.view(seq, n_heads, d_head)
heads = []
for h in range(n_heads):
    scores = Q_h[:, h] @ K_h[:, h].t() / math.sqrt(d_head)
    soft = torch.softmax(scores, dim=-1)
    heads.append(soft @ V_h[:, h])
x = torch.cat(heads, dim=-1) @ W_O.t() + b_O
x = x + saved
x = lnorm(x, ln_sa_g, ln_sa_b)
saved = x.clone()
x = x @ W_ff1.t() + b_ff1; x = gelu(x); x = x @ W_ff2.t() + b_ff2
x = x + saved
x = lnorm(x, ln_out_g, ln_out_b)
out = x @ W_cls.t() + b_cls
out_flat = out.view(-1).detach().numpy()
sd = {
    'embeddings.word_embeddings.weight': tok_emb,
    'embeddings.position_embeddings.weight': pos_emb,
    'embeddings.LayerNorm.weight': ln_emb_g, 'embeddings.LayerNorm.bias': ln_emb_b,
    'transformer.layer.0.attention.q_lin.weight': W_Q, 'transformer.layer.0.attention.q_lin.bias': b_Q,
    'transformer.layer.0.attention.k_lin.weight': W_K, 'transformer.layer.0.attention.k_lin.bias': b_K,
    'transformer.layer.0.attention.v_lin.weight': W_V, 'transformer.layer.0.attention.v_lin.bias': b_V,
    'transformer.layer.0.attention.out_lin.weight': W_O, 'transformer.layer.0.attention.out_lin.bias': b_O,
    'transformer.layer.0.sa_layer_norm.weight': ln_sa_g, 'transformer.layer.0.sa_layer_norm.bias': ln_sa_b,
    'transformer.layer.0.output_layer_norm.weight': ln_out_g, 'transformer.layer.0.output_layer_norm.bias': ln_out_b,
    'fc1.weight': W_ff1, 'fc1.bias': b_ff1,
    'fc2.weight': W_ff2, 'fc2.bias': b_ff2,
    'fc3.weight': W_cls, 'fc3.bias': b_cls,
}
torch.save(sd, 'synth.pt')
np.array(token_ids, dtype=np.uint8).tofile('tokens.bin')
out_flat.tofile('expected.f32')
