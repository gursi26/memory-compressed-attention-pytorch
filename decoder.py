from torch import nn
import torch
import math
import torch.nn.functional as F
from utils import PositionalEncoder, FeedForward
from attention import CompressedMultiHeadAttention

class DecoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads, compress_factor, dropout_p):
        super(DecoderLayer, self).__init__()
        self.dropout_p = dropout_p
        self.masked_compressed_mha = CompressedMultiHeadAttention(input_dim, output_dim, num_heads, compress_factor)
        self.feed_forward = FeedForward(output_dim)
        self.ln1 = nn.LayerNorm(output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

    def forward(self, x, mask, prev_kv=None):
        skip_x = x
        x, kv = self.masked_compressed_mha(x, mask=mask, prev_kv=prev_kv)
        x = self.ln1(F.dropout(x, self.dropout_p) + skip_x)
        skip_x = x
        x = self.feed_forward(x)
        x = self.ln2(F.dropout(x, self.dropout_p) + skip_x)
        return x, kv
    

class DecoderOnlyModel(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads, num_layers, compress_factor, dropout_p, max_seq_len=10000):
        super(DecoderOnlyModel, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoder(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_model, num_heads, compress_factor, dropout_p) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.out_proj.weight = self.embed.weight

    def forward(self, x, prev_kvs=None):
        """
        inputs:
            x: [batch_size, seq_len] shaped tensor of labels upto vocab_size
            prev_kvs: list of length len(self.layers), Each element is of shape [batch_size, context_len, d_model]

        outputs:
            output1: [batch_size, seq_len, vocab_size]
                logits vector
            output2: [num_layers, 2, batch_size, seq_len, d_model]
                concatenated, stored k and v embeddings for all layers
        """
        x = self.pos_enc(self.embed(x) * math.sqrt(self.d_model))

        if prev_kvs is None:
            prev_kvs = [None] * len(self.layers)
        assert len(prev_kvs) == len(self.layers)
        
        kv_out = []
        for layer, prev_kv in zip(self.layers, prev_kvs):
            x, kv = layer(x, mask=True, prev_kv=prev_kv)
            kv_out.append(kv.unsqueeze(0))

        return self.out_proj(x), torch.cat(kv_out, dim=0)  