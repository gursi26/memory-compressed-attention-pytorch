from torch import nn
import torch
import math
import torch.nn.functional as F
from utils import PositionalEncoder, FeedForward
from attention import CompressedMultiHeadAttention

class MemoryCompressedDecoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads, compress_factor, dropout_p):
        super(MemoryCompressedDecoderLayer, self).__init__()
        self.dropout_p = dropout_p
        self.masked_compressed_mha = CompressedMultiHeadAttention(input_dim, output_dim, num_heads, compress_factor)
        self.feed_forward = FeedForward(output_dim)
        self.ln1 = nn.LayerNorm(output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

    def forward(self, x, mask):
        skip_x = x
        x = self.masked_compressed_mha(x, mask=mask)
        x = self.ln1(F.dropout(x, self.dropout_p) + skip_x)
        skip_x = x
        x = self.feed_forward(x)
        x = self.ln2(F.dropout(x, self.dropout_p) + skip_x)
        return x
    

class LocalAttentionDecoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads, block_size, dropout_p):
        super(LocalAttentionDecoderLayer, self).__init__()
        self.dropout_p = dropout_p
        self.local_attention_mha = CompressedMultiHeadAttention(input_dim, output_dim, num_heads, block_size)
        self.feed_forward = FeedForward(output_dim)
        self.ln1 = nn.LayerNorm(output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

    def forward(self, x, mask):
        skip_x = x
        x = self.local_attention_mha(x, mask=mask)
        x = self.ln1(F.dropout(x, self.dropout_p) + skip_x)
        skip_x = x
        x = self.feed_forward(x)
        x = self.ln2(F.dropout(x, self.dropout_p) + skip_x)
        return x
    

class DecoderOnlyModel(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads, num_layers, compress_factor, block_size, dropout_p, max_seq_len=10000):
        super(DecoderOnlyModel, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoder(d_model, max_seq_len)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                self.layers.append(LocalAttentionDecoderLayer(d_model, d_model, num_heads, block_size, dropout_p))
            else:
                self.layers.append(MemoryCompressedDecoderLayer(d_model, d_model, num_heads, compress_factor, dropout_p))

        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.out_proj.weight = self.embed.weight

    def forward(self, x):
        """
        inputs:
            x: [batch_size, seq_len] shaped tensor of labels upto vocab_size

        outputs:
            output1: [batch_size, seq_len, vocab_size]
                logits vector
        """
        x = self.pos_enc(self.embed(x) * math.sqrt(self.d_model))
        
        for layer in self.layers:
            x = layer(x, mask=True)

        return self.out_proj(x) 