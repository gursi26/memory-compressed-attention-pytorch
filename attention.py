from torch import nn
import torch
import math

class QKVLayer(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads, bias=True):
        super(QKVLayer, self).__init__()
        assert output_dim % num_heads == 0
        self.num_heads = num_heads
        self.qkv_layer = nn.Linear(input_dim, output_dim * 3, bias=bias)

    def reshape(self, x):
        return x.view(x.shape[0], x.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)

    def forward(self, x):
        """
        x1: [batch_size, seq1_len, embed_dim]
        x2: [batch_size, seq2_len, embed_dim]
        (x2 is None if self attention)
        """
        q, k, v = self.qkv_layer(x).chunk(3, dim=-1)
        return q, k, v
        

def scaled_dot_product_attention(q, k, v, mask):
    """
    q: [batch_size, num_heads, head_dim, seq1_len]
    k: [batch_size, num_heads, head_dim, seq2_len]
    v: [batch_size, num_heads, head_dim, seq2_len]
    mask: [batch_size, num_heads, seq1_len, seq1_len]
    (seq1_len = seq2_len for self attention)
    """
    qk = q.matmul(k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if mask:
        mask = torch.tril(torch.ones(1, 1, qk.shape[-2], qk.shape[-1])).type(torch.bool).to(qk.device)
        qk = qk.masked_fill(~mask, -torch.inf)
    attn_weights = qk.softmax(dim=-1)
    return attn_weights.matmul(v)


class CompressKV(nn.Module):

    def __init__(self, input_shape, compress_factor):
        super(CompressKV, self).__init__()
        self.compress_factor = compress_factor
        padding_tensor = torch.zeros(1, 1, compress_factor)
        self.register_buffer("padding_tensor", padding_tensor, persistent=False)
        self.conv = nn.Conv1d(
            in_channels=input_shape,
            out_channels=input_shape,
            kernel_size=compress_factor,
            stride=compress_factor
        )

    def pad(self, x):
        pad_amt = self.compress_factor - (x.shape[-1] % self.compress_factor)
        pad_amt = 0 if pad_amt == self.compress_factor else pad_amt
        return torch.cat([x, self.padding_tensor.repeat(x.shape[0], x.shape[1], 1)[:, :, :pad_amt]], dim=-1)

    def forward(self, x):
        """
        x: [batch_size, seq_len, embed_dim]
        """
        x = x.permute(0, 2, 1)
        x = self.pad(x)
        return self.conv(x).permute(0, 2, 1)


class LocalAttention(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads, block_size):
        super(LocalAttention, self).__init__()
        self.block_size = block_size
        self.qkv_layer = QKVLayer(input_dim, output_dim, num_heads)
        self.out_proj = nn.Linear(output_dim, output_dim)

    def reshape(self, x):
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)

    def forward(self, x, mask):
        q, k, v = self.qkv_layer(x)
        q, k, v = self.qkv_layer.reshape(q), self.qkv_layer.reshape(k), self.qkv_layer.reshape(v)
        num_chunks = (q.shape[-2] // self.block_size) + 1
        q, k, v = q.chunk(num_chunks, dim=-2), k.chunk(num_chunks, dim=-2), v.chunk(num_chunks, dim=-2)
        attn_outputs = []
        for q_chunk, k_chunk, v_chunk in zip(q, k, v):
            attn_outputs.append(scaled_dot_product_attention(q_chunk, k_chunk, v_chunk, mask=True))
        attn_outputs = torch.cat(attn_outputs, dim=-2)
        return self.reshape(self.out_proj(attn_outputs))


class CompressedMultiHeadAttention(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads, compress_factor):
        super(CompressedMultiHeadAttention, self).__init__()
        self.qkv_layer = QKVLayer(input_dim, output_dim, num_heads)
        self.compress_k = CompressKV(output_dim, compress_factor)
        self.compress_v = CompressKV(output_dim, compress_factor)
        self.out_proj = nn.Linear(output_dim, output_dim)

    def reshape(self, x):
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)

    def forward(self, x, mask):
        """
        x: [batch_size, seq_len, embed_dim]
        kv: (k, v) where k, v: [batch_size, seq_len, num_heads * per_head_dim]
        (kv is not None when attending to stored keys and values)
        """
        q, k, v = self.qkv_layer(x)
        k, v = self.compress_k(k), self.compress_v(v)
        q, k, v = self.qkv_layer.reshape(q), self.qkv_layer.reshape(k), self.qkv_layer.reshape(v)
        attn_outputs = scaled_dot_product_attention(q, k, v, mask)
        attn_outputs = self.out_proj(attn_outputs)
        return self.reshape(attn_outputs)