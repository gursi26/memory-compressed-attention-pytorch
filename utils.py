from torch import nn
import torch
import math


class PositionalEncoder(nn.Module):

    def __init__(self, d_model, max_seq_len, p=0.1):
        super(PositionalEncoder, self).__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=p)

    # x has shape [batch, seq_len, embed_dim]
    def forward(self, x):
        _, seq_len, _ = x.shape
        out = self.dropout(x + self.pe[:, :seq_len, :])
        return out
    

class FeedForward(nn.Module):

    def __init__(self, input_dim):
        super(FeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim)
        )

    def forward(self, x):
        return self.layer(x)
