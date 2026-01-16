import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, embed_dim, n_tokens):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tokens = n_tokens
        self.embedding = nn.Embedding(n_tokens, embed_dim)

    def forward(self, x):
        return self.embedding * math.sqrt(self.embed_dim)
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, n_tokens, dropout_rate):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tokens = n_tokens
        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(n_tokens, embed_dim)
        position = torch.arange(0, n_tokens).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)