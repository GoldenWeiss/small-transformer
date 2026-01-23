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

class LayerNormalization(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x