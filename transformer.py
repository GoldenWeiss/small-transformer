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
    
    