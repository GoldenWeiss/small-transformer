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

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_rate):
        super().__init__()
        self.embed_dim = embedding_dim
        self.num_heads = num_heads
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embedding_dim // num_heads # Dimension per head
        self.w_q = nn.Linear(embedding_dim, embedding_dim) # Query linear layer
        self.w_k = nn.Linear(embedding_dim, embedding_dim) # Key linear layer
        self.w_v = nn.Linear(embedding_dim, embedding_dim) # Value linear layer
        self.w_o = nn.Linear(embedding_dim, embedding_dim) # Output linear layer
        self.dropout = nn.Dropout(dropout_rate)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        p_attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
    
    def forward(self, q, k, v, mask):
        batch_size = q.size(0)

        # Linear projections
        query = self.w_q(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.w_k(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.w_v(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply attention
        x, attn = self.attention(query, key, value, mask, self.dropout)

        # Concatenate heads and apply final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, embed_dim, dropout_rate):
        super().__init__()
        self.norm = LayerNormalization(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate):
        super().__init__()
        self.attention = MultiHeadAttentionBlock(embed_dim, num_heads, dropout_rate)
        self.feed_forward = FeedForwardBlock(embed_dim, ff_dim, dropout_rate)
        self.residual1 = ResidualConnection(embed_dim, dropout_rate)
        self.residual2 = ResidualConnection(embed_dim, dropout_rate)

    def forward(self, x, mask):
        x = self.residual1(x, lambda x: self.attention(x, x, x, mask))
        x = self.residual2(x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, embed_dim, layers :nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(embed_dim)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, self_attention_block : MultiHeadAttentionBlock, cross_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout_rate):
        super().__init__()
        self.self_attention = self_attention_block
        self.cross_attention = cross_attention_block
        self.feed_forward = feed_forward_block
        self.residual1 = ResidualConnection(embed_dim, dropout_rate)
        self.residual2 = ResidualConnection(embed_dim, dropout_rate)
        self.residual3 = ResidualConnection(embed_dim, dropout_rate)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual2(x, lambda x: self.cross_attention(x, enc_output, enc_output, src_mask))
        x = self.residual3(x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, embed_dim, layers :nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(embed_dim)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, embed_dim, n_tokens):
        super().__init__()
        self.linear = nn.Linear(embed_dim, n_tokens)

    def forward(self, x):
        return self.linear(x)