import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                             (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, max_len):
        super(TransformerLayer, self).__init__()
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Apply positional encoding
        x = self.pos_encoding(x)
        
        # Transpose for attention layer, which expects (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)
        
        # Apply multi-head attention
        attn_output, attn_weights = self.attention(x, x, x)
        
        # Add residual connection and layer normalization
        x = self.norm(attn_output + x)
        
        # Transpose back to (batch, seq_len, embed_dim)
        return x.transpose(0, 1), attn_weights
