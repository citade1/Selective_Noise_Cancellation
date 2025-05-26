import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """classic sinusoidal positional encoding function"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        self.pe = torch.zeros(max_len, d_model) # shape = (T, D) -> (max sequence length, embedding dimensions)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # pos, shape=(T, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # denominator: e^(-log(10000)*2i/d_model), shape=(D/2,)

        self.pe[:, 0::2] = torch.sin(position * div_term) # shape = (T, D/2)
        self.pe[:, 1::2] = torch.cos(position * div_term) # shape = (T, D/2)

        self.pe = self.pe.unsqueeze(0) # shape = (1, T, D)

        self.register_buffer("pe", self.pe) # non-learnable

    def forward(self, x):
        # x : (B, T, D) ->(batch, sequence_len, embedding_dim)
        T = x.size(1) 
        x = x + self.pe[:, :T, :]

        return x

class MultiHeadSelfAttention(nn.Module):
    """Vanilla Multi-head Self-attention from Attention is all you need
    
    Note:
    - Q, K, V are computed using a single linear layer (`qkv_projection`) for efficiency.
    - The output is a combination of all attention heads followed by a final projection.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model # embedding dimension OR input/output dimension of attention block : 256
        self.n_heads = n_heads # 4
        self.d_k = d_model // n_heads # dimension of each head : 64 

        self.qkv_projection = nn.Linear(d_model, 3*d_model) # projects input into a single matrix, which will be divided into Q, K, V

        self.output_matrix = nn.Linear(d_model, d_model) # Projects the concatenated attn back to the original embedding size
    
    def forward(self, x):
        
        # x : (B, T, D) -> (batch_size, timesteps/sequence_length, dimension)
        B, T, D = x.shape

        qkv = self.qkv_projection(x) # (B, T, 3*D)
        qkv = qkv.reshape(B, T, self.n_heads, 3*self.d_k).permute(0, 2, 1, 3) # (B, n_heads, T, 3*d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # (B, n_heads, T, d_k)

        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k) # (B, n_heads, T, T)
        attn = F.softmax(scores, dim=-1) 
        attended = torch.matmul(attn, v) # (B, n_heads, T, d_k)
        attended = attended.permute(0, 2, 1, 3).reshape(B, T, D)
        
        return self.output_matrix(attended)


class TransformerEncoderBlock(nn.Module):
    """
    A single encoder block consisting of:
    - Multi-head Self-Attention
    - Feedforward Network
    - LayerNorm and Dropout
    """

    def __init__(self, d_model, n_heads, dim_ff, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x : (B, T, D)
        
        # attention layer
        x = self.norm1(x + self.dropout(self.attn(x)))
        
        # position-wise feed fordward network layer
        x = self.norm2(x + self.dropout(self.ff(x)))

        return x 

class StackedTransformerEncoder(nn.Module):
    """
    Stacks multiple TransformerEncoderBlocks.
    Useful for deeper attention bottleneck.
    """
    def __init__(self, d_model, n_heads, dim_ff, num_layers=1, dropout=0.1, max_len=5000):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.Sequential(*[TransformerEncoderBlock(d_model, n_heads, dim_ff, dropout) for _ in range(num_layers)])
    
    def forward(self, x):
        x = self.pos_enc(x)
        return self.layers(x)
    


        