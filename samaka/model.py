import torch
from torch import nn

class Head(nn.Module):
  def __init__(self, embed_size, head_size, dropout = 0.2):
    super().__init__()

    self.head_size = head_size

    self.K = nn.Linear(embed_size, head_size, bias = False)
    self.Q = nn.Linear(embed_size, head_size, bias = False)
    self.V = nn.Linear(embed_size, head_size, bias = False)

    self.dropout = nn.Dropout(dropout)

  def forward(self, x : torch.tensor):
    k, q = self.K(x), self.Q(x)
    w = q @ k.transpose(-2, -1) * self.head_size ** -0.5
    w = self.dropout(w)
    w = w @ self.V(x)
    return w

class MultiHeadAttention(nn.Module):
  def __init__(self, embed_size, num_heads, dropout = 0.2):
    super().__init__()

    head_size = embed_size // num_heads
    assert (num_heads * head_size) == embed_size, "num_heads must be divisable on embed_size"

    self.heads = [Head(embed_size, head_size, dropout) for _ in range(num_heads)]
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([head(x) for head in self.heads], dim = -1)
    out = self.dropout(out)
    return out

class FeedForward(nn.Module):
  def __init__(self, embed_size, dropout = 0.2):
    super().__init__()

    self.net = nn.Sequential(
        nn.Linear(embed_size, embed_size * 4),
        nn.ReLU(),
        nn.Linear(embed_size * 4, embed_size * 6),
        nn.ReLU(),
        nn.Linear(embed_size * 6, embed_size * 4),
        nn.ReLU(),
        nn.Linear(embed_size * 4, embed_size),
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.dropout( self.net( x ) )

class Block(nn.Module):
  def __init__(self, embed_size, num_heads, dropout = 0.2):
    super().__init__()

    self.multihead = MultiHeadAttention(embed_size, num_heads, dropout)
    self.ff = FeedForward(embed_size, dropout)
    self.ln1 = nn.LayerNorm(embed_size)
    self.ln2 = nn.LayerNorm(embed_size)

  def forward(self, x):
    x = self.ln1( self.multihead( x ) )
    x = self.ln2( self.ff( x ) )
    return x

class Samaka(nn.Module):
  def __init__(self, embed_size, num_layers, num_heads, dropout = 0.2):
    super().__init__()

    self.table = nn.Embedding(13, embed_size)
    self.layers = nn.Sequential(
        *[
            Block(embed_size, num_heads, dropout) for _ in range(num_layers)
        ]
    )
    self.ff = FeedForward(embed_size, dropout)
    self.ln = nn.LayerNorm(embed_size)

    self.eval_pos = nn.Sequential(
        nn.Linear(embed_size, embed_size * 4),
        nn.ReLU(),
        nn.Linear(embed_size * 4, embed_size * 4),
        nn.ReLU(),
        nn.Linear(embed_size * 4, embed_size * 2),
        nn.ReLU(),
        nn.Linear(embed_size * 2, 1),
    )

  def forward(self, x):
    x = self.table(x)
    x = self.ff( self.layers( x ) )
    x = self.ln( x )
    x = self.eval_pos( x )
    x = torch.sum(x)
    return x