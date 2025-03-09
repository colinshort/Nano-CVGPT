import torch
import torch.nn as nn
from layers import ComplexLinear, ComplexLayerNorm, ComplexDropout
from attention import MultiHeadAttention
from activations import CReLU


class ComplexBlock(nn.Module):
  """ Transformer block: communication followed by computation """
  def __init__(self, n_embed, n_head, dropout, device='cpu'):
    super().__init__()
    self.sa_heads = MultiHeadAttention(n_embed, n_head, attn_dropout=dropout)
    self.ffwd = ComplexFeedForward(n_embed, dropout, device=device)
    self.ln1 = ComplexLayerNorm(n_embed, device=device)
    self.ln2 = ComplexLayerNorm(n_embed, device=device)

  def forward(self, x):
    x = x + self.sa_heads(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
  

class ComplexFeedForward(nn.Module):
  """ simple linear layyer followed by a non-linearity """
  def __init__(self, n_embed, dropout, device='cpu'):
    super().__init__()
    self.net = nn.Sequential(
      ComplexLinear(n_embed, 4 * n_embed),
      CReLU(),
      ComplexLinear(4 * n_embed, n_embed),
      ComplexDropout(dropout, device=device),
    )

  def forward(self, x):
    return self.net(x)