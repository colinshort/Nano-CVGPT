import torch
import torch.nn as nn
from layers import ComplexLinear, ComplexLayerNorm, ComplexDropout
from attention import RelPartialLearnableMultiHeadAttn
from activations import CReLU


class ComplexBlock(nn.Module):
  """ Transformer block: communication followed by computation """
  def __init__(self, n_embed, n_head, d_head, d_inner, dropout, device='cpu', sharing_phase_weight=False):
      super(ComplexBlock, self).__init__()

      self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, n_embed, d_head, dropout, sharing_phase_weight)
      self.pos_ff = ComplexFeedForward(n_embed, d_inner, dropout, device)
      self.ln_1 = ComplexLayerNorm(n_embed)
      self.ln_2 = ComplexLayerNorm(n_embed)

  def forward(self, x):
    x = x + self.dec_attn(self.ln_1(x))
    x = x + self.pos_ff(self.ln_2(x))
    
    return x

class ComplexFeedForward(nn.Module):
  """ Simple linear layer followed by a non-linearity """
  def __init__(self, n_embed, d_inner, dropout, device='cpu'):
    super().__init__()
    self.net = nn.Sequential(
      ComplexLinear(n_embed, d_inner),
      CReLU(),
      ComplexLinear(d_inner, n_embed),
      ComplexDropout(dropout, device=device),
    )

  def forward(self, x):
    return self.net(x)