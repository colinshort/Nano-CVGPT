import torch
from torch import nn
import torch.nn.functional as F
from layers import AdaptiveEmbedding, ComplexLinear, ComplexLayerNorm
from components import ComplexBlock


class CVGPT(nn.Module):  # Transformer for sequence generation
    def __init__(self, vocab_size, n_embed, n_head, n_layer, block_size, dropout, device='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout
        self.device = device

        # learned embedding that is a function of token index and position
        self.embedding = AdaptiveEmbedding(vocab_size, n_embed, d_proj=n_embed, cutoffs=[], div_val=1)
        self.blocks = nn.Sequential(*[ComplexBlock(self.n_embed, self.n_head, self.dropout, device=self.device) for _ in range(self.n_layer)])
        self.ln_f = ComplexLayerNorm(self.n_embed, device=device) # final layer norm
        self.lm_head = ComplexLinear(self.n_embed, self.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        x = self.embedding(idx)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x).real # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
        

