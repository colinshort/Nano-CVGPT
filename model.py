import torch
from torch import nn
import torch.nn.functional as F
from layers import ComplexLinear, ComplexLayerNorm
from components import ComplexBlock
from positional_encoding import positional_encoding


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

        # each token directly reads off the logits for the next token from a lookup table
        # self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embed)
        # self.position_embedding_table = nn.Embedding(self.block_size, self.n_embed)

        self.in_embedding = ComplexLinear(self.vocab_size, self.n_embed)
        self.pos_enc = positional_encoding(embed_dim=self.n_embed, device=self.device)
        self.blocks = nn.Sequential(*[ComplexBlock(self.n_embed, self.n_head, self.dropout) for _ in range(self.n_layer)])
        self.ln_f = ComplexLayerNorm(self.n_embed) # final layer norm
        self.lm_head = ComplexLinear(self.n_embed, self.vocab_size)

        # self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, ComplexLinear):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        # x = tok_emb + pos_emb # (B,T,C)

        x = self.in_embedding(idx) + self.pos_enc(torch.arange(T))
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
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
        

