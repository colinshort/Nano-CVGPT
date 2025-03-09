import torch
from torch import nn


class positional_encoding(nn.Module):
    def __init__(self, embed_dim, max_freq=10000, device='cpu'):

        self.embed_dim = embed_dim
        super().__init__()
        half_dim = embed_dim // 2
        powers = torch.arange(start=0, end=half_dim) / half_dim
        self.div = max_freq**(powers).unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, input):
        """
        gets (relativ) positions of current token, returns positional embedding as vector of size embed_dim
        """
        tokens_num = input.shape[0]

        sin_pos = torch.sin(input.unsqueeze(-1) / self.div).unsqueeze(-1)
        cos_pos = torch.cos(input.unsqueeze(-1) / self.div).unsqueeze(-1)
        pos = torch.cat((sin_pos, cos_pos), dim=-1).reshape([1, tokens_num, self.embed_dim])  # alternating reshape because last dim first
        return pos
