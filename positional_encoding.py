import torch
from torch import nn


class positional_encoding(nn.Module):
    def __init__(self, n_embed, max_freq=10000, device='cpu'):

        self.n_embed = n_embed
        super().__init__()
        half_dim = n_embed // 2
        powers = torch.arange(start=0, end=half_dim) / half_dim
        self.div = max_freq**(powers).unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, input):
        """
        gets (relativ) positions of current token, returns positional embedding as vector of size n_embed
        """
        tokens_num = input.shape[0]

        sin_pos = torch.sin(input.unsqueeze(-1) / self.div).unsqueeze(-1)
        cos_pos = torch.cos(input.unsqueeze(-1) / self.div).unsqueeze(-1)
        pos = torch.cat((sin_pos, cos_pos), dim=-1).reshape([1, tokens_num, self.n_embed])  # alternating reshape because last dim first
        return pos
