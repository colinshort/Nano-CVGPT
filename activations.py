import torch
import torch.nn as nn
import numpy as np 


class CReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.LeakyReLU(inplace=inplace)

    def forward(self, input):
        out_real = self.relu(input.real)
        out_imag = self.relu(input.imag)
        return out_real + 1j * out_imag  # torch.complex(out_real, out_imag) not used because of memory requirements
