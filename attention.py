import torch
from torch import nn
from torch.nn.functional import softmax
from layers import ComplexLinear, ComplexDropout


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_head, attn_dropout=0, bias=True, sm_variante='realip', device='cpu'):
        super().__init__()
        self.sm_variante = sm_variante
        self.n_embed = n_embed
        self.n_head = n_head
        self.attn_dropout = attn_dropout
        self.head_size = n_embed // n_head
        self.device = device
        assert self.head_size * n_head == self.n_embed, "n_embed must be divisible by n_head"
        self.scaling = self.head_size ** -0.5
        self.in_proj_q = ComplexLinear(n_embed, n_embed, bias=bias)
        self.in_proj_k = ComplexLinear(n_embed, n_embed, bias=bias)
        self.in_proj_v = ComplexLinear(n_embed, n_embed, bias=bias)
        self.out_proj = ComplexLinear(n_embed, n_embed, bias=bias)

        self.cdropout = ComplexDropout(attn_dropout)
        self.product = self.sm_variante[-2:]
        self.sm_variante = self.sm_variante[:-2]

    def forward(self, x):
        batch_size, input_len, n_embed = x.size()

        q = self.in_proj_q(x)
        k = self.in_proj_k(x)
        v = self.in_proj_v(x)

        q = q.transpose(1, 0).contiguous().view(input_len, batch_size * self.n_head, self.head_size).transpose(1, 0)
        k = k.transpose(1, 0).contiguous().view(input_len, batch_size * self.n_head, self.head_size).transpose(1, 0)
        v = v.transpose(1, 0).contiguous().view(input_len, batch_size * self.n_head, self.head_size).transpose(1, 0)

        if self.product == 'cp':
            attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scaling
        elif self.product == 'ip':
            attn_weights = torch.bmm(q, torch.conj_physical(k).transpose(1, 2)) * self.scaling
        else:
            raise ValueError(f'{self.product} is not a valid argument')


        attn_mask = self.generate_square_subsequent_mask(x.shape[1])
        attn_weights = self.softmax_variants(attn_weights, attn_mask=attn_mask, sm_variante=self.sm_variante)
        attn_weights = self.cdropout(attn_weights)
        attn = torch.bmm(attn_weights, v)

        attn = attn.transpose(1, 0).contiguous().view(input_len, batch_size, self.n_embed).transpose(1, 0)
        attn = self.out_proj(attn)

        return attn

    def softmax_variants(self, input, attn_mask=None, sm_variante='real'):
        if sm_variante == 'real':
            return self.softmax_real(input, attn_mask=attn_mask)
        elif sm_variante == 'abs':
            return self.softmax_abs(input, attn_mask=attn_mask)
        elif sm_variante == 'naiv':
            return self.softmax_naiv(input, attn_mask=attn_mask)
        elif sm_variante == 'absonly':
            return self.softmax_abs_only(input, attn_mask=attn_mask)
        else:
            raise ValueError(f'{sm_variante} is not a valid variant for C-softmax')

    def softmax_abs(self, input, attn_mask=None):
        abso = torch.abs(input)
        if attn_mask is not None:
            abso += attn_mask.unsqueeze(0).real.to(self.device)
        return softmax(abso, dim=-1).type(torch.complex64) * torch.sgn(input)

    def softmax_naiv(self, input, attn_mask=None):
        if attn_mask is not None:
            # input += attn_mask.unsqueeze(0).to(self.device)
            input = torch.complex(input.real + attn_mask.unsqueeze(0).to(self.device).real, input.imag + attn_mask.unsqueeze(0).to(self.device).imag)
        return torch.complex(softmax(input.real, dim=-1), softmax(input.imag, dim=-1))

    def softmax_abs_only(self, input, attn_mask=None):
        abso = torch.abs(input)
        if attn_mask is not None:
            abso += attn_mask.unsqueeze(0).real.to(self.device)
        # abso[abso == float('inf')] = -abso[abso == float('inf')]
        return softmax(abso, dim=-1).type(torch.complex64)

    def softmax_real(self, input, attn_mask=None):
        real = torch.real(input)
        if attn_mask is not None:
            real += attn_mask.unsqueeze(0).real.to(self.device)
        # abso[abso == float('inf')] = -abso[abso == float('inf')]
        return softmax(real, dim=-1).type(torch.complex64)

    def min_max_real(self, input, attn_mask=None):  # attnmask does not work yet
        real = torch.real(input)
        mini = torch.min(real, dim=-1)[0].unsqueeze(-1)
        maxi = torch.max(real, dim=-1)[0].unsqueeze(-1)
        return ((real - mini) / (maxi - mini)).type(torch.complex64)

    def min_max_naiv(self, input, attn_mask=None):  # attnmask does not work yet
        real = torch.real(input)
        rmini = torch.min(real, dim=-1)
        rmaxi = torch.max(real, dim=-1)
        imag = torch.imag(input)
        imini = torch.min(imag, dim=-1)
        imaxi = torch.max(imag, dim=-1)
        return torch.complex((real - rmini) / (rmaxi - rmini), (imag - imini) / (imaxi - imini))

    def min_max_complex(self, input, attn_mask=None):  # attnmask does not work yet
        mini = torch.take_along_dim(input, torch.argmin(input.real, dim=-1, keepdims=True), dim=-1).unsqueeze(-1)
        pos = input - mini
        maxi = torch.take_along_dim(input, torch.argmax(torch.abs(pos), dim=-1, keepdims=True), dim=-1).unsqueeze(-1)
        return pos / maxi

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return torch.complex(mask, mask)
    