import sys
import math
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(self, n_head, n_embed, d_head, dropout, dropatt=0, pre_lnorm=False):
        super(RelPartialLearnableMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_head = d_head
        self.n_embed = n_embed
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm

        self.qkv_net = nn.Linear(n_embed, 3 * n_head * d_head, bias=False)
        self.qkv_net1 = nn.Linear(n_embed, 3 * n_head * d_head, bias=False)
        self.r_net = nn.Linear(n_embed, n_head * d_head, bias=False)
        self.o_net = nn.Linear(n_head * d_head, n_embed, bias=False)
        self.o_net1 = nn.Linear(n_head * d_head, n_embed, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.layer_norm = nn.LayerNorm(n_embed)

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = x_padded[1:].view_as(x)
        return x

    def forward(self, w_real, w_phase, r, r_w_bias, r_r_bias, attn_mask=None, mems=None, mems_phase=None):
        qlen, rlen, bsz = w_real.size(0), r.size(0), w_real.size(1)

        if mems is not None:
            cat_real = torch.cat([mems, w_real], 0)
            cat_phase = torch.cat([mems_phase, w_phase], 0)
            if self.pre_lnorm:
                w_heads_real = self.qkv_net(self.layer_norm(cat_real))
                w_heads_phase = self.qkv_net1(self.layer_norm(cat_phase))
            else:
                w_heads_real = self.qkv_net(cat_real)
                w_heads_phase = self.qkv_net1(cat_phase)
            r_head_k = self.r_net(r)
            w_head_q_real, w_head_k_real, w_head_v_real = torch.chunk(w_heads_real, 3, dim=-1)
            w_head_q_phase, w_head_k_phase, w_head_v_phase = torch.chunk(w_heads_phase, 3, dim=-1)
            w_head_q_real = w_head_q_real[-qlen:]
            w_head_q_phase = w_head_q_phase[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads_real = self.qkv_net(self.layer_norm(w_real))
                w_heads_phase = self.qkv_net1(self.layer_norm(w_phase))
            else:
                w_heads_real = self.qkv_net(w_real)
                w_heads_phase = self.qkv_net1(w_phase)
            r_head_k = self.r_net(r)
            w_head_q_real, w_head_k_real, w_head_v_real = torch.chunk(w_heads_real, 3, dim=-1)
            w_head_q_phase, w_head_k_phase, w_head_v_phase = torch.chunk(w_heads_phase, 3, dim=-1)

        klen = w_head_k_real.size(0)

        w_head_q_real = w_head_q_real.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_q_phase = w_head_q_phase.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        w_head_k_real = w_head_k_real.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k_phase = w_head_k_phase.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        w_head_v_real = w_head_v_real.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v_phase = w_head_v_phase.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        #### compute attention score
        rw_head_q_real = w_head_q_real + r_w_bias  # qlen x bsz x n_head x d_head
        rw_head_q_phase = w_head_q_phase + r_w_bias  # qlen x bsz x n_head x d_head

        AC_real = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q_real, w_head_k_real)) - torch.einsum('ibnd,jbnd->ijbn', (
        rw_head_q_phase, w_head_k_phase))  # qlen x klen x bsz x n_head
        AC_phase = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q_real, w_head_k_phase)) + torch.einsum('ibnd,jbnd->ijbn', (
        rw_head_q_real, w_head_k_phase))  # qlen x klen x bsz x n_head

        rr_head_q_real = w_head_q_real + r_r_bias
        rr_head_q_phase = w_head_q_phase + r_r_bias

        BD_real = torch.einsum('ibnd,jnd->ijbn', (rr_head_q_real, r_head_k))  # qlen x klen x bsz x n_head
        BD_phase = torch.einsum('ibnd,jnd->ijbn', (rr_head_q_phase, r_head_k))  # qlen x klen x bsz x n_head

        BD_real = self._rel_shift(BD_real)
        BD_phase = self._rel_shift(BD_phase)

        # [qlen x klen x bsz x n_head]
        AC = AC_real * AC_real + AC_phase * AC_phase
        AC = torch.sqrt(AC)

        BD = BD_real * BD_real + BD_phase * BD_phase
        BD = torch.sqrt(BD)

        attn_score = AC + BD

        attn_score.mul_(self.scale)
        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec_real = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v_real))
        attn_vec_phase = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v_phase))

        # [qlen x bsz x n_head x d_head]

        attn_vec_real = attn_vec_real.contiguous().view(attn_vec_real.size(0), attn_vec_real.size(1),
                                                        self.n_head * self.d_head)
        attn_vec_phase = attn_vec_phase.contiguous().view(attn_vec_phase.size(0), attn_vec_phase.size(1),
                                                          self.n_head * self.d_head)

        ##### linear projection
        attn_out_real = self.o_net(attn_vec_real)
        attn_out_phase = self.o_net1(attn_vec_phase)
        attn_out_real = self.drop(attn_out_real)
        attn_out_phase = self.drop(attn_out_phase)

        if self.pre_lnorm:
            ##### residual connection
            output_real = attn_out_real
            output_phase = attn_out_phase
        else:
            ##### residual connection + layer normalization
            output_real = self.layer_norm(w_real + attn_out_real)
            output_phase = self.layer_norm(w_phase + attn_out_phase)


        return output_real, output_phase