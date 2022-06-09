import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    '''a module doing (N, L) -> (N, L)'''
    def __init__(self, n_channels, depth, kernel_size=24, ):
        super().__init__()
        self.mixer = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv1d(n_channels, n_channels, kernel_size,
                              groups=n_channels, padding="same", padding_mode="replicate"),
                    nn.ELU(),
                )),
                nn.Conv1d(n_channels, n_channels, kernel_size=1),
                nn.ELU(),
            ) for i in range(depth)],
        )

    def forward(self, x: t.Tensor):
        x = x.unsqueeze(0)
        out = self.mixer(x)
        out = out.squeeze(0)
        return out


class LKA(nn.Module):
    '''another flavor of mixer encoding: (N, L) -> (N, L)
    ref. https://github.com/Visual-Attention-Network/VAN-Classification/blob/main/models/van.py
    '''
    def __init__(self, n_channels, depth, kernel_size=72, dilation=12):
        super().__init__()
        K1, rem = divmod(kernel_size, dilation)
        assert rem == 0  # convenience
        K0 = 2*dilation - 1

        self.attn_layers = nn.ModuleList(
            [nn.Sequential(
                nn.ELU(),
                nn.Conv1d(n_channels, n_channels, K0,
                          groups=n_channels, padding="same", padding_mode="replicate"),
                nn.Conv1d(n_channels, n_channels, K1, dilation=dilation,
                          groups=n_channels, padding="same", padding_mode="replicate"),
                nn.Conv1d(n_channels, n_channels, kernel_size=1),
            ) for i in range(depth)],
        )

    def forward(self, x: t.Tensor):
        x = x.unsqueeze(0)
        for layer in self.attn_layers:
            attn = layer(x)
            u = attn * x
            x = u + x
        out = x.squeeze(0)
        return out


def filtered_attention(
    q: t.Tensor,
    k: t.Tensor,
    v: t.Tensor,
    topk: Optional[int] = 20
) -> Tuple[t.Tensor, t.Tensor]:
    r"""ref. F._scaled_dot_product_attention
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = t.bmm(q, k.transpose(-2, -1))
    # filter top k
    if topk is not None:
        _, i = t.topk(attn, topk)
        attn_mask = t.empty_like(attn)
        attn_mask.fill_(float("-inf"))
        attn_mask.scatter_(-1, i, 0)
        attn += attn_mask
    # fuse weight
    attn = F.softmax(attn, dim=-1)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = t.bmm(attn, v)
    return output, attn


class SeriesAttn(nn.MultiheadAttention):
    '''modify nn.MultiheadAttention to include topk select
    '''

    def __init__(self, embed_dim=512, num_heads=8,
                 topk: Optional[int] = 20, **kwargs):
        super().__init__(embed_dim, num_heads, **kwargs)
        self.topk = topk

    def forward(self, x, **kwargs):
        x = x.unsqueeze(1)
        # F.multi_head_attention_forward
        query, key, value = x, x, x
        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        head_dim = embed_dim // self.num_heads
        q, k, v = F._in_projection_packed(query, key, value, 
            self.in_proj_weight, self.in_proj_bias)
        # reshape q, k, v for multihead attention and make em batch first
        q = q.view(tgt_len, self.num_heads, head_dim).transpose(0, 1)
        k = k.view(tgt_len, self.num_heads, head_dim).transpose(0, 1)
        v = v.view(tgt_len, self.num_heads, head_dim).transpose(0, 1)
        # (deep breath) calculate attention and out projection
        attn_output, attn_output_weights = filtered_attention(
            q, k, v, self.topk
        )
        attn_output = attn_output.transpose(
            0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, 
            self.out_proj.weight, self.out_proj.bias)
        return attn_output.squeeze(1)  # , attn_score.squeeze(0)


class SeriesAttn1(nn.Module):
    '''modify nn.MultiheadAttention to 
    include topk select and qkv dim
    '''

    def __init__(self, embed_dim=512, qkv_dim=32*3, num_heads=3,
                 topk: Optional[int] = 20, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_dim = qkv_dim
        self.head_dim = qkv_dim // num_heads
        self.num_heads = num_heads
        self.topk = topk
        self.in_proj_weight = nn.Parameter(t.empty((3 * qkv_dim, embed_dim)))
        self.in_proj_bias = nn.Parameter(t.empty(3 * qkv_dim))
        self.out_proj = nn.Linear(qkv_dim, embed_dim, )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x, **kwargs):
        x = x.unsqueeze(1)
        # F.multi_head_attention_forward
        query, key, value = x, x, x
        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim
        src_len, _, _ = key.shape
        q, k, v = F._in_projection_packed(query, key, value,
                                          self.in_proj_weight, self.in_proj_bias)
        # reshape q, k, v for multihead attention and make em batch first
        head_dim = self.head_dim
        q = q.view(tgt_len, self.num_heads, head_dim).transpose(0, 1)
        k = k.view(tgt_len, self.num_heads, head_dim).transpose(0, 1)
        v = v.view(tgt_len, self.num_heads, head_dim).transpose(0, 1)
        # (deep breath) calculate attention and out projection
        attn_output, attn_output_weights = filtered_attention(
            q, k, v, self.topk
        )
        attn_output = attn_output.transpose(
            0, 1).contiguous().view(tgt_len, bsz, self.qkv_dim)
        attn_output = F.linear(attn_output,
                               self.out_proj.weight, self.out_proj.bias)
        return attn_output.squeeze(1)  # , attn_score.squeeze(0)


class DefAttn(nn.Module):
    def __init__(self, embed_dim=512, vdim=96,
                 adj_mx: t.Tensor = None, ):
        super().__init__()
        self.attn_w = F.softmax(adj_mx, -1)
        self.embed_dim = embed_dim
        self.vdim = vdim
        self.v_proj = nn.Linear(embed_dim, vdim)
        self.out_proj = nn.Linear(vdim, embed_dim)

    def forward(self, x, **kwargs):
        attn_w = self.attn_w.type_as(x)
        v = self.v_proj(x)
        attn_output = attn_w @ v
        attn_output = self.out_proj(attn_output)

        return attn_output
