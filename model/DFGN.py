import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DFGN(nn.Module):
    '''Distinct (Dynamic in this project) Filter Generation Network
     impl. FFN as in the paper
    '''
    def __init__(
        self,
        memory_size=16,
        hiddens=[16, 4,],
        out_shapes=[(4, 3), (4,)],
    ):
        super().__init__()
        self.out_shapes = out_shapes
        num_layers = len(hiddens) + 1
        out_size = 0
        for out_shape in out_shapes:
            out_size += np.prod(out_shape)
        channels = [memory_size] + hiddens + [out_size]
        layers = []
        for i in range(num_layers):
            in_feat = channels[i]
            out_feat = channels[i+1]
            layers.append(nn.Linear(in_feat, out_feat))
            if i != num_layers-1:  # act except last layer
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def forward(self, memory: torch.Tensor):
        outs_flat = self.net(memory)
        outs = []; l = 0
        for out_shape in self.out_shapes:
            out_nelem = np.prod(out_shape)
            out = outs_flat[..., l:l+out_nelem].reshape(out_shape)
            outs.append(out)
            l += out_nelem
        return outs


class D_MLP(nn.Module):
    def __init__(
        self,
        channels=[370, 64, 16],
    ):
        super().__init__()
        self.num_layers = len(channels) - 1
        dfgns = []
        for i in range(self.num_layers):
            in_feat = channels[i]
            out_feat = channels[i+1]
            # use indivdual DFGN to gen weight and bias for each MLP layer
            dfgn = DFGN(out_shapes=[(out_feat, in_feat), (out_feat,)])
            dfgns.append(dfgn)
        self.dfgns = nn.ModuleList(dfgns)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor, memory: torch.Tensor):
        i = 0
        for dfgn in self.dfgns:
            weight, bias = dfgn(memory)
            x = F.linear(x, weight, bias)
            if i != self.num_layers-1:
                x = self.act(x)
        return x
