from typing import Any, Tuple
import torch as t
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.batchnorm import _NormBase
import torch.nn.functional as F


class RevIN(_NormBase):
    '''Applies Reversible Instance Normalization.
    Similar to InstanceNorm1d but provides inverse transform.
    '''
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
    ):
        super(RevIN, self).__init__(
            num_features, eps, None, affine, False, )

    def _get_no_batch_dim(self):
        return 2

    def _check_input_dim(self, input):
        if input.dim() not in (2, 3):
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

    def _handle_no_batch_input(self, input):
        out, state = self._apply_instance_norm(input.unsqueeze(0))
        return out.squeeze(0), state

    def _apply_instance_norm(self, x):
        means = x.mean(-1, keepdim=True).detach()
        stdev = t.sqrt(t.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps)
        x = (x - means) / stdev
        
        if self.affine:
            w = self.weight.unsqueeze(-1); b = self.bias.unsqueeze(-1)
            x = x * w + b

        return x, (means, stdev)

    def forward(self, input: Tensor) -> Tuple[Tensor, Any]:
        self._check_input_dim(input)

        if input.dim() == self._get_no_batch_dim():
            return self._handle_no_batch_input(input)

        return self._apply_instance_norm(input)

    def inverse_transform(self, input: Tensor, state: Any):
        self._check_input_dim(input)

        if input.dim() == self._get_no_batch_dim():
            y = input.unsqueeze(0)
        else:
            y = input

        if self.affine:
            w = self.weight.unsqueeze(-1); b = self.bias.unsqueeze(-1)
            y = (y - b) / (w + self.eps)

        (means, stdev) = state
        y = y * stdev + means

        if input.dim() == self._get_no_batch_dim():
            return y.squeeze(0)
        
        return y


class SNorm(nn.Module):
    '''Input shape is (bs, features, n_nodes, n_timesteps)
    '''
    def __init__(self, channels):
        super(SNorm, self).__init__()
        self.beta = nn.Parameter(t.zeros(channels))
        self.gamma = nn.Parameter(t.ones(channels))

    def forward(self, x):
        x_norm = (x - x.mean(2, keepdims=True)) / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5

        out = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return out


class TNorm(nn.Module):
    '''Input shape is (bs, features, n_nodes, n_timesteps)
    '''
    def __init__(self, num_nodes, channels, track_running_stats=True, momentum=0.1):
        super(TNorm, self).__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(t.zeros(1, channels, num_nodes, 1))
        self.gamma = nn.Parameter(t.ones(1, channels, num_nodes, 1))
        self.register_buffer('running_mean', t.zeros(1, channels, num_nodes, 1))
        self.register_buffer('running_var', t.ones(1, channels, num_nodes, 1))
        self.momentum = momentum

    def forward(self, x):
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdims=True)
            var = x.var((0, 3), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[3] * x.shape[0]
                with t.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((3), keepdims=True)
            var = x.var((3), keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta
        return out


class STNorm(nn.Module):
    '''An adaptor for input of shape (n_nodes, n_timesteps)
    '''

    def __init__(self, num_nodes, use_sn=True, use_tn=True):
        super().__init__()
        channels = 1
        self.funcs = nn.ModuleList([nn.Identity()])
        if use_sn:
            self.funcs.append(SNorm(channels))
        if use_tn:
            self.funcs.append(TNorm(num_nodes, channels,))

        n_funcs = len(self.funcs)
        self.fuse = nn.Linear(n_funcs, 1)
        nn.init.constant_(self.fuse.weight, 1/n_funcs)
        nn.init.zeros_(self.fuse.bias)
    
    def forward(self, x: Tensor):
        n_nodes, n_timesteps = x.shape
        x = x.view(1, 1, n_nodes, n_timesteps)

        outs = []
        for fn in self.funcs:
            outs.append(fn(x))
        out = t.stack(outs, -1)
        out = self.fuse(out)

        out = out.view(n_nodes, n_timesteps)
        return out