import torch
import torch.nn as nn


class EvoNormS0(nn.Module):
    """
    Liu H, Brock A, Simonyan K, et al.
    Evolving Normalization-Activation Layers[J]. arXiv preprint arXiv:2004.02967, 2020.
    """

    def __init__(self, num_channels, num_groups=32, affine=True, nonlinearity=True, eps=1e-5):
        super(EvoNormS0, self).__init__()
        self.eps = eps
        self.num_groups = num_groups
        self.affine = affine
        self.nonlinearity = nonlinearity
        if nonlinearity:
            self.v = nn.Parameter(torch.Tensor(num_channels))
            nn.init.ones_(self.v)
        else:
            self.register_parameter('v', None)
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def _affine(self, x):
        if self.affine:
            return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        else:
            return x

    def forward(self, x):
        if self.nonlinearity:
            N, C, H, W = x.shape
            assert C % self.num_groups == 0

            num = x * torch.sigmoid(self.v.view(1, -1, 1, 1) * x)

            x = torch.reshape(x, (N, self.num_groups, C // self.num_groups, H, W))
            var = torch.var(x, dim=(2, 3, 4), keepdim=True)
            std = torch.sqrt(var + self.eps)

            num = num.reshape(N, self.num_groups, C // self.num_groups, H, W) / std
            num = num.reshape(N, C, H, W)
            return self._affine(num)
        else:
            return self._affine(x)
