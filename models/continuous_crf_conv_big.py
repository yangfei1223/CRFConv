import torch
import torch.nn.functional as F
import torch.nn as nn
from .common import MLP


class ContinuousGaussianCRFConv(nn.Module):
    def __init__(self,
                 unary_channels,
                 pairwise_channels,
                 out_channels=None,
                 steps=1):
        super(ContinuousGaussianCRFConv, self).__init__()
        self.unary_channels = unary_channels
        self.pairwise_channels = pairwise_channels
        self.out_channels = out_channels if out_channels is not None else pairwise_channels
        self.hidden_channels = self.out_channels // 4
        self.steps = steps

        self.unary_nn = nn.Sequential(
            MLP(self.unary_channels, self.hidden_channels, activation=nn.LeakyReLU(negative_slope=0.1)),
            MLP(self.hidden_channels, self.hidden_channels, activation=None)
        )
        self.pairwise_nn = nn.Sequential(
            MLP(self.pairwise_channels, self.hidden_channels, activation=nn.LeakyReLU(negative_slope=0.1)),
            MLP(self.hidden_channels, self.hidden_channels, activation=None)
        )
        self.out_nn = MLP(self.hidden_channels, self.out_channels, activation=nn.LeakyReLU(negative_slope=0.1))
        self.fusion_nn = MLP(self.out_channels * 2, self.out_channels, activation=nn.LeakyReLU(negative_slope=0.1))

        self.c = nn.Parameter(torch.Tensor(self.hidden_channels, self.hidden_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.eye_(self.c)

    @staticmethod
    def _gather_neighbors(x, neighbor_idx):
        B, F, K = x.shape[0], x.shape[-1], neighbor_idx.shape[-1]
        neighbor_idx = neighbor_idx.reshape(B, -1, 1).repeat(1, 1, F)
        neighbors = x.gather(dim=1, index=neighbor_idx).reshape(B, -1, K, F)
        return neighbors.squeeze()

    @staticmethod
    def _remove_self_loop(neighbor_idx):
        return neighbor_idx[:, :, 1:]       # index 0 is the nearest point i.e. self-loop

    def _compute_similarity(self, x, neighbor_idx):
        neighbors = self._gather_neighbors(x, neighbor_idx)      # [B, N, K, F]
        s = x.unsqueeze(2) - neighbors        # [B, N, K, F]
        s = s.pow(2).sum(dim=-1, keepdim=True)
        s = (-s).softmax(dim=2)       # [B, N, K, 1]
        return s

    def forward(self, unary, pairwise, up_idx, neighbor_idx):
        neighbor_idx = self._remove_self_loop(neighbor_idx)
        x = self.unary_nn(unary)
        y = self.pairwise_nn(pairwise)
        x = self._gather_neighbors(x, up_idx)
        s = self._compute_similarity(y, neighbor_idx)

        # initialize
        z = x
        I = torch.eye(self.hidden_channels, dtype=torch.float, device=x.device)
        C = torch.mm(self.c.t(), self.c)        # c^t * c to ensure positive-define
        # mean-field steps
        for _ in range(self.steps):
            x = self._gather_neighbors(x, neighbor_idx)      # [B, N, K, F], gather neighbors
            x = (s * x).sum(dim=2)                          # [B, N, F], weight summation (message passing)
            x = z + x.matmul(C)                             # [B, N, F], compatibility transform
            x = x.matmul((I + C).inverse())                 # [B, N, F], update (normalization)

        x = self.out_nn(x)

        x = self.fusion_nn(torch.cat([x, pairwise], dim=-1))

        return x
