import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing, PointConv, fps, radius, radius_graph, knn, knn_graph, knn_interpolate, inits
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

from torch_geometric.nn.inits import zeros, glorot, reset


class DiscreteCRFConv(nn.Module):
    def __init__(self,
                 n_channels,    # class num L
                 e_channels,    # feature dimension D
                 hidden_channels=64,   # hidden space dimension H
                 num_kernels=5,    # number gaussian
                 radius=0.2,
                 kernel_size=32,  # knn neighbors
                 steps=5):      # mean field steps
        super(DiscreteCRFConv, self).__init__()
        self.n_channels = n_channels
        self.e_channels = e_channels
        self.hidden_channels = hidden_channels
        self.radius = radius
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.steps = steps

        self.F = nn.Parameter(torch.Tensor(self.num_kernels, self.e_channels, self.hidden_channels))    # [K, D, H]
        self.W = nn.Parameter(torch.Tensor(self.num_kernels, 1))    # [K, 1]
        self.C = nn.Parameter(torch.Tensor(self.n_channels, self.n_channels))   # [L, L]

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.F)
        nn.init.constant_(self.W, 1/self.num_kernels)
        nn.init.eye_(self.C)

    def forward(self, pos, p, f=None, batch=None):
        N = pos.shape[0]
        # build graph
        # col, row = knn_graph(pos, k=self.kernel_size, batch=batch)
        col, row = radius_graph(pos, r=self.radius, batch=batch, max_num_neighbors=self.kernel_size)

        u = -torch.log(p)   # unary

        # compute weights
        f = f.unsqueeze(0).repeat(self.num_kernels, 1, 1)
        f = torch.bmm(f, self.F)    # [K, N, H]
        f = f.permute((1, 0, 2))    # [N, K, H]
        f = f[col] - f[row]     # [E, K, H]
        w = torch.exp(-torch.sum(f**2, dim=-1))   # [E, K]
        w = torch.mm(w, self.W)     # [E, 1]

        # mean field steps
        q = p   # initialize q [N, L]
        for _ in range(self.steps):
            q = scatter_add(q[col] * w, row, dim=0, dim_size=N)     # message passing [N, L]
            q = torch.mm(q, self.C)     # compatibility transformation [N, L]
            q = torch.softmax(-u-q, dim=-1)

        return q

