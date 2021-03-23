import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing, fps, radius, radius_graph, knn, knn_graph, GMMConv
from torch_geometric.utils import softmax, remove_self_loops, add_self_loops
from torch_scatter import scatter, scatter_add, scatter_max


class GuideGaussianCRFConv(nn.Module):
    def __init__(self,
                 in_n_channels,
                 in_e_channels,
                 out_channels=None,
                 radius=0.1,
                 kernel_size=32,
                 steps=1):
        super(GuideGaussianCRFConv, self).__init__()
        self.in_n_channels = in_n_channels
        self.in_e_channels = in_e_channels
        self.out_channels = out_channels if out_channels is not None else in_e_channels
        self.radius = radius
        self.kernel_size = kernel_size
        self.steps = steps
        self.unary = nn.Sequential(
            nn.Linear(self.in_n_channels, self.out_channels, bias=False),
            nn.BatchNorm1d(self.out_channels),
            # nn.LeakyReLU(inplace=True),
            # nn.Linear(self.out_channels, self.out_channels, bias=False),
            # nn.BatchNorm1d(self.out_channels),
        )
        self.pairwise = nn.Sequential(
            nn.Linear(self.in_e_channels, self.out_channels, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(self.out_channels, self.out_channels, bias=False),
            # nn.BatchNorm1d(self.out_channels),
            # nn.LeakyReLU(inplace=True)
        )
        # self.U = nn.Parameter(torch.Tensor(self.in_n_channels, self.out_channels))
        # self.P = nn.Parameter(torch.Tensor(self.in_e_channels, self.out_channels))
        self.c = nn.Parameter(torch.Tensor(self.out_channels, self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.uniform_(self.U)
        # nn.init.uniform_(self.P)
        nn.init.eye_(self.c)

    def forward(self, x, y, pos, batch):
        N = pos.shape[0]
        # col, row = knn_graph(pos, self.kernel_size, batch=batch)
        col, row = radius_graph(pos, r=self.radius, batch=batch, max_num_neighbors=self.kernel_size)
        x = self.unary(x)
        y = self.pairwise(y)
        s = torch.sum((y[row]-y[col])**2, dim=1, keepdim=True)
        s = softmax(-s, row, num_nodes=N)

        z = x
        I = torch.eye(self.out_channels, dtype=torch.float, device=x.device)
        C = torch.mm(self.c.t(), self.c)
        # mean field steps
        for _ in range(self.steps):
            x = s * x[col]      # [E, F] message j -> i
            x = scatter_add(x, row, dim=0, dim_size=N)    # [N, F] message passing / aggregate
            x = z + torch.mm(x, C)  # [N, F] compatibility transform
            x = torch.mm(x, (I + C).inverse())   # [N, F] update

        return F.leaky_relu(x)


class ContinuousGaussianCRFConv(nn.Module):
    def __init__(self,
                 unary_channels,
                 pairwise_channels,
                 hidden_channels=None,
                 out_channels=None,
                 steps=1):
        super(ContinuousGaussianCRFConv, self).__init__()
        self.unary_channels = unary_channels
        self.pairwise_channels = pairwise_channels
        self.out_channels = out_channels if out_channels is not None else pairwise_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else self.out_channels // 4
        self.steps = steps
        self.unary_net = nn.Sequential(
            nn.Linear(self.unary_channels, self.hidden_channels, bias=False),
            nn.BatchNorm1d(self.hidden_channels),
        )
        self.pairwise_net = nn.Sequential(
            nn.Linear(self.pairwise_channels, self.hidden_channels, bias=False),
            nn.BatchNorm1d(self.hidden_channels),
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_channels, self.out_channels, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.fusion_net = nn.Sequential(
            nn.Linear(self.out_channels * 2, self.out_channels, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.c = nn.Parameter(torch.Tensor(self.hidden_channels, self.hidden_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.eye_(self.c)

    def forward(self, x, y, pos, edge_index):
        N = pos.shape[0]
        i, j = edge_index
        x = self.unary_net(x)
        s = self.pairwise_net(y)
        s = torch.sum((s[i] - s[j]) ** 2, dim=1, keepdim=True)
        s = softmax(-s, i, num_nodes=N)

        z = x
        I = torch.eye(self.hidden_channels, dtype=torch.float, device=x.device)
        C = torch.mm(self.c.t(), self.c)
        # mean-field steps
        for _ in range(self.steps):
            x = s * x[j]                                # [E, F] message j -> i
            x = scatter_add(x, i, dim=0, dim_size=N)    # [N, F] message passing / aggregate
            x = z + torch.mm(x, C)                      # [N, F] compatibility transform
            x = torch.mm(x, (I + C).inverse())          # [N, F] update

        x = self.mlp(x)
        x = self.fusion_net(torch.cat([x, y], dim=-1))

        return x


