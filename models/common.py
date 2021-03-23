import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import fps, radius, knn
from torch_scatter import scatter, scatter_max
from torch_points3d.core.common_modules import FastBatchNorm1d


def fps_pooling(pos, x, edge_attr, batch=None, k=16, r=0.5, reduce='sum'):
    assert reduce in ['max', 'mean', 'add', 'sum']
    idx = fps(pos, batch, ratio=r)
    i, j = knn(pos, pos[idx], k, batch, batch[idx])
    x = scatter(x[j], i, dim=0, reduce=reduce)
    pos, edge_attr, batch = pos[idx], edge_attr[idx], batch[idx]
    return x, pos, edge_attr, batch


def fps_max_pooling(pos, x, batch=None, k=16, r=0.5):
    idx = fps(pos, batch, ratio=r)
    i, j = knn(pos, pos[idx], k, batch, batch[idx])
    x = scatter(x[j], i, dim=0, reduce='max')
    pos, batch = pos[idx], batch[idx]
    return x, pos, batch


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, activation=None):
        super(MLP, self).__init__()
        bias = False if bn else True
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        self.bn = FastBatchNorm1d(out_channels) if bn else None
        self.activation = activation

    def forward(self, x, *args, **kwargs):
        x = self.lin(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv1x1(nn.Module):
    """
    Conv -> BN -> ReLU
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 transpose=False,
                 padding_mode='zeros',
                 bn=False,
                 activation=None):
        super(SharedMLP, self).__init__()

        bias = False if bn else True
        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.conv = conv_fn(in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                            bias=bias,
                            padding_mode=padding_mode)

        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        if activation is not None:
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.activation = None

    def forward(self, x):
        """
        :param x: [B, C, N, K]
        :return: [B, C, N, K]
        """
        x = x.transpose(0, -1)
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        x = x.transpose(0, -1)

        return x


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
