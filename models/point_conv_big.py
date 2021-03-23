import torch
import torch.nn.functional as F
import torch.nn as nn
from .common import MLP, Base
from .continuous_crf_conv_big import ContinuousGaussianCRFConv as CRFConv


class PointConv(nn.Module):
    """
    Re-implementation of original used depth-wise separable point conv in paper,
    the new version will be tested on large scale dataset
    """
    def __init__(self, d_model):
        super(PointConv, self).__init__()
        # self.weight_nn = nn.Sequential(
        #     MLP(3, d_model, bn=False, activation=nn.LeakyReLU(negative_slope=0.1)),
        #     MLP(d_model, d_model, bn=False, activation=None),
        #     nn.Softmax(dim=2)
        # )
        self.weight_nn = nn.Sequential(
            MLP(3, d_model, activation=nn.LeakyReLU(negative_slope=0.1)),
            MLP(d_model, d_model, activation=None)
        )

    @staticmethod
    def gather_neighbors(x, idx):
        """
        :param x: [B, N, F]
        :param idx: [B, N', K]
        :return: [B, N', K, F]
        """
        B, F, K = x.shape[0], x.shape[-1], idx.shape[-1]
        idx = idx.reshape(B, -1, 1).repeat(1, 1, F)
        x = x.gather(dim=1, index=idx).reshape(B, -1, K, F)
        return x

    def _compute_weights(self, pos, neighbors):
        B, N, D, K = pos.shape[0], pos.shape[1], pos.shape[2], neighbors.shape[2]
        pos = pos.reshape(B, N, 1, D).repeat(1, 1, K, 1)  # [B, N, K, D]
        rel_pos = pos - neighbors  # [B, N, K, D]
        # rel_dist = rel_pos.norm(dim=-1, keepdim=True)  # [B, N, K, 1]
        # pos_embedding = torch.cat([rel_dist, rel_pos, pos, neighbors], dim=-1)  # [B, N, K, 10]
        weights = self.weight_nn(rel_pos.reshape(B, N * K, D)).reshape(B, N, K, -1)
        return weights

    def forward(self, x, pos, neighbor_idx):
        if torch.is_tensor(pos):
            neighbors = self.gather_neighbors(pos, neighbor_idx)
            w = self._compute_weights(pos, neighbors)
        else:
            pos, sub_pos = pos
            neighbors = self.gather_neighbors(pos, neighbor_idx)
            w = self._compute_weights(sub_pos, neighbors)

        x = self.gather_neighbors(x, neighbor_idx)  # [B, -1, K, F]
        x = (w * x).sum(dim=2)

        return x


class ResNetBBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBBlock, self).__init__()
        hidden_channels = out_channels // 4
        self.lin_in = MLP(in_channels, hidden_channels, activation=nn.LeakyReLU(negative_slope=0.1))
        self.lin_out = MLP(hidden_channels, out_channels, activation=None)
        if in_channels != out_channels:
            self.shortcut = MLP(in_channels, out_channels, activation=None)
        else:
            self.shortcut = nn.Identity()

        self.point_conv = PointConv(hidden_channels)

    @staticmethod
    def max_pooling(x, idx):
        x = PointConv.gather_neighbors(x, idx)
        return x.max(dim=2)[0]

    def forward(self, x, pos, neighbor_idx):
        residual = self.shortcut(x)
        if not torch.is_tensor(pos):
            residual = self.max_pooling(residual, neighbor_idx)

        x = self.lin_in(x)
        x = self.point_conv(x, pos, neighbor_idx)
        x = self.lin_out(x)

        return F.leaky_relu(x + residual)


class Upsampling(nn.Module):
    def __init__(self, down_channels, up_channels, out_channels):
        super(Upsampling, self).__init__()
        self.lin = MLP(down_channels, up_channels, activation=nn.LeakyReLU(negative_slope=0.1))
        self.fusion = MLP(up_channels * 2, out_channels, activation=nn.LeakyReLU(negative_slope=0.1))

    @staticmethod
    def upsampling(x, idx):
        idx = idx.repeat(1, 1, x.shape[-1])     # [B, N, F]
        x = x.gather(dim=1, index=idx)          # [B, N, F]
        return x

    def forward(self, x_down, x_up, up_idx, neighbor_idx=None):
        x_down = self.upsampling(x_down, up_idx)
        x_down = self.lin(x_down)
        x_fusion = self.fusion(torch.cat([x_up, x_down], dim=-1))
        return x_fusion


class PointConvResNet(Base):
    def __init__(self, in_channels, n_classes, use_crf=True, steps=1):
        super(PointConvResNet, self).__init__()
        layers = [32, 64, 128, 256, 512]
        self.C = n_classes

        self.conv1_1 = ResNetBBlock(in_channels, layers[0])
        self.conv1_2 = ResNetBBlock(layers[0], layers[0])

        self.conv2_1 = ResNetBBlock(layers[0], layers[1])
        self.conv2_2 = ResNetBBlock(layers[1], layers[1])

        self.conv3_1 = ResNetBBlock(layers[1], layers[2])
        self.conv3_2 = ResNetBBlock(layers[2], layers[2])

        self.conv4_1 = ResNetBBlock(layers[2], layers[3])
        self.conv4_2 = ResNetBBlock(layers[3], layers[3])

        self.conv5_1 = ResNetBBlock(layers[3], layers[4])
        self.conv5_2 = ResNetBBlock(layers[4], layers[4])

        self.deconv4 = CRFConv(layers[4], layers[3], layers[3], steps=steps) if use_crf else Upsampling(layers[4], layers[3], layers[3])
        self.deconv3 = CRFConv(layers[3], layers[2], layers[2], steps=steps) if use_crf else Upsampling(layers[3], layers[2], layers[2])
        self.deconv2 = CRFConv(layers[2], layers[1], layers[1], steps=steps) if use_crf else Upsampling(layers[2], layers[1], layers[1])
        self.deconv1 = CRFConv(layers[1], layers[0], layers[0], steps=steps) if use_crf else Upsampling(layers[1], layers[0], layers[0])

        self.classifier = nn.Sequential(
            MLP(layers[0], layers[0] * 4, activation=nn.LeakyReLU(negative_slope=0.1)),
            nn.Dropout(p=0.5),
            nn.Linear(layers[0] * 4, n_classes)
        )

    def forward(self, data):
        x, multiscale = data.x, data.multiscale

        x1 = self.conv1_1(x, multiscale[0].pos, multiscale[0].neighbor_idx)
        x1 = self.conv1_2(x1, multiscale[0].pos, multiscale[0].neighbor_idx)

        x2 = self.conv2_1(x1, (multiscale[0].pos, multiscale[1].pos), multiscale[0].sub_idx)
        x2 = self.conv2_2(x2, multiscale[1].pos, multiscale[1].neighbor_idx)

        x3 = self.conv3_1(x2, (multiscale[1].pos, multiscale[2].pos), multiscale[1].sub_idx)
        x3 = self.conv3_2(x3, multiscale[2].pos, multiscale[2].neighbor_idx)

        x4 = self.conv4_1(x3, (multiscale[2].pos, multiscale[3].pos), multiscale[2].sub_idx)
        x4 = self.conv4_2(x4, multiscale[3].pos, multiscale[3].neighbor_idx)

        x = self.conv5_1(x4, (multiscale[3].pos, multiscale[4].pos), multiscale[3].sub_idx)
        x = self.conv5_2(x, multiscale[4].pos, multiscale[4].neighbor_idx)

        x = self.deconv4(x, x4, multiscale[3].up_idx, multiscale[3].neighbor_idx)
        x = self.deconv3(x, x3, multiscale[2].up_idx, multiscale[2].neighbor_idx)
        x = self.deconv2(x, x2, multiscale[1].up_idx, multiscale[1].neighbor_idx)
        x = self.deconv1(x, x1, multiscale[0].up_idx, multiscale[0].neighbor_idx)

        x = self.classifier(x)

        return x.reshape(-1, self.C)
