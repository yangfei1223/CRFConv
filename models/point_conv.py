# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing, fps, radius, radius_graph, knn, knn_graph
from torch_geometric.utils import softmax, remove_self_loops, add_self_loops
from torch_scatter import scatter, scatter_add, scatter_max
from .common import Base
from .continuous_crf_conv import GuideGaussianCRFConv as GCRFConv
from .discrete_crf_conv import DiscreteCRFConv


class DepthwiseSeparablePointConv(MessagePassing):
    # Depth-wise separable point convolution, can be symmetric or bipartite graph.
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparablePointConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = out_channels // 4

        self.mlp1 = nn.Sequential(
            nn.Linear(3, self.hidden_channels, bias=False),
            nn.BatchNorm1d(self.hidden_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            nn.BatchNorm1d(self.hidden_channels)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_channels, bias=False),
            nn.BatchNorm1d(self.hidden_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(self.hidden_channels, self.out_channels, bias=False),
            nn.BatchNorm1d(self.out_channels)
        )
        if self.in_channels != self.out_channels:
            self.mlp4 = nn.Sequential(
                nn.Linear(self.in_channels, self.out_channels),
                nn.BatchNorm1d(self.out_channels)
            )

    def forward(self, x, pos, edge_index):
        # Add self-loops for symmetric adjacencies.
        if torch.is_tensor(pos):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=pos.size(0))

        residual = x
        if not torch.is_tensor(pos):
            col, row = edge_index
            # residual = scatter_max(residual[col], row)
            residual, _ = scatter_max(residual[col], row, dim=0)
        if self.in_channels != self.out_channels:
            residual = self.mlp4(residual)

        x = self.mlp2(x)
        x = self.propagate(edge_index, x=x, pos=pos)
        x = self.mlp3(x)
        return F.leaky_relu(x+residual)

    def message(self, x_j, pos_i, pos_j):
        # x_j [E, C_in]
        weights = self.mlp1(pos_i-pos_j)    # [E, C_in]
        msg = weights*x_j
        return msg  # [E, C_in]


class Baseline(nn.Module):
    def __init__(self, in_channels,
                 method='radius',
                 ratio=None,
                 radius=None,
                 kernel_size=16,
                 dilation=None):
        super(Baseline, self).__init__()
        assert method in ['radius', 'knn']
        self.method = method
        self.ratio = ratio
        self.radius = radius
        self.kernel_size = kernel_size
        self.dilation = dilation

        # encoder
        self.conv1_1 = DSPointConv(in_channels, 32)
        self.conv1_2 = DSPointConv(32, 32)

        self.conv2_1 = DSPointConv(32, 64)
        self.conv2_2 = DSPointConv(64, 64)

        self.conv3_1 = DSPointConv(64, 128)
        self.conv3_2 = DSPointConv(128, 128)

        self.conv4_1 = DSPointConv(128, 256)
        self.conv4_2 = DSPointConv(256, 256)

        self.conv5_1 = DSPointConv(256, 512)
        self.conv5_2 = DSPointConv(512, 512)

        # decoder
        self.lin4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True)
        )

        self.fusion3 = nn.Sequential(
            nn.Linear(256+256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.lin3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True)
        )

        self.fusion2 = nn.Sequential(
            nn.Linear(128+128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.lin2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True)
        )

        self.fusion1 = nn.Sequential(
            nn.Linear(64+64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.lin1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True)
        )

    def build_graph(self,  pos, batch, method='radius', r=0.1, k=16, dilation=1, loop=True):
        '''
        Build knn graph or radius graph
        :param pos: (torch.Tensor) The position of the points
        :param batch: (torch.Tensor) The batch index for each point
        :param method: (str) Specify which method to use, ['radius', 'knn']
        :param r: (float) If 'radius' is adopted, the radius of the support domain
        :param k: (int) IF 'knn' is adopted, the number of neighbors
        :param dilation: (int) If 'knn' is adopted, the dilation rate
        :return: (torch.Tensor) The edge index, position and batch of the sampled points
        '''
        assert method in ['radius', 'knn']
        if method == 'radius':
            edge_index = radius_graph(pos, r, batch, loop=loop, max_num_neighbors=k)
        if method == 'knn':
            edge_index = knn_graph(pos, k*dilation, batch, loop=loop)
            if dilation > 1:
                n = pos.shape[0]
                index = torch.randint(
                    k * dilation, (n, k), dtype=torch.long, device=edge_index.device)
                arange = torch.arange(n, dtype=torch.long, device=edge_index.device)
                arange = arange * (k * dilation)
                index = (index + arange.view(-1, 1)).view(-1)
                edge_index = edge_index[:, index]

        return edge_index

    def build_bipartite_graph(self, pos, batch, ratio, method='radius', r=0.1, k=32, dilation=1):
        '''
        Build a bipartite graph given a pos vector
        :param pos: (torch.Tensor) The position of the points
        :param batch: (torch.Tensor) The batch index for each point
        :param ratio: (float) The sample ratio
        :param method: (str) Specify which method to use, ['radius', 'knn']
        :param r: (float) If 'radius' is adopted, the radius of the support domain
        :param k: (int) IF 'knn' is adopted, the number of neighbors
        :param dilation: (int) If 'knn' is adopted, the dilation rate
        :return: (torch.Tensor) The edge index, position and batch of the sampled points
        '''
        assert method in ['radius', 'knn']
        idx = fps(pos, batch, ratio=ratio)
        if method == 'radius':
            row, col = radius(pos, pos[idx], r, batch, batch[idx], max_num_neighbors=k)
        if method == 'knn':
            row, col = knn(pos, pos[idx], k*dilation, batch, batch[idx])
            if dilation > 1:
                n = idx.shape[0]
                index = torch.randint(
                    k * dilation, (n, k), dtype=torch.long, device=row.device)
                arange = torch.arange(n, dtype=torch.long, device=row.device)
                arange = arange * (k * dilation)
                index = (index + arange.view(-1, 1)).view(-1)
                row, col = row[index], col[index]

        edge_index = torch.stack([col, row], dim=0)
        return edge_index, pos[idx], batch[idx]

    def forward(self, x, pos, batch):
        # conv1
        edge_index = self.build_graph(pos, batch,
                                      method=self.method,
                                      r=self.radius[0],
                                      k=self.kernel_size[0],
                                      dilation=self.dilation[0])
        x = self.conv1_1(x, pos, edge_index)
        x = self.conv1_2(x, pos, edge_index)

        # conv2
        edge_index, pos1, batch1 = self.build_bipartite_graph(pos, batch,
                                                              self.ratio[0],
                                                              method=self.method,
                                                              r=self.radius[0],
                                                              k=self.kernel_size[0],
                                                              dilation=self.dilation[0])
        x1 = self.conv2_1(x, (pos, pos1), edge_index)
        edge_index = self.build_graph(pos1, batch1,
                                      method=self.method,
                                      r=self.radius[1],
                                      k=self.kernel_size[1],
                                      dilation=self.dilation[1])
        x1 = self.conv2_2(x1, pos1, edge_index)

        # conv3
        edge_index, pos2, batch2 = self.build_bipartite_graph(pos1, batch1,
                                                              self.ratio[1],
                                                              method=self.method,
                                                              r=self.radius[1],
                                                              k=self.kernel_size[1],
                                                              dilation=self.dilation[1])
        x2 = self.conv3_1(x1, (pos1, pos2), edge_index)
        edge_index = self.build_graph(pos2, batch2,
                                      method=self.method,
                                      r=self.radius[2],
                                      k=self.kernel_size[2],
                                      dilation=self.dilation[2])
        x2 = self.conv3_2(x2, pos2, edge_index)

        # conv4
        edge_index, pos3, batch3 = self.build_bipartite_graph(pos2, batch2,
                                                              self.ratio[2],
                                                              method=self.method,
                                                              r=self.radius[2],
                                                              k=self.kernel_size[2],
                                                              dilation=self.dilation[2])
        x3 = self.conv4_1(x2, (pos2, pos3), edge_index)
        edge_index = self.build_graph(pos3, batch3,
                                      method=self.method,
                                      r=self.radius[3],
                                      k=self.kernel_size[3],
                                      dilation=self.dilation[3])
        x3 = self.conv4_2(x3, pos3, edge_index)

        # conv4
        edge_index, pos4, batch4 = self.build_bipartite_graph(pos3, batch3,
                                                              self.ratio[3],
                                                              method=self.method,
                                                              r=self.radius[3],
                                                              k=self.kernel_size[3],
                                                              dilation=self.dilation[3])
        x4 = self.conv5_1(x3, (pos3, pos4), edge_index)
        edge_index = self.build_graph(pos4, batch4,
                                      method=self.method,
                                      r=self.radius[4],
                                      k=self.kernel_size[4],
                                      dilation=self.dilation[4])
        x4 = self.conv5_2(x4, pos4, edge_index)

        x_ = knn_interpolate(x4, pos4, pos3, batch4, batch3, k=3)
        x_ = self.lin4(x_)

        x_ = self.fusion3(torch.cat([x_, x3], dim=1))
        x_ = knn_interpolate(x_, pos3, pos2, batch3, batch2, k=3)
        x_ = self.lin3(x_)

        x_ = self.fusion2(torch.cat([x_, x2], dim=1))
        x_ = knn_interpolate(x_, pos2, pos1, batch2, batch1, k=3)
        x_ = self.lin2(x_)

        x_ = self.fusion1(torch.cat([x_, x1], dim=1))
        x_ = knn_interpolate(x_, pos1, pos, batch1, batch, k=3)
        x_ = self.lin1(x_)

        return torch.cat([x_, x], dim=1)


class PointConvGassuianCRFNet(nn.Module):
    def __init__(self, in_channels,
                 method='radius',
                 ratio=None,
                 radius=None,
                 kernel_size=16,
                 dilation=None,
                 steps=1):
        super(PointConvGassuianCRFNet, self).__init__()
        assert method in ['radius', 'knn']
        self.method = method
        self.ratio = ratio
        self.radius = radius
        self.kernel_size = kernel_size
        self.dilation = dilation

        # encoder
        self.conv1_1 = DSPointConv(in_channels, 32)
        self.conv1_2 = DSPointConv(32, 32)

        self.conv2_1 = DSPointConv(32, 64)
        self.conv2_2 = DSPointConv(64, 64)

        self.conv3_1 = DSPointConv(64, 128)
        self.conv3_2 = DSPointConv(128, 128)

        self.conv4_1 = DSPointConv(128, 256)
        self.conv4_2 = DSPointConv(256, 256)

        self.conv5_1 = DSPointConv(256, 512)
        self.conv5_2 = DSPointConv(512, 512)

        # decoder
        self.deconv4 = GCRFConv(512, 256, radius=self.radius[3], kernel_size=self.kernel_size[3], steps=steps)

        self.fusion3 = nn.Sequential(
            nn.Linear(256+256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.deconv3 = GCRFConv(256, 128, radius=self.radius[2], kernel_size=self.kernel_size[2], steps=steps)

        self.fusion2 = nn.Sequential(
            nn.Linear(128+128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.deconv2 = GCRFConv(128, 64, radius=self.radius[1], kernel_size=self.kernel_size[1], steps=steps)

        self.fusion1 = nn.Sequential(
            nn.Linear(64+64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.deconv1 = GCRFConv(64, 32, radius=self.radius[0], kernel_size=self.kernel_size[0], steps=steps)

    def build_graph(self,  pos, batch, method='radius', r=0.1, k=16, dilation=1, loop=True):
        '''
        Build knn graph or radius graph
        :param pos: (torch.Tensor) The position of the points
        :param batch: (torch.Tensor) The batch index for each point
        :param method: (str) Specify which method to use, ['radius', 'knn']
        :param r: (float) If 'radius' is adopted, the radius of the support domain
        :param k: (int) IF 'knn' is adopted, the number of neighbors
        :param dilation: (int) If 'knn' is adopted, the dilation rate
        :return: (torch.Tensor) The edge index, position and batch of the sampled points
        '''
        assert method in ['radius', 'knn']
        if method == 'radius':
            edge_index = radius_graph(pos, r, batch, loop=loop, max_num_neighbors=k)
        if method == 'knn':
            edge_index = knn_graph(pos, k*dilation, batch, loop=loop)
            if dilation > 1:
                n = pos.shape[0]
                index = torch.randint(
                    k * dilation, (n, k), dtype=torch.long, device=edge_index.device)
                arange = torch.arange(n, dtype=torch.long, device=edge_index.device)
                arange = arange * (k * dilation)
                index = (index + arange.view(-1, 1)).view(-1)
                edge_index = edge_index[:, index]

        return edge_index

    def build_bipartite_graph(self, pos, batch, ratio, method='radius', r=0.1, k=32, dilation=1):
        '''
        Build a bipartite graph given a pos vector
        :param pos: (torch.Tensor) The position of the points
        :param batch: (torch.Tensor) The batch index for each point
        :param ratio: (float) The sample ratio
        :param method: (str) Specify which method to use, ['radius', 'knn']
        :param r: (float) If 'radius' is adopted, the radius of the support domain
        :param k: (int) IF 'knn' is adopted, the number of neighbors
        :param dilation: (int) If 'knn' is adopted, the dilation rate
        :return: (torch.Tensor) The edge index, position and batch of the sampled points
        '''
        assert method in ['radius', 'knn']
        idx = fps(pos, batch, ratio=ratio)
        if method == 'radius':
            row, col = radius(pos, pos[idx], r, batch, batch[idx], max_num_neighbors=k)
        if method == 'knn':
            row, col = knn(pos, pos[idx], k*dilation, batch, batch[idx])
            if dilation > 1:
                n = idx.shape[0]
                index = torch.randint(
                    k * dilation, (n, k), dtype=torch.long, device=row.device)
                arange = torch.arange(n, dtype=torch.long, device=row.device)
                arange = arange * (k * dilation)
                index = (index + arange.view(-1, 1)).view(-1)
                row, col = row[index], col[index]

        edge_index = torch.stack([col, row], dim=0)
        return edge_index, pos[idx], batch[idx]

    def forward(self, x, pos, batch):
        # conv1
        edge_index = self.build_graph(pos, batch,
                                      method=self.method,
                                      r=self.radius[0],
                                      k=self.kernel_size[0],
                                      dilation=self.dilation[0])
        x = self.conv1_1(x, pos, edge_index)
        x = self.conv1_2(x, pos, edge_index)

        # conv2
        edge_index, pos1, batch1 = self.build_bipartite_graph(pos, batch,
                                                              self.ratio[0],
                                                              method=self.method,
                                                              r=self.radius[0],
                                                              k=self.kernel_size[0],
                                                              dilation=self.dilation[0])
        x1 = self.conv2_1(x, (pos, pos1), edge_index)
        edge_index = self.build_graph(pos1, batch1,
                                      method=self.method,
                                      r=self.radius[1],
                                      k=self.kernel_size[1],
                                      dilation=self.dilation[1])
        x1 = self.conv2_2(x1, pos1, edge_index)

        # conv3
        edge_index, pos2, batch2 = self.build_bipartite_graph(pos1, batch1,
                                                              self.ratio[1],
                                                              method=self.method,
                                                              r=self.radius[1],
                                                              k=self.kernel_size[1],
                                                              dilation=self.dilation[1])
        x2 = self.conv3_1(x1, (pos1, pos2), edge_index)
        edge_index = self.build_graph(pos2, batch2,
                                      method=self.method,
                                      r=self.radius[2],
                                      k=self.kernel_size[2],
                                      dilation=self.dilation[2])
        x2 = self.conv3_2(x2, pos2, edge_index)

        # conv4
        edge_index, pos3, batch3 = self.build_bipartite_graph(pos2, batch2,
                                                              self.ratio[2],
                                                              method=self.method,
                                                              r=self.radius[2],
                                                              k=self.kernel_size[2],
                                                              dilation=self.dilation[2])
        x3 = self.conv4_1(x2, (pos2, pos3), edge_index)
        edge_index = self.build_graph(pos3, batch3,
                                      method=self.method,
                                      r=self.radius[3],
                                      k=self.kernel_size[3],
                                      dilation=self.dilation[3])
        x3 = self.conv4_2(x3, pos3, edge_index)

        # conv4
        edge_index, pos4, batch4 = self.build_bipartite_graph(pos3, batch3,
                                                              self.ratio[3],
                                                              method=self.method,
                                                              r=self.radius[3],
                                                              k=self.kernel_size[3],
                                                              dilation=self.dilation[3])
        x4 = self.conv5_1(x3, (pos3, pos4), edge_index)
        edge_index = self.build_graph(pos4, batch4,
                                      method=self.method,
                                      r=self.radius[4],
                                      k=self.kernel_size[4],
                                      dilation=self.dilation[4])
        x4 = self.conv5_2(x4, pos4, edge_index)

        x_ = knn_interpolate(x4, pos4, pos3, batch4, batch3, k=3)
        x_ = self.deconv4(x_, x3, pos3, batch3)

        x_ = self.fusion3(torch.cat([x_, x3], dim=1))
        x_ = knn_interpolate(x_, pos3, pos2, batch3, batch2, k=3)
        x_ = self.deconv3(x_, x2, pos2, batch2)

        x_ = self.fusion2(torch.cat([x_, x2], dim=1))
        x_ = knn_interpolate(x_, pos2, pos1, batch2, batch1, k=3)
        x_ = self.deconv2(x_, x1, pos1, batch1)

        x_ = self.fusion1(torch.cat([x_, x1], dim=1))
        x_ = knn_interpolate(x_, pos1, pos, batch1, batch, k=3)
        x_ = self.deconv1(x_, x, pos, batch)

        return torch.cat([x_, x], dim=1)


'''
Parts Segmentation
'''


class CRFSegNet_Part(nn.Module):
    '''
    The max and min point number are around 2900 and 530.
    '''
    def __init__(self, in_channels, n_classes=2, steps=1):
        super(CRFSegNet_Part, self).__init__()
        self.feature = PointConvGassuianCRFNet(in_channels,
                                               method='knn',
                                               ratio=[0.25, 0.5, 0.5, 0.5],
                                               radius=[0.2, 0.4, 0.6, 0.8, 1.0],
                                               kernel_size=[32, 16, 8, 8, 8],
                                               dilation=[1, 2, 4, 2, 1],
                                               steps=steps)
        self.classifier = nn.Sequential(
            nn.Linear(64+64+16, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes)
        )

    def forward(self, data):
        pos, norm, category, batch = data.pos, data.norm, data.category, data.batch
        c = F.one_hot(category[batch], num_classes=16).float()
        x = self.feature(x=torch.cat([pos, norm], dim=1), pos=pos, batch=batch)
        x = self.classifier(torch.cat([x, c], dim=1))
        return F.log_softmax(x, dim=-1)


'''
Semantic Segmentation
'''


class BaselineSegNet(nn.Module):
    def __init__(self, in_channels, n_classes=2):
        super(BaselineSegNet, self).__init__()
        self.feature = Baseline(in_channels,
                                method='knn',
                                ratio=[0.25, 0.25, 0.25, 0.25],
                                radius=[0.2, 0.2, 0.2, 0.2, 0.2],
                                kernel_size=[16, 16, 16, 16, 16],
                                dilation=[1, 1, 1, 1, 1])
        self.classifier = nn.Sequential(
            nn.Linear(32+32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes)
        )

    def forward(self, data):
        pos, x, batch = data.pos, data.x, data.batch
        x = self.feature(x=x, pos=pos, batch=batch)
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)


class BaselineDiscreteCRFSegNet(nn.Module):
    def __init__(self, in_channels, n_classes=2, steps=1):
        super(BaselineDiscreteCRFSegNet, self).__init__()
        self.feature = Baseline(in_channels,
                                method='knn',
                                ratio=[0.25, 0.375, 0.375, 0.375],
                                radius=[0.2, 0.2, 0.2, 0.2, 0.2],
                                kernel_size=[32, 16, 16, 16, 16],
                                dilation=[1, 2, 4, 4, 2])
        self.classifier = nn.Sequential(
            nn.Linear(64+64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes)
        )
        self.crf = DiscreteCRFConv(n_classes, in_channels, radius=0.2, kernel_size=32, steps=steps)

    def forward(self, data):
        pos, x, batch = data.pos, data.x, data.batch
        p = self.feature(x=x, pos=pos, batch=batch)
        p = self.classifier(p)
        p = torch.softmax(p, dim=-1)
        q = self.crf(pos, p, f=x, batch=batch)
        return torch.log(p), torch.log(q)


class CRFSegNet(nn.Module):
    def __init__(self, in_channels, n_classes=2, steps=1):
        super(CRFSegNet, self).__init__()
        self.feature = PointConvGassuianCRFNet(in_channels,
                                               # method='radius',
                                               method='knn',
                                               ratio=[0.25, 0.25, 0.25, 0.25],
                                               radius=[0.2, 0.2, 0.2, 0.2, 0.2],
                                               kernel_size=[16, 16, 16, 16, 16],
                                               dilation=[1, 1, 1, 1, 1],
                                               steps=steps)
        self.classifier = nn.Sequential(
            nn.Linear(32+32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes)
        )

    def forward(self, data):
        pos, x, batch = data.pos, data.x, data.batch
        x = self.feature(x=x, pos=pos, batch=batch)
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)


class DualCRFSegNet(nn.Module):
    def __init__(self, in_channels, n_classes=2, steps=1):
        super(DualCRFSegNet, self).__init__()
        self.feature = PointConvGassuianCRFNet(in_channels,
                                               # method='radius',
                                               method='knn',
                                               ratio=[0.25, 0.375, 0.375, 0.375],
                                               radius=[0.2, 0.2, 0.2, 0.2, 0.2],
                                               kernel_size=[32, 16, 16, 16, 16],
                                               dilation=[1, 2, 4, 4, 2],
                                               steps=steps)
        self.classifier = nn.Sequential(
            nn.Linear(64+64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes)
        )
        self.crf = DiscreteCRFConv(n_classes, in_channels, radius=0.2,  kernel_size=32, steps=steps)

    def forward(self, data):
        pos, x, batch = data.pos, data.x, data.batch
        p = self.feature(x=x, pos=pos, batch=batch)
        p = self.classifier(p)
        p = torch.softmax(p, dim=-1)
        q = self.crf(pos, p, f=x, batch=batch)
        return torch.log(p), torch.log(q)

