# -*- coding:utf-8 -*-
import random
import os, sys, glob, pickle
from itertools import chain
from utils import read_ply, write_ply
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch_points_kernels.points_cpu as tpcpu
import torch_points_kernels.points_cuda as tpcuda
from torch_geometric.transforms import FixedPoints
from torch_points3d.core.data_transform import *
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.batch import SimpleBatch
from torch_points3d.datasets.multiscale_data import MultiScaleData, MultiScaleBatch
from torch_points3d.datasets.segmentation.s3dis import S3DISOriginalFused, S3DISSphere, S3DISCylinder
from torch_points3d.datasets.segmentation.shapenet import ShapeNetDataset
from utils import cpp_subsampling, nearest_neighbors


CLASS_NAMES = {'ceiling': 0, 'floor': 1, 'wall': 2, 'beam': 3, 'column': 4, 'window': 5, 'door': 6,
               'table': 7, 'chair': 8, 'sofa': 9, 'bookcase': 10, 'board': 11, 'clutter': 12}


class S3DISDataset(InMemoryDataset):
    """
    S3DIS dataset.
    --folder', '-f', help='Path to data folder
    --max_point_num', '-m', help='Max point number of each sample', type=int, default=8192
    --block_size', '-b', help='Block size', type=float, default=1.5
    --grid_size', '-g', help='Grid size', type=float, default=0.03
    --save_ply', '-s', help='Convert .pts to .ply', action='store_true'
    """
    def __init__(self,
                 root,
                 train=True,
                 test_area=5,
                 sample_per_epoch=-1,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None
                 ):
        """Constructor.
        Args:
            root (str): Path of raw dataset.
            area_id (int): which area to test, default: 5
            transform (obj): Data transformer.
            pre_transform (obj): Data transformer.
            pre_filter (obj): Data filter.
        """
        assert test_area in [1, 2, 3, 4, 5, 6]
        self.min_point_num = 100
        self.block_size = 1.0
        self.stride = 0.5
        self.padding = 0.1
        self.proportion = 0.02
        self.sample_per_epoch = sample_per_epoch
        super(S3DISDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.filelist = []
        if train:
            for i, path in enumerate(self.processed_paths):
                if (i + 1) != test_area:
                    filelist = os.listdir(path)
                    filelist = [os.path.join(path, f) for f in filelist]
                    self.filelist += filelist
        else:
            path = self.processed_paths[test_area-1]
            filelist = os.listdir(path)
            filelist = [os.path.join(path, f) for f in filelist]
            self.filelist += filelist

    @property
    def raw_file_names(self):
        return ['Area_1_anno.txt', 'Area_2_anno.txt', 'Area_3_anno.txt',
                'Area_4_anno.txt', 'Area_5_anno.txt', 'Area_6_anno.txt']

    @property
    def processed_file_names(self):
        return ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']

    def __len__(self):
        if self.sample_per_epoch > 0:
            return self.sample_per_epoch
        else:
            return len(self.filelist)

    def _get_random(self):
        idx = np.random.randint(len(self.filelist))
        return torch.load(self.filelist[idx])

    def get(self, idx):
        return self._get_random() if self.sample_per_epoch > 0 else torch.load(self.filelist[idx])

    def download(self):
        pass

    def process_room(self, anno_paths, output_path):
        label_dict = {}
        for room_idx, anno_path in enumerate(anno_paths):
            print('Processing {}...'.format(anno_path))
            points = []
            labels = []
            # Note: there is an extra character in line 180389 of Area_5/hallway_6/Annotations/ceiling_1.txt
            for f in glob.glob(os.path.join(anno_path, '*.txt')):
                print('Collecting {}...'.format(f))
                label = os.path.basename(f).split('_')[0]
                if label not in CLASS_NAMES:
                    label = 'clutter'
                # cls_points = np.loadtxt(f)
                cls_points = pd.read_csv(f, header=None, delim_whitespace=True).values   # pandas read faster than numpy
                cls_labels = np.full((cls_points.shape[0]), CLASS_NAMES[label], dtype=np.int32)
                points.append(cls_points)
                labels.append(cls_labels)
            points = np.concatenate(points, axis=0)
            labels = np.concatenate(labels, axis=0)
            point_num = points.shape[0]
            print('Collect {} points in room {}.'.format(point_num, room_idx))

            label_dict[room_idx] = labels
            point_indices = np.arange(point_num).astype(np.int32)

            xyz, rgb = np.split(points, 2, axis=-1)

            xyz_min = np.amin(xyz, axis=0)
            xyz -= xyz_min      # align to the min point
            limit = np.amax(xyz, axis=0)

            rgb /= 255.
            xyz_room_normalized = xyz / limit

            xbeg_list = []
            ybeg_list = []
            num_block_x = int(np.ceil((limit[0] - self.block_size) / self.stride)) + 1
            num_block_y = int(np.ceil((limit[1] - self.block_size) / self.stride)) + 1
            for i in range(num_block_x):
                for j in range(num_block_y):
                    xbeg_list.append(i * self.stride)
                    ybeg_list.append(j * self.stride)

            print('There is {} blocks in total.'.format(num_block_x*num_block_y))
            # collect points
            block_count = 0
            for xbeg, ybeg in zip(xbeg_list, ybeg_list):
                xcond = (xyz[:, 0] >= (xbeg - self.padding)) & (xyz[:, 0] <= (xbeg + self.block_size + self.padding))
                ycond = (xyz[:, 1] >= (ybeg - self.padding)) & (xyz[:, 1] <= (ybeg + self.block_size + self.padding))
                cond = xcond & ycond
                if np.sum(cond) < self.min_point_num:
                    continue

                block_xyz = xyz[cond]
                maskx = (block_xyz[:, 0] >= xbeg) & (block_xyz[:, 0] <= xbeg + self.block_size)
                masky = (block_xyz[:, 1] >= ybeg) & (block_xyz[:, 1] <= ybeg + self.block_size)
                mask = maskx & masky
                if np.sum(mask) / mask.shape[0] < self.proportion:
                    continue

                pos = torch.from_numpy(block_xyz.astype(np.float32))
                x = torch.from_numpy(np.concatenate((rgb[cond], xyz_room_normalized[cond]), axis=-1).astype(np.float32))
                y = torch.from_numpy(labels[cond].astype(np.long))
                mask = torch.from_numpy(mask.astype(np.int8))
                indices = torch.from_numpy(point_indices[cond].astype(np.long))
                data = Data(pos=pos, x=x, y=y, mask=mask, indices=indices)
                save_path = os.path.join(output_path, 'room_{:02d}_{:06d}.pt'.format(room_idx, block_count))
                print('Saving {} points to "{}".'.format(data.num_nodes, save_path))
                torch.save(data, save_path)
                block_count += 1
            print('Split {} points to {} blocks.'.format(point_num, block_count))
        return label_dict

    def process(self):
        data_dir = 'Stanford3dDataset_v1.2_Aligned_Version'
        for i, path in enumerate(self.raw_paths):
            if os.path.exists(self.processed_paths[i]):
                continue
            os.makedirs(self.processed_paths[i])
            print("Processing Area_{}...".format(i+1))
            anno_paths = [line.rstrip() for line in open(path)]
            anno_paths = [os.path.join(self.raw_dir, data_dir, p) for p in anno_paths]
            label_dict = self.process_room(anno_paths, self.processed_paths[i])
            np.save(os.path.join(self.processed_dir, 'label_area_{}.npy'.format(i+1)), label_dict)


class S3DISRoom(InMemoryDataset):
    data_dir = 'Stanford3dDataset_v1.2_Aligned_Version'

    def __init__(self,
                 root,
                 test_area=5,
                 grid_size=0.04,
                 num_points=65536,
                 sample_per_epoch=100,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None
                 ):
        assert test_area in [1, 2, 3, 4, 5, 6]
        super(S3DISRoom, self).__init__(root, transform, pre_transform, pre_filter)
        self.test_area = 'Area_{}'.format(test_area)
        self.grid_size = grid_size
        self.num_points = num_points
        self.sample_per_epoch = sample_per_epoch
        self.train = train

        self.label_values = np.sort([v for k, v in CLASS_NAMES.items()])        # [0-12]

        self.possibility = []
        self.min_possibility = []
        self.input_trees = []
        self.input_rgb = []
        self.input_labels = []
        self.input_names = []

        if not self.train:
            self.val_proj = []
            self.val_labels = []

        # load processed data
        self._load_processed()

        # random init probability
        for tree in self.input_trees:
            self.possibility += [np.random.randn(tree.data.shape[0]) * 1e-3]
            self.min_possibility += [float(np.min(self.possibility[-1]))]

    @property
    def raw_file_names(self):
        return ['Area_1_anno.txt', 'Area_2_anno.txt', 'Area_3_anno.txt',
                'Area_4_anno.txt', 'Area_5_anno.txt', 'Area_6_anno.txt']

    @property
    def processed_file_names(self):
        return ['original', 'sampled']

    def __len__(self):
        if self.sample_per_epoch > 0:
            return self.sample_per_epoch
        else:
            return len(self.input_trees)

    def get(self, idx):
        return self._get_random()

    def download(self):
        pass

    @staticmethod
    def _grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size, verbose=verbose)

    def process(self):
        # Note: there is an extra character in line 180389 of Area_5/hallway_6/Annotations/ceiling_1.txt
        for path in self.processed_paths:
            os.makedirs(path)
        for i, path in enumerate(self.raw_paths):
            print("Processing Area_{}...".format(i + 1))
            anno_paths = [line.rstrip() for line in open(path)]
            anno_paths = [os.path.join(self.raw_dir, self.data_dir, p) for p in anno_paths]
            for anno_path in anno_paths:
                print('Processing {}...'.format(anno_path))
                elements = anno_path.split('/')
                filename = elements[-3] + '_' + elements[-2]
                data_list = []
                for f in glob.glob(os.path.join(anno_path, '*.txt')):
                    print('Collecting {}...'.format(f))
                    label = os.path.basename(f).split('_')[0]
                    if label not in CLASS_NAMES:
                        label = 'clutter'
                    # cls_points = np.loadtxt(f)
                    cls_points = pd.read_csv(f, header=None, delim_whitespace=True).values  # pandas for faster reading
                    cls_labels = np.full((cls_points.shape[0], 1), CLASS_NAMES[label], dtype=np.int32)
                    data_list.append(np.concatenate([cls_points, cls_labels], axis=1))  # Nx7

                points_labels = np.concatenate(data_list, axis=0)

                xyz_min = np.amin(points_labels, axis=0)[0:3]
                points_labels[:, 0:3] -= xyz_min    # aligned to the minimal point
                xyz = points_labels[:, 0:3].astype(np.float32)
                rgb = points_labels[:, 3:6].astype(np.uint8)
                labels = points_labels[:, 6].astype(np.uint8)

                org_ply_file = os.path.join(self.processed_paths[0], filename + '.ply')
                write_ply(org_ply_file, [xyz, rgb, labels], ['x', 'y', 'z', 'r', 'g', 'b', 'class'])

                # save sub_cloud and KDTree files
                sub_xyz, sub_rgb, sub_labels = self._grid_sub_sampling(xyz, rgb, labels, self.grid_size)
                sub_rgb = sub_rgb / 255.

                sub_ply_file = os.path.join(self.processed_paths[1], filename + '.ply')
                write_ply(sub_ply_file, [sub_xyz, sub_rgb, sub_labels], ['x', 'y', 'z', 'r', 'g', 'b', 'class'])

                search_tree = KDTree(sub_xyz)
                kd_tree_file = os.path.join(self.processed_paths[1], filename + '_KDTree.pkl')
                with open(kd_tree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
                proj_idx = proj_idx.astype(np.int32)
                proj_file = os.path.join(self.processed_paths[1], filename + '_proj.pkl')
                with open(proj_file, 'wb') as f:
                    pickle.dump([proj_idx, labels], f)

    def _load_processed(self):
        for f in glob.glob(os.path.join(self.processed_paths[0], '*.ply')):
            name = f.split('/')[-1][:-4]
            if self.train:
                if self.test_area in name:
                    continue
            else:
                if self.test_area not in name:
                    continue

            kd_tree_file = os.path.join(self.processed_paths[1], '{}_KDTree.pkl'.format(name))
            sub_ply_file = os.path.join(self.processed_paths[1], '{}.ply'.format(name))
            data = read_ply(sub_ply_file)
            sub_rgb = np.vstack((data['r'], data['g'], data['b'])).T
            sub_labels = data['class']
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees += [search_tree]
            self.input_rgb += [sub_rgb]
            self.input_labels += [sub_labels]
            self.input_names += [name]

            if not self.train:
                proj_file = os.path.join(self.processed_paths[1], '{}_proj.pkl'.format(name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]

    def _get_random(self):
        cloud_idx = int(np.argmin(self.min_possibility))
        pick_idx = np.argmin(self.possibility[cloud_idx])
        points = np.array(self.input_trees[cloud_idx].data, copy=False)
        pick_point = points[pick_idx, :].reshape(1, -1)

        noise = np.random.normal(scale=3.5 / 10, size=pick_point.shape)
        pick_point = pick_point + noise.astype(pick_point.dtype)

        if len(points) < self.num_points:
            query_idx = self.input_trees[cloud_idx].query(pick_point, k=len(points))[1][0]
        else:
            query_idx = self.input_trees[cloud_idx].query(pick_point, k=self.num_points)[1][0]

        np.random.shuffle(query_idx)
        query_xyz = points[query_idx] - pick_point
        query_rgb = self.input_rgb[cloud_idx][query_idx]
        query_labels = self.input_labels[cloud_idx][query_idx]

        # update possibility, reduce the posibility of chosen cloud and point
        dists = np.sum(np.square(query_xyz.astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[cloud_idx][query_idx] += delta
        self.min_possibility[cloud_idx] = float(np.min(self.possibility[cloud_idx]))

        pos = torch.from_numpy(query_xyz).to(torch.float32)
        rgb = torch.from_numpy(query_rgb).to(torch.float32)
        labels = torch.from_numpy(query_labels).to(torch.long)
        point_idx = torch.from_numpy(query_idx).to(torch.long)
        cloud_idx = torch.Tensor([cloud_idx]).to(torch.long)
        data = Data(x=torch.cat([pos, rgb], dim=-1), pos=pos, y=labels, point_idx=point_idx, cloud_idx=cloud_idx)

        # upsampled with minimal replacement
        if len(points) < self.num_points:
            data = FixedPoints(self.num_points, replace=False, allow_duplicates=True)(data)

        return data


class S3DISRoomDataset:
    def __init__(self,
                 root,
                 test_area=5,
                 grid_size=0.04,
                 num_points=65536,
                 train_sample_per_epoch=500,
                 test_sample_per_epoch=100,
                 train_transform=None,
                 test_transform=None):
        self.kernel_size = [16, 16, 16, 16, 16]
        self.ratio = [4, 4, 4, 4, 2]

        self.train_set = S3DISRoom(root,
                                   test_area=test_area,
                                   grid_size=grid_size,
                                   num_points=num_points,
                                   sample_per_epoch=train_sample_per_epoch,
                                   train=True,
                                   transform=train_transform)

        self.test_set = S3DISRoom(root,
                                  test_area=test_area,
                                  grid_size=grid_size,
                                  num_points=num_points,
                                  sample_per_epoch=test_sample_per_epoch,
                                  train=False,
                                  transform=test_transform)

    @staticmethod
    def _knn_search(support_points, query_points, k):
        neighbor_idx = nearest_neighbors.knn_batch(support_points, query_points, k, omp=True)
        return torch.from_numpy(neighbor_idx)

    def _multiscale_compute_fn(self,
                               batch,
                               collate_fn=None,
                               precompute_multi_scale=False,
                               num_scales=0,
                               sample_method='random'):
        batch = collate_fn(batch)
        if not precompute_multi_scale:
            return batch
        multiscale = []
        pos = batch.pos     # [B, N, 3]
        for i in range(num_scales):
            neighbor_idx = self._knn_search(pos, pos, self.kernel_size[i])      # [B, N, K]
            sample_num = pos.shape[1] // self.ratio[i]
            if sample_method.lower() == 'random':
                choice = torch.randperm(pos.shape[1])[:sample_num]
                sub_pos = pos[:, choice, :]             # random sampled pos   [B, S, 3]
                sub_idx = neighbor_idx[:, choice, :]    # the pool idx  [B, S, K]
            elif sample_method.lower() == 'fps':
                choice = tpcuda.furthest_point_sampling(pos.cuda(), sample_num).to(torch.long).cpu()
                sub_pos = pos.gather(dim=1, index=choice.unsqueeze(-1).repeat(1, 1, pos.shape[-1]))
                sub_idx = neighbor_idx.gather(dim=1, index=choice.unsqueeze(-1).repeat(1, 1, neighbor_idx.shape[-1]))
            else:
                raise NotImplementedError('Only `random` or `fps` sampling method is implemented!')

            up_idx = self._knn_search(sub_pos, pos, 1)      # [B, N, 1]
            multiscale.append(Data(pos=pos, neighbor_idx=neighbor_idx, sub_idx=sub_idx, up_idx=up_idx))
            pos = sub_pos

        return MultiScaleData(x=batch.x,
                              y=batch.y,
                              point_idx=batch.point_idx,
                              cloud_idx=batch.cloud_idx,
                              multiscale=multiscale)

    def _dataloader(self, dataset, precompute_multi_scale, num_scales, sample_method, **kwargs):
        batch_collate_function = partial(self._multiscale_compute_fn,
                                         collate_fn=SimpleBatch.from_data_list,
                                         precompute_multi_scale=precompute_multi_scale,
                                         num_scales=num_scales,
                                         sample_method=sample_method)
        data_loader_function = partial(DataLoader, collate_fn=batch_collate_function, worker_init_fn=np.random.seed)
        return data_loader_function(dataset, **kwargs)

    def create_dataloader(self,
                          batch_size: int,
                          shuffle: bool,
                          num_workers: int,
                          precompute_multi_scale: bool,
                          num_scales: int,
                          sample_method='random'
                          ):

        self.train_loader = self._dataloader(self.train_set,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             precompute_multi_scale=precompute_multi_scale,
                                             num_scales=num_scales,
                                             sample_method=sample_method)

        self.test_loader = self._dataloader(self.test_set,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            precompute_multi_scale=precompute_multi_scale,
                                            num_scales=num_scales,
                                            sample_method=sample_method)


if __name__ == '__main__':
    root = '/media/yangfei/HGST3/DATA/S3DISRoom'
    # root = '/home/disk1/DATA/S3DIS'
    dataset = S3DISRoomDataset(root, test_area=5, num_points=65536)
    dataset.create_dataloader(batch_size=16,
                              shuffle=True,
                              num_workers=0,
                              precompute_multi_scale=True,
                              num_scales=5)
    for data in dataset.train_loader:
        print(data)
    print(dataset)


