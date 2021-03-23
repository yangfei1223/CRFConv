# -*- coding:utf-8 -*-
import os, sys, glob, pickle
from functools import partial
import random
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import torch
from plyfile import PlyData
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch_points_kernels.points_cpu as tpcpu
import torch_points_kernels.points_cuda as tpcuda
from torch_points3d.datasets.batch import SimpleBatch
from torch_points3d.datasets.multiscale_data import MultiScaleData, MultiScaleBatch
from utils import cpp_subsampling, nearest_neighbors
from utils import read_ply, write_ply
from utils import Plot


CLASS_NAMES = {'unlabeled': 0, 'man-made terrain': 1, 'natural terrain': 2, 'high vegetation': 3, 'low vegetation': 4,
               'buildings': 5, 'hard scape': 6, 'scanning artefacts': 7, 'cars': 8}


class Semantic3DDataset(Dataset):
    """
    ShapeNet dataset.
    --folder', '-f', help='Path to data folder
    --max_point_num', '-m', help='Max point number of each sample', type=int, default=8192
    --block_size', '-b', help='Block size', type=float, default=1.5
    --grid_size', '-g', help='Grid size', type=float, default=0.03
    --save_ply', '-s', help='Convert .pts to .ply', action='store_true'
    """
    def __init__(self,
                 root,
                 split='train',
                 max_length=10000,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None
                 ):
        """Constructor.
        Args:
            root (str): Path of raw dataset.
            split (str): Specify train, val, or test set to load.
            max_length (int) : How much data to load.
            transform (obj): Data transformer.
            pre_transform (obj): Data transformer.
            pre_filter (obj): Data filter.
        """
        self.min_point_num = 500
        self.block_size = 5.0
        self.stride = 3.0
        self.padding = 0.5
        self.proportion = 0.02
        assert split in ['train', 'val', 'test']
        self.split = split
        super(Semantic3DDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if self.split in ['train', 'val']:
            filelist = list(np.loadtxt(os.path.join(self.processed_dir, self.split + '.txt'), dtype=np.str))
            random.shuffle(filelist)
            self.filelist = filelist[:max_length]
        else:
            filelist = os.listdir(os.path.join(self.processed_dir, self.split))
            self.filelist = [os.path.join(self.processed_dir, self.split, f) for f in filelist]

    @property
    def raw_file_names(self):
        return ['trainval.txt', 'test.txt']

    @property
    def processed_file_names(self):
        return ['trainval', 'test']

    def __len__(self):
        return len(self.filelist)

    def download(self):
        pass

    def get_raw_file_list(self):
        with open(self.raw_paths[0], 'r') as f:
            trainval_list = [d.rstrip('\n') for d in f.readlines()]

        with open(self.raw_paths[1], 'r') as f:
            test_list = [d.rstrip('\n') for d in f.readlines()]

        return [trainval_list, test_list]

    def process_raw_path(self, filelist, output_path):
        if os.path.exists(output_path):
            return
        os.makedirs(output_path)
        split = output_path.split('/')[-1]
        for filename in filelist:
            path = os.path.join(self.raw_dir, 'ply', filename + '.ply')
            print('Loading {} ...'.format(filename))
            with open(path, 'rb') as f:
                plydata = PlyData.read(f)
            data = np.array(plydata.elements[0].data)
            xyz = np.stack((data['x'], data['y'], data['z']), axis=1).astype(np.float32)
            rgb = np.stack((data['r'], data['g'], data['b']), axis=1).astype(np.float32)
            if split != 'test':
                labels = data['c'] - 1  # 0 denotes unlabeled

            point_num = xyz.shape[0]
            point_indices = np.arange(point_num).astype(np.long)

            xyz_min = np.amin(xyz, axis=0)
            xyz -= xyz_min  # align to the min point
            limit = np.amax(xyz, axis=0)
            rgb /= 255.

            xbeg_list = []
            ybeg_list = []
            num_block_x = int(np.ceil(limit[0] - self.block_size) / self.stride) + 1
            num_block_y = int(np.ceil(limit[1] - self.block_size) / self.stride) + 1
            for i in range(num_block_x):
                for j in range(num_block_y):
                    xbeg_list.append(i * self.stride)
                    ybeg_list.append(j * self.stride)

            print('Collect {} blocks.'.format(num_block_x*num_block_y))

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
                if np.sum(mask)/mask.shape[0] < self.proportion:
                    continue

                block_min = np.amin(block_xyz, axis=0, keepdims=True)
                block_max = np.amax(block_xyz, axis=0, keepdims=True)
                block_center = (block_min + block_max) / 2
                block_center[0][-1] = block_min[0][-1]
                block_normalized = block_xyz - block_center  # align to block bottom center

                pos = torch.from_numpy(block_xyz.astype(np.float32))
                x = torch.from_numpy(np.concatenate((block_normalized, rgb[cond]), axis=-1).astype(np.float32))
                y = torch.from_numpy(labels[cond].astype(np.long)) if split != 'test' else None
                indices = torch.from_numpy(point_indices[cond].astype(np.long))
                mask = torch.from_numpy(mask.astype(np.int8))
                data = Data(pos=pos, x=x, y=y, indices=indices, mask=mask)
                save_path = os.path.join(output_path, filename + '_{:06d}.pt'.format(block_count))
                print('Saving {} points to "{}".'.format(data.num_nodes, save_path))
                torch.save(data, save_path)
                block_count += 1
            print('Split {} points to {} blocks.'.format(point_num, block_count))

    def process_train_val_split(self):
        if os.path.exists(os.path.join(self.processed_dir, 'train.txt') or os.path.join(self.processed_dir, 'val.txt')):
            return
        filelist = os.listdir(os.path.join(self.processed_paths[0]))
        random.shuffle(filelist)
        idx = (len(filelist) // 6) * 5
        train_filelist = [os.path.join(self.processed_paths[0], f) for f in filelist[:idx]]
        val_filelist = [os.path.join(self.processed_paths[0], f) for f in filelist[idx:]]
        np.savetxt(os.path.join(self.processed_dir, 'train.txt'), np.array(train_filelist), fmt='%s')
        np.savetxt(os.path.join(self.processed_dir, 'val.txt'), np.array(val_filelist), fmt='%s')

    def process(self):
        trainval_file_list, test_file_list = self.get_raw_file_list()
        print('Processing trainval ...')
        self.process_raw_path(trainval_file_list, self.processed_paths[0])
        print('Processing testing ...')
        self.process_raw_path(test_file_list, self.processed_paths[1])
        print('Generating train/val split ... ')
        self.process_train_val_split()

    def get(self, idx):
        return torch.load(self.filelist[idx])


class Semantic3D(InMemoryDataset):
    def __init__(self,
                 root,
                 split='train',
                 grid_size=0.06,
                 num_points=65536,
                 sample_per_epoch=100,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None
                 ):
        assert split in ['train', 'val', 'test']
        self.grid_size = grid_size
        self.num_points = num_points
        self.sample_per_epoch = sample_per_epoch
        self.split = split
        self.label_values = np.sort([v for k, v in CLASS_NAMES.items()])        # [0-8], 0 for unlabeled
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignore_labels = np.argsort([0])
        super(Semantic3D, self).__init__(root, transform, pre_transform, pre_filter)

        # Following KPConv and RandLA-Net train-val split
        self.all_splits = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
        self.val_split = 1

        # Initial train-val-test files
        self.train_files = []
        self.val_files = []
        self.test_files = []
        cloud_names = [filename[:-4] for filename in os.listdir(self.raw_paths[0]) if filename[-4:] == '.txt']
        for cloud_name in cloud_names:
            if os.path.exists(os.path.join(self.raw_paths[0], cloud_name + '.labels')):
                self.train_files.append(os.path.join(self.processed_paths[1], cloud_name + '.ply'))
            else:
                self.test_files.append(os.path.join(self.processed_paths[0], cloud_name + '.ply'))
        self.train_files = np.sort(self.train_files)
        self.test_files = np.sort(self.test_files)

        for i, filename in enumerate(self.train_files):
            if self.all_splits[i] == self.val_split:
                self.val_files.append(filename)

        self.train_files = np.sort([x for x in self.train_files if x not in self.val_files])

        # Initial containers
        self.test_proj = []
        self.test_labels = []

        self.possibility = []
        self.min_possibility = []
        self.class_weight = []
        self.input_trees = []
        self.input_rgb = []
        self.input_labels = []
        # self.input_names = []

        # Ascii files dict for testing
        self.ascii_files = {
            'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply': 'marketsquarefeldkirch4-reduced.labels',
            'sg27_station10_rgb_intensity-reduced.ply': 'sg27_10-reduced.labels',
            'sg28_Station2_rgb_intensity-reduced.ply': 'sg28_2-reduced.labels',
            'StGallenCathedral_station6_rgb_intensity-reduced.ply': 'stgallencathedral6-reduced.labels',
            'birdfountain_station1_xyz_intensity_rgb.ply': 'birdfountain1.labels',
            'castleblatten_station1_intensity_rgb.ply': 'castleblatten1.labels',
            'castleblatten_station5_xyz_intensity_rgb.ply': 'castleblatten5.labels',
            'marketplacefeldkirch_station1_intensity_rgb.ply': 'marketsquarefeldkirch1.labels',
            'marketplacefeldkirch_station4_intensity_rgb.ply': 'marketsquarefeldkirch4.labels',
            'marketplacefeldkirch_station7_intensity_rgb.ply': 'marketsquarefeldkirch7.labels',
            'sg27_station10_intensity_rgb.ply': 'sg27_10.labels',
            'sg27_station3_intensity_rgb.ply': 'sg27_3.labels',
            'sg27_station6_intensity_rgb.ply': 'sg27_6.labels',
            'sg27_station8_intensity_rgb.ply': 'sg27_8.labels',
            'sg28_station2_intensity_rgb.ply': 'sg28_2.labels',
            'sg28_station5_xyz_intensity_rgb.ply': 'sg28_5.labels',
            'stgallencathedral_station1_intensity_rgb.ply': 'stgallencathedral1.labels',
            'stgallencathedral_station3_intensity_rgb.ply': 'stgallencathedral3.labels',
            'stgallencathedral_station6_intensity_rgb.ply': 'stgallencathedral6.labels'}

        # load processed data
        self._load_processed()

        # random init probability
        for tree in self.input_trees:
            self.possibility += [np.random.randn(tree.data.shape[0]) * 1e-3]
            self.min_possibility += [float(np.min(self.possibility[-1]))]

        if split is not 'test':
            _, num_class_total = np.unique(np.hstack(self.input_labels), return_counts=True)
            self.class_weight += [np.squeeze([num_class_total / np.sum(num_class_total)], axis=0)]

    @property
    def raw_file_names(self):
        return ['txt']

    @property
    def processed_file_names(self):
        return ['original_reduced', 'sampled']

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
    def _load_cloud(filename):
        pc = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float32)
        pc = pc.values
        return pc

    @staticmethod
    def _load_label(filename):
        label = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.uint8)
        label = label.values
        return label

    @staticmethod
    def _grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)

    def process(self):
        for path in self.processed_paths:
            os.makedirs(path)
        for pc_path in glob.glob(os.path.join(self.raw_paths[0], '*.txt')):
            print('Processing {} ...'.format(pc_path))
            cloud_name = pc_path.split('/')[-1][:-4]
            if os.path.exists(os.path.join(self.processed_paths[1], cloud_name + '_KDTree.pkl')):
                continue
            pc = self._load_cloud(pc_path)
            label_path = pc_path[:-4] + '.labels'
            if os.path.exists(label_path):
                labels = self._load_label(label_path)
                org_ply_path = os.path.join(self.processed_paths[0], cloud_name + '.ply')
                # Subsample the training set cloud to the same resolution 0.01 as the test set
                xyz, rgb, labels = self._grid_sub_sampling(pc[:, :3].astype(np.float32),
                                                           pc[:, 4:7].astype(np.uint8),
                                                           labels, grid_size=0.01)
                labels = np.squeeze(labels)
                # save sub-sampled original cloud
                write_ply(org_ply_path, [xyz, rgb, labels], ['x', 'y', 'z', 'r', 'g', 'b', 'class'])

                # save sub_cloud and KDTree file
                sub_xyz, sub_rgb, sub_labels = self._grid_sub_sampling(xyz, rgb, labels, grid_size=self.grid_size)
                sub_rgb = sub_rgb / 255.
                sub_labels = np.squeeze(sub_labels)
                sub_ply_file = os.path.join(self.processed_paths[1], cloud_name + '.ply')
                write_ply(sub_ply_file, [sub_xyz, sub_rgb, sub_labels], ['x', 'y', 'z', 'r', 'g', 'b', 'class'])

                search_tree = KDTree(sub_xyz, leaf_size=50)
                kd_tree_file = os.path.join(self.processed_paths[1], cloud_name + '_KDTree.pkl')
                with open(kd_tree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
                proj_idx = proj_idx.astype(np.int32)
                proj_file = os.path.join(self.processed_paths[1], cloud_name + '_proj.pkl')
                with open(proj_file, 'wb') as f:
                    pickle.dump([proj_idx, labels], f)

            else:
                org_ply_path = os.path.join(self.processed_paths[0], cloud_name + '.ply')
                write_ply(org_ply_path, [pc[:, :3].astype(np.float32), pc[:, 4:7].astype(np.uint8)],
                          ['x', 'y', 'z', 'r', 'g', 'b'])

                sub_xyz, sub_rgb = self._grid_sub_sampling(pc[:, :3].astype(np.float32),
                                                           pc[:, 4:7].astype(np.uint8),
                                                           grid_size=self.grid_size)

                sub_rgb = sub_rgb / 255.
                sub_ply_file = os.path.join(self.processed_paths[1], cloud_name + '.ply')
                write_ply(sub_ply_file, [sub_xyz, sub_rgb], ['x', 'y', 'z', 'r', 'g', 'b'])

                search_tree = KDTree(sub_xyz, leaf_size=50)
                kd_tree_file = os.path.join(self.processed_paths[1], cloud_name + '_KDTree.pkl')
                with open(kd_tree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                labels = np.zeros(pc.shape[0], dtype=np.uint8)
                proj_idx = np.squeeze(search_tree.query(pc[:, :3].astype(np.float32), return_distance=False))
                proj_idx = proj_idx.astype(np.int32)
                proj_file = os.path.join(self.processed_paths[1], cloud_name + '_proj.pkl')
                with open(proj_file, 'wb') as f:
                    pickle.dump([proj_idx, labels], f)

    def _load_processed(self):
        if self.split is 'train':
            file_list = self.train_files
        elif self.split == 'val':
            file_list = self.val_files
        elif self.split == 'test':
            file_list = self.test_files
        else:
            raise ValueError('Only `train`, `val` or `test` split is supported !')
        for i, filename in enumerate(file_list):
            cloud_name = filename.split('/')[-1][:-4]
            print('Load cloud {:d}: {:s}'.format(i, cloud_name))
            kd_tree_file = os.path.join(self.processed_paths[1], cloud_name + '_KDTree.pkl')
            sub_ply_file = os.path.join(self.processed_paths[1], cloud_name + '.ply')
            # read ply data
            data = read_ply(sub_ply_file)
            sub_rgb = np.vstack((data['r'], data['g'], data['b'])).T
            if self.split is 'test':
                sub_labels = None
            else:
                sub_labels = data['class']

            # read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees += [search_tree]
            self.input_rgb += [sub_rgb]
            if self.split in ['train', 'val']:
                self.input_labels += [sub_labels]

            if self.split in ['val', 'test']:
                print('Preparing re-projection indices for val and test')
                proj_file = os.path.join(self.processed_paths[1], cloud_name + '_proj.pkl')
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                    self.test_proj += [proj_idx]
                    self.test_labels += [labels]

        print('Finished.')

    def _get_random(self):
        cloud_idx = int(np.argmin(self.min_possibility))
        pick_idx = np.argmin(self.possibility[cloud_idx])
        points = np.array(self.input_trees[cloud_idx].data, copy=False)
        pick_point = points[pick_idx, :].reshape(1, -1)

        noise = np.random.normal(scale=3.5 / 10, size=pick_point.shape)
        pick_point = pick_point + noise.astype(pick_point.dtype)

        # Semantic3D is a big dataset with large cloud, so there is no need to resample
        query_idx = self.input_trees[cloud_idx].query(pick_point, k=self.num_points)[1][0]

        np.random.shuffle(query_idx)
        query_xyz = points[query_idx]
        query_xyz[:, 0:2] = query_xyz[:, 0:2] - pick_point[:, 0:2]      # centerize in xOy plane
        query_rgb = self.input_rgb[cloud_idx][query_idx]
        if self.split is 'test':
            query_labels = np.zeros(query_xyz.shape[0])
            query_weights = 1
        else:
            query_labels = self.input_labels[cloud_idx][query_idx]
            query_labels = np.array([self.label_to_idx[l] for l in query_labels])
            query_weights = np.array([self.class_weight[0][n] for n in query_labels])

        # update possibility, reduce the possibility of chosen cloud and point
        dists = np.sum(np.square(points[query_idx] - pick_point).astype(np.float32), axis=1)
        delta = np.square(1 - dists / np.max(dists)) * query_weights
        self.possibility[cloud_idx][query_idx] += delta
        self.min_possibility[cloud_idx] = float(np.min(self.possibility[cloud_idx]))

        pos = torch.from_numpy(query_xyz).to(torch.float32)
        rgb = torch.from_numpy(query_rgb).to(torch.float32)
        labels = torch.from_numpy(query_labels).to(torch.long)
        point_idx = torch.from_numpy(query_idx).to(torch.long)
        cloud_idx = torch.Tensor([cloud_idx]).to(torch.long)
        data = Data(pos=pos, rgb=rgb, y=labels, point_idx=point_idx, cloud_idx=cloud_idx)

        return data


class Semantic3DWholeDataset:
    def __init__(self,
                 root,
                 grid_size=0.06,
                 num_points=65536,
                 train_sample_per_epoch=500,
                 test_sample_per_epoch=100,
                 train_transform=None,
                 test_transform=None):
        self.kernel_size = [16, 16, 16, 16, 16]
        self.ratio = [4, 4, 4, 4, 2]

        self.train_set = Semantic3D(root,
                                    split='train',
                                    grid_size=grid_size,
                                    num_points=num_points,
                                    sample_per_epoch=train_sample_per_epoch,
                                    transform=train_transform)

        self.val_set = Semantic3D(root,
                                  split='val',
                                  grid_size=grid_size,
                                  num_points=num_points,
                                  sample_per_epoch=test_sample_per_epoch,
                                  transform=test_transform)

        self.test_set = Semantic3D(root,
                                   split='test',
                                   grid_size=grid_size,
                                   num_points=num_points,
                                   sample_per_epoch=test_sample_per_epoch,
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

        self.val_loader = self._dataloader(self.val_set,
                                           batch_size=batch_size,
                                           shuffle=False,
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
    root = '/media/yangfei/HGST3/DATA/Semantic3D'
    dataset = Semantic3DWholeDataset(root=root, grid_size=0.06, num_points=65536, split='trainval')
    # for data in dataset.train_set:
    #     Plot.draw_pc(torch.cat([data.pos, data.rgb], dim=-1).numpy())
    #     Plot.draw_pc_sem_ins(data.pos.numpy(), data.y.numpy())
    dataset.create_dataloader(batch_size=16, shuffle=True, num_workers=0, precompute_multi_scale=True, num_scales=5)
    for data in dataset.train_loader:
        print(data)
    print(dataset)

