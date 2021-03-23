# -*- coding:utf-8 -*-
import sys
import os
import json
import pickle
import math
import random
import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class ScanNetDataset(Dataset):
    """
    ScanNet dataset.
    """
    def __init__(self,
                 root,
                 train=True,
                 max_length=10000,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None
                 ):
        """Constructor.
        Args:
            root (str): Path of raw dataset.
            train (bool): Boolean that indicates if this is the train or test dataset.
            transform (obj): Data transformer.
            pre_transform (obj): Data transformer.
            pre_filter (obj): Data filter.
        """
        self.min_point_num = 200
        self.block_size = 1.5
        self.stride = 1.0
        self.padding = 0.2
        self.proportion = 0.02
        super(ScanNetDataset, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        filelist = os.listdir(path)
        random.shuffle(filelist)
        self.filelist = [os.path.join(path, f) for f in filelist[:max_length]]

    @property
    def raw_file_names(self):
        return ['scannet_train.pickle', 'scannet_test.pickle']

    @property
    def processed_file_names(self):
        return ['train', 'test']

    def __len__(self):
        return len(self.filelist)

    def download(self):
        pass

    def process_raw_path(self, filename, output_path):
        if os.path.exists(output_path):
            return
        os.makedirs(output_path)
        print('Loading {}...'.format(filename))
        with open(filename, 'rb') as file_pickle:
            xyz_all = pickle.load(file_pickle, encoding='latin1')
            labels_all = pickle.load(file_pickle, encoding='latin1')
        label_dict = {}
        for room_idx, xyz in enumerate(xyz_all):
            point_num = xyz.shape[0]
            print('Processing room {}, collect {} points.'.format(room_idx, point_num))

            labels = labels_all[room_idx] - 1   # {0~20 -> -1~19, because 0 is unannotated class}
            label_dict[room_idx] = labels.astype(np.int32)
            point_indices = np.arange(point_num).astype(np.int32)

            xyz_min = np.amin(xyz, axis=0)
            xyz -= xyz_min  # align to the min point
            limit = np.amax(xyz, axis=0)
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
                if np.sum(mask)/mask.shape[0] < self.proportion:
                    continue

                pos = torch.from_numpy(block_xyz.astype(np.float32))
                x = torch.from_numpy(xyz_room_normalized[cond].astype(np.float32))
                y = torch.from_numpy(labels[cond].astype(np.long))
                mask = torch.from_numpy(mask.astype(np.int8))
                indices = torch.from_numpy(point_indices[cond].astype(np.long))
                data = Data(pos=pos, x=x, y=y, mask=mask, indices=indices)
                save_path = os.path.join(output_path, 'room_{:04d}_{:06d}.pt'.format(room_idx, block_count))
                print('Saving {} points to "{}".'.format(data.num_nodes, save_path))
                torch.save(data, save_path)
                block_count += 1
            print('Split {} points to {} blocks.'.format(point_num, block_count))
        return label_dict

    def process(self):
        print('Processing training ...')
        label_dict = self.process_raw_path(self.raw_paths[0], self.processed_paths[0])
        np.save(os.path.join(self.processed_dir, 'train_label.npy'), label_dict)
        print('Processing testing ...')
        label_dict = self.process_raw_path(self.raw_paths[1], self.processed_paths[1])
        np.save(os.path.join(self.processed_dir, 'test_label.npy'), label_dict)

    def get(self, idx):
        return torch.load(self.filelist[idx])


'''
class ScanNetDataset(InMemoryDataset):
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
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None
                 ):
        """Constructor.
        Args:
            root (str): Path of raw dataset.
            train (bool): Boolean that indicates if this is the train or test dataset.
            transform (obj): Data transformer.
            pre_transform (obj): Data transformer.
            pre_filter (obj): Data filter.
        """
        self.max_point_num = 8192
        self.block_size = 1.5
        self.grid_size = 0.03
        self.offset = self.block_size/2
        super(ScanNetDataset, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['scannet_train.pickle', 'scannet_test.pickle']

    @property
    def processed_file_names(self):
        return ['training.pt', 'testing.pt']

    def download(self):
        pass

    def process_raw_path(self, filename):
        print('Loading {}...'.format(filename))
        file_pickle = open(filename, 'rb')
        xyz_all = pickle.load(file_pickle, encoding='latin1')
        labels_all = pickle.load(file_pickle, encoding='latin1')
        file_pickle.close()
        data_list = []
        for room_idx, xyz in enumerate(xyz_all):
            print('Processing room {}...'.format(room_idx))
            # align to room bottom center
            xyz_min = np.amin(xyz, axis=0, keepdims=True)
            xyz_max = np.amax(xyz, axis=0, keepdims=True)
            xyz_center = (xyz_min+xyz_max)/2
            xyz_center[0][-1] = xyz_min[0][-1]
            xyz = xyz - xyz_center  # bottom center of the room

            labels = labels_all[room_idx]
            print('Computing block id of {} points...'.format(xyz.shape[0]))
            xyz_min = np.amin(xyz, axis=0, keepdims=True)-self.offset   # new xyz_min
            xyz_max = np.amax(xyz, axis=0, keepdims=True)   # new xyz_max
            block_size = np.array([self.block_size, self.block_size, 2*(xyz_max[0, -1]-xyz_min[0, -1])])
            xyz_blocks = np.floor((xyz-xyz_min)/block_size).astype(np.int)

            print('Collecting points belong to each block...'.format(xyz.shape[0]))
            blocks, point_block_indices, block_point_counts = np.unique(
                xyz_blocks, return_inverse=True, return_counts=True, axis=0)
            block_point_indices = np.split(np.argsort(point_block_indices), np.cumsum(block_point_counts[:-1]))
            print('Data is split into {} blocks.'.format(blocks.shape[0]))

            block_to_block_idx_map = dict()
            for block_idx, block in enumerate(blocks):
                block_to_block_idx_map[(block[0], block[1])] = block_idx

            # merge small blocks into one of their big neighbors
            block_point_count_threshold = self.max_point_num / 10
            first_nbr_block_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4 first neighbors
            second_nbr_block_offsets = [(-1, 1), (1, 1), (1, -1), (-1, -1)]  # 4 second neighbors
            block_merge_count = 0
            for block_idx in range(blocks.shape[0]):
                if block_point_counts[block_idx] >= block_point_count_threshold or block_point_counts[block_idx] == 0:
                    continue

                merge_flag = False
                valid_neighbors = [block_idx]
                block = (blocks[block_idx][0], blocks[block_idx][1])
                # merge to first neighbor
                merge_block_idx = -1
                merge_block_point_counts = sys.maxsize
                for x, y in first_nbr_block_offsets:
                    nbr_block = (block[0] + x, block[1] + y)
                    if nbr_block not in block_to_block_idx_map:
                        continue
                    nbr_block_idx = block_to_block_idx_map[nbr_block]
                    valid_neighbors.append(nbr_block_idx)    # all valid neighbors
                    cur_point_counts = block_point_counts[nbr_block_idx]
                    if cur_point_counts < block_point_count_threshold:
                        continue
                    # find smallest valid neighbor
                    if cur_point_counts < merge_block_point_counts:
                        merge_block_point_counts = cur_point_counts
                        merge_block_idx = nbr_block_idx

                if merge_block_idx != -1:
                    # merge to neighbor
                    block_point_indices[merge_block_idx] = np.concatenate(
                        [block_point_indices[merge_block_idx], block_point_indices[block_idx]], axis=-1)
                    block_point_counts[merge_block_idx] += block_point_counts[block_idx]
                    block_point_indices[block_idx] = np.array([], dtype=np.int)
                    block_point_counts[block_idx] = 0
                    block_merge_count += 1
                    merge_flag = True

                # merge to second neighbor
                if merge_flag is False:
                    merge_block_idx = -1
                    merge_block_offset_x = 0
                    merge_block_offset_y = 0
                    merge_block_point_counts = sys.maxsize
                    for x, y in second_nbr_block_offsets:
                        nbr_block = (block[0] + x, block[1] + y)
                        if nbr_block not in block_to_block_idx_map:
                            continue
                        nbr_block_idx = block_to_block_idx_map[nbr_block]
                        valid_neighbors.append(nbr_block_idx)  # all effective neighbors
                        cur_point_counts = block_point_counts[nbr_block_idx]
                        if cur_point_counts < block_point_count_threshold:
                            continue
                        # find smallest valid neighbor
                        if cur_point_counts < merge_block_point_counts:
                            merge_block_point_counts = cur_point_counts
                            merge_block_idx = nbr_block_idx
                            merge_block_offset_x = x
                            merge_block_offset_y = y

                    if merge_block_idx != -1:
                        # the first neighbor (0, y) and (x, 0)
                        merge_blocks = [merge_block_idx]
                        first_nbr_block1 = (block[0] + merge_block_offset_x, block[1])
                        if first_nbr_block1 in block_to_block_idx_map:
                            first_nbr_block_idx1 = block_to_block_idx_map[first_nbr_block1]
                            merge_blocks.append(first_nbr_block_idx1)
                        first_nbr_block2 = (block[0], block[1] + merge_block_offset_y)
                        if first_nbr_block2 in block_to_block_idx_map:
                            first_nbr_block_idx2 = block_to_block_idx_map[first_nbr_block2]
                            merge_blocks.append(first_nbr_block_idx2)
                        merge_blocks.append(block_idx)

                        block_point_indices[merge_block_idx] = np.concatenate(
                            [block_point_indices[idx] for idx in merge_blocks], axis=-1)
                        for idx in merge_blocks[1:]:
                            block_point_counts[merge_block_idx] += block_point_counts[idx]
                            block_point_indices[idx] = np.array([], dtype=np.int)
                            block_point_counts[idx] = 0
                        block_merge_count += len(merge_blocks) - 1
                        merge_flag = True

                # if a block is not merged to any neighbor, then merge all neighbors into one block
                if merge_flag is False:
                    block_point_indices[block_idx] = np.concatenate(
                        [block_point_indices[idx] for idx in valid_neighbors], axis=-1)
                    for idx in valid_neighbors[1:]:
                        block_point_counts[block_idx] += block_point_counts[idx]
                        block_point_indices[idx] = np.array([], dtype=np.int)
                        block_point_counts[idx] = 0
                    block_merge_count += len(valid_neighbors) - 1
            print('{} of {} blocks are merged.'.format(block_merge_count, blocks.shape[0]))

            idx_last_non_empty_block = 0
            for block_idx in reversed(range(blocks.shape[0])):
                if block_point_indices[block_idx].shape[0] != 0:
                    idx_last_non_empty_block = block_idx
                    break

            # uniformly sample each block
            for block_idx in range(idx_last_non_empty_block + 1):
                point_indices = block_point_indices[block_idx]
                if point_indices.shape[0] == 0:
                    continue
                block_points = xyz[point_indices]
                block_min = np.amin(block_points, axis=0, keepdims=True)
                xyz_grids = np.floor((block_points - block_min) / self.grid_size).astype(np.int)
                grids, point_grid_indices, grid_point_counts = np.unique(xyz_grids, return_inverse=True,
                                                                         return_counts=True, axis=0)
                grid_point_indices = np.split(np.argsort(point_grid_indices), np.cumsum(grid_point_counts[:-1]))
                grid_point_count_avg = int(np.average(grid_point_counts))
                point_indices_repeated = []
                for grid_idx in range(grids.shape[0]):
                    point_indices_in_block = grid_point_indices[grid_idx]
                    repeat_num = math.ceil(grid_point_count_avg / point_indices_in_block.shape[0])
                    if repeat_num > 1:
                        point_indices_in_block = np.repeat(point_indices_in_block, repeat_num)
                        np.random.shuffle(point_indices_in_block)
                        point_indices_in_block = point_indices_in_block[:grid_point_count_avg]
                    point_indices_repeated.extend(list(point_indices[point_indices_in_block]))
                block_point_indices[block_idx] = np.array(point_indices_repeated)
                block_point_counts[block_idx] = len(point_indices_repeated)

            # split big blocks
            for block_idx in range(idx_last_non_empty_block + 1):
                point_indices = block_point_indices[block_idx]
                if point_indices.shape[0] == 0:
                    continue

                block_point_num = point_indices.shape[0]
                block_split_num = int(math.ceil(block_point_num / self.max_point_num))
                point_num_avg = int(math.ceil(block_point_num / block_split_num))
                point_nums = [point_num_avg] * block_split_num
                point_nums[-1] = block_point_num - (point_num_avg * (block_split_num - 1))
                starts = [0] + list(np.cumsum(point_nums))

                np.random.shuffle(point_indices)
                block_points = xyz[point_indices]
                block_labels = labels[point_indices]

                for block_split_idx in range(block_split_num):
                    start = starts[block_split_idx]
                    point_num = point_nums[block_split_idx]
                    end = start + point_num
                    pos = torch.from_numpy(block_points[start:end, :].astype(np.float32))
                    y = torch.from_numpy(block_labels[start:end].astype(np.long))
                    ind_in_room = point_indices[start:end]
                    indices = torch.from_numpy(
                        np.stack([np.zeros_like(ind_in_room) + room_idx, ind_in_room], axis=-1).astype(np.long))
                    data = Data(pos=pos, y=y, indices=indices)
                    data_list.append(data)

        return data_list

    def process(self):
        print('Processing training ...')
        train_data_list = self.process_raw_path(self.raw_paths[0])
        print('Processing testing ...')
        test_data_list = self.process_raw_path(self.raw_paths[1])

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])
'''

if __name__ == '__main__':
    root = '/home/disk1/DATA/ScanNet'
    dataset = ScanNetDataset(root=root, train=True, max_length=500)
    print(dataset)
    min_y=100
    for data in dataset:
        y = data.y.max()
        if y<min_y:
            min_y=y
    print(min_y)

