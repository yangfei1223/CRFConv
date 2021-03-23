# -*- coding:utf-8 -*-
import sys
import os
import random
from plyfile import PlyData
import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class NPM3DDataset(Dataset):
    """
    ShapeNet dataset.
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
        super(NPM3DDataset, self).__init__(root, transform, pre_transform, pre_filter)
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
        label_dict = {}
        for filename in filelist:
            path = os.path.join(self.raw_dir, filename + '.ply')
            print('Loading {} ...'.format(path))
            with open(path, 'rb') as f:
                plydata = PlyData.read(f)

            data = np.array(plydata.elements[0].data)
            x, y, z, ref = data['x'], data['y'], data['z'], data['reflectance']
            xyz = np.stack((x, y, z), axis=1)

            if output_path.split('/')[-1] != 'test':
                labels = data['class'] - 1      # 0: unclassified
                label_dict[filename] = labels

            point_num = xyz.shape[0]

            point_indices = np.arange(point_num).astype(np.long)

            xyz_min = np.amin(xyz, axis=0)
            xyz -= xyz_min  # align to the min point
            limit = np.amax(xyz, axis=0)

            intensity = (ref / 255.).reshape((-1, 1))

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
                block_normalized = block_xyz - block_center     # align to block bottom center

                pos = torch.from_numpy(block_xyz.astype(np.float32))
                x = torch.from_numpy(
                    np.concatenate((block_normalized, intensity[cond]), axis=-1).astype(np.float32))
                y = torch.from_numpy(labels[cond].astype(np.long)) if label_dict else None
                indices = torch.from_numpy(point_indices[cond].astype(np.long))
                mask = torch.from_numpy(mask.astype(np.int8))
                data = Data(pos=pos, x=x, y=y, indices=indices, mask=mask)
                save_path = os.path.join(output_path, filename + '_{:06d}.pt'.format(block_count))
                print('Saving {} points to "{}".'.format(data.num_nodes, save_path))
                torch.save(data, save_path)
                block_count += 1
            print('Split {} points to {} blocks.'.format(point_num, block_count))

        if label_dict:
            np.save(os.path.join(self.processed_dir, 'label_{}.npy'.format(output_path.split('/')[-1])), label_dict)

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


if __name__ == '__main__':
    root = '/media/yangfei/Repository/DATA/Paris-Lille-3D'
    # root = '/home/disk1/DATA/Paris-Lille-3D'
    dataset = NPM3DDataset(root=root, split='val')
    print(dataset)
