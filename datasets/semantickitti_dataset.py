# -*- coding:utf-8 -*-
import sys
import os
import yaml
import math
import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class SemanticKITTIDataset(Dataset):
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
                 sequences=None,
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
        self.lut = None
        self.split = None
        self.load_config('/media/yangfei/Repository/DATA/SemanticKITTI/raw/semantic-kitti.yaml')
        super(SemanticKITTIDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if sequences is None:
            self.sequences = ['{:02d}'.format(i) for i in self.split['train']]
        elif sequences in ['train', 'val', 'valid', 'test']:
            if sequences == 'val':
                sequences = 'valid'
            self.sequences = ['{:02d}'.format(i) for i in self.split[sequences]]
        else:
            assert sequences in self.processed_file_names
            self.sequences = sequences
        self.filelist = []
        for seq in sequences:
            seq_path = os.path.join(self.processed_dir, seq)
            seq_filelist = os.listdir(seq_path)
            for i in range(len(seq_filelist)):
                self.filelist.append(os.path.join(seq_path, seq_filelist[i]))

    @property
    def raw_file_names(self):
        return ['semantic-kitti.yaml']

    @property
    def processed_file_names(self):
        return ['{:02d}'.format(i) for i in range(22)]

    def __len__(self):
        return len(self.filelist)

    def download(self):
        pass

    def load_config(self, filename):
        data = yaml.safe_load(open(filename, 'r'))
        remap_dict = data['learning_map']
        max_key = max(remap_dict.keys())
        self.lut = np.zeros((max_key + 100), dtype=np.int32)
        self.lut[list(remap_dict.keys())] = list(remap_dict.values())
        self.split = data['split']

    def load_labels(self, path):
        labels = np.fromfile(path, dtype=np.uint32)
        labels = labels.reshape((-1))
        sem_labels = labels & 0xFFFF
        inst_labels = labels >> 16
        assert ((sem_labels + (inst_labels << 16) == labels).all())
        return sem_labels, inst_labels

    def process_raw_path(self, seq_path, out_path):
        points_path = os.path.join(seq_path, 'velodyne')
        filelist = os.listdir(points_path)
        labels_path = os.path.join(seq_path, 'labels')

        for filename in filelist:
            frame = filename.split('.')[0]
            path = os.path.join(points_path, filename)
            scan = np.fromfile(path, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            points = scan[:, :3]
            intensity = scan[:, 3]
            if os.path.exists(labels_path):
                path = os.path.join(labels_path, frame + '.label')
                labels, _ = self.load_labels(path)
                # remap labels from 25 to 19
                labels = self.lut[labels]

            pos = torch.from_numpy(points.astype(np.float32))
            x = torch.from_numpy(intensity.astype(np.float32))
            y = torch.from_numpy(labels.astype(np.long)) if os.path.exists(labels_path) else None
            data = Data(pos=pos, x=x, y=y)
            save_path = os.path.join(out_path, frame + '.pt')
            print('Saving {} points to "{}".'.format(data.num_nodes, save_path))
            torch.save(data, save_path)

    def process(self):
        for i, seq in enumerate(self.processed_file_names):
            print('Processing sequence {}...'.format(seq))
            if os.path.exists(self.processed_paths[i]):
                continue
            os.makedirs(self.processed_paths[i])
            seq_path = os.path.join('/media/yangfei/Repository/KITTI/data_odometry/dataset/sequences', seq)
            self.process_raw_path(seq_path, self.processed_paths[i])

    def get(self, idx):
        return torch.load(self.filelist[idx])


if __name__ == '__main__':
    dataset = SemanticKITTIDataset(root='/media/yangfei/Repository/DATA/SemanticKITTI')
    min_n = sys.maxsize
    for data in dataset:
        if data.num_nodes < min_n:
            min_n = data.num_nodes
    print(min_n)
    print(dataset)