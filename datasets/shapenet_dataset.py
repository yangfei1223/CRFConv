import sys
import os
import json
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


class ShapeNetNormalDataset(InMemoryDataset):
    """
    ShapeNet dataset.
    """
    def __init__(self,
                 root,
                 categories=None,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None
                 ):
        """Constructor.
        Args:
            root (str): Path of raw dataset.
            train (bool): Boolean that indicates if this is the train or test dataset.
            categories (tuple): tuple of categories
            transform (obj): Data transformer.
            pre_transform (obj): Data transformer.
            pre_filter (obj): Data filter.
        """
        self.category_ids = {}
        self.obj_classes = {}
        with open(os.path.join(root, 'raw', 'synsetoffset2category.txt'), 'r') as file:
            for i, line in enumerate(file):
                strings = line.replace("\n", "").split("\t")
                self.category_ids[strings[0]] = strings[1]
                self.obj_classes[strings[0]] = i

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29],
                            'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21],
                            'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
                            'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15],
                            'Knife': [22, 23]}

        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories

        super(ShapeNetNormalDataset, self).__init__(root, transform, pre_transform, pre_filter)
        # self.categories = list(self.category_ids.keys())
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['train_test_split/shuffled_train_file_list.json',       # training split
                'train_test_split/shuffled_val_file_list.json',         # validation split
                'train_test_split/shuffled_test_file_list.json']        # testing split

    @property
    def processed_file_names(self):
        # cats = '_'.join([cat[:3].lower() for cat in self.categories])
        # return ['{}_{}.pt'.format(cats, s) for s in ['training', 'testing']]
        return ['training.pt', 'testing.pt']

    def download(self):
        pass

    def get_raw_file_list(self):
        with open(self.raw_paths[0], 'r') as f:
            train_list = list([os.path.join(self.raw_dir, d.split('/')[1], d.split('/')[2] + '.txt') for d in json.load(f)])

        with open(self.raw_paths[1], 'r') as f:
            val_list = list([os.path.join(self.raw_dir, d.split('/')[1], d.split('/')[2] + '.txt') for d in json.load(f)])

        with open(self.raw_paths[2], 'r') as f:
            test_list = list([os.path.join(self.raw_dir, d.split('/')[1], d.split('/')[2] + '.txt') for d in json.load(f)])

        return [train_list, val_list, test_list]

    def process_raw_path(self, file_list):
        data_list = []
        for i, filename in enumerate(file_list):
            print('[%d/%d] %s' % (i, len(file_list), filename))
            category = torch.from_numpy(np.array(
                [self.obj_classes[k] for (k, v) in self.category_ids.items() if v == filename.split('/')[2]],
                dtype=np.long))
            if category not in self.categories:
                continue
            raw = np.loadtxt(filename)
            pos = torch.from_numpy(raw[:, 0:3].astype(np.float32))
            norm = torch.from_numpy(raw[:, 3:6].astype(np.float32))
            y = torch.from_numpy(raw[:, -1].astype(np.long))

            data = Data(x=None, y=y, pos=pos, norm=norm, category=category)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        return data_list

    def process(self):
        train_file_list, val_file_list, test_file_list = self.get_raw_file_list()
        print('Processing %d training ...' % len(train_file_list))
        train_data_list = self.process_raw_path(train_file_list)
        print('Processing %d validation ...' % len(val_file_list))
        val_data_list = self.process_raw_path(val_file_list)
        print('Processing %d testing ...' % len(test_file_list))
        test_data_list = self.process_raw_path(test_file_list)

        torch.save(self.collate(train_data_list + val_data_list), self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])


if __name__ == '__main__':
    root = '/media/yangfei/Repository/DATA/ShapeNetNormal'
    dataset = ShapeNetNormalDataset(root=root, train=False)
    max_num = -1
    min_num = sys.maxsize
    for data in dataset:
        n = data.num_nodes
        if n > max_num:
            max_num = n
        if n < min_num:
            min_num = n
    print('Max size: {}, Min size: {}'.format(max_num, min_num))
    print(dataset)