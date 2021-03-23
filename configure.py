import numpy as np
import torch
'''
ShapeNet Parts

'Airplane': '2349/341', '4',
'Bag': '62/14', '2',
'Cap': '44/11', '2',
'Car': '740/158', '4',
'Chair': '3054/704', '4'
'Earphone': '55/14', '3'
'Guitar': '628/159', '3'
'Knife': '312/80', '2'
'Lamp': '1261/286', '4'
'Laptop': '368/83', '2'
'Motorbike': '151/51', '6'
'Mug': '146/38', '2'
'Pistol': '239/44', '3'
'Rocket': '54/12', '3'
'Skateboard': '121/31', '3'
'Table': '4423/848', '3'
'''

'''
ScanNet Labels:
['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 
'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator', 'picture', 'cabinet', 'otherfurniture']
'''


def get_class_weights(dataset):
    # pre-calculate the number of points in each category
    num_per_class = []
    if dataset is 'S3DIS':
        num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                                  650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
    elif dataset is 'Semantic3D':
        num_per_class = np.array([5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353],
                                 dtype=np.int32)
    elif dataset is 'SemanticKITTI':
        num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                  240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                  9833174, 129609852, 4506626, 1168181])
    weight = num_per_class / float(sum(num_per_class))
    ce_label_weight = 1 / (weight + 0.02)
    # return np.expand_dims(ce_label_weight, axis=0)
    return torch.from_numpy(ce_label_weight.astype(np.float32))


class ShapeNetConfig(object):
    device = 'cuda'
    device_id = 0
    dataset = 'ShapeNet'
    # root = '/home/disk1/DATA/ShapeNetNormal'
    root = '/media/yangfei/Repository/DATA/ShapeNetNormal'
    # model = 'BaselineSegNet_Part'
    model = 'CRFSegNet_Part'
    # model = 'DualCRFSegNet_Part'
    steps = 10
    num_classes = 50
    epochs = 100
    batch_size = 16
    gamma = 0.1**0.02
    lr = 1e-2
    momentum = 0.95
    weight_decay = 1e-4
    # model_path = None
    # model_path = 'checkpoints/BaselineSegNet_Part_on_ShapeNet.ckpt'
    model_path = 'checkpoints/{}_on_{}.ckpt'.format(model, dataset)
    # model_path = '../checkpoints/CRFSegNet_Part_on_ShapeNet.ckpt'


class S3DISConfig(object):
    device = 'cuda'
    device_id = 0
    dataset = 'S3DIS'
    # root = '/home/disk1/DATA/S3DIS'
    # root = '/media/Repository/DATA/S3DIS'
    # root = '/media/yangfei/Repository/DATA/S3DIS'
    root = '/media/yangfei/HGST3/DATA/S3DIS'
    # model_name = 'PointTransformerSegNet'
    # model_name = 'ResidualTransformerSegNet'
    # model_name = 'RandLANet'
    # model_name = 'PointConvBig'
    # model = 'DualCRFSegNet'
    model_name = 'BaselineSegNet'
    # model_name = 'CRFSegNet'
    # model = 'BaselineDiscreteCRFSegNet'
    mode = 'train'
    use_crf = False
    steps = 0
    grid_size = 0.04
    repeat_num = 3
    sample_num = 8192
    num_classes = 13   # (12 denotes clutters)
    class_weights = get_class_weights(dataset)
    epochs = 100
    batch_size = 8
    train_samples_per_epoch = batch_size * 100
    val_samples_per_epoch = batch_size * 100
    gamma = 0.95
    lr = 1e-2
    momentum = 0.95
    weight_decay = 1e-4
    prefix = '{}_on_{}_bs_{}_epochs_{}'.format(model_name, dataset, batch_size, epochs)
    # model_path = 'checkpoints/{}_big.ckpt'.format(prefix)
    model_path = None

class ScanNetConfig(object):
    device = 'cuda'
    device_id = 0
    dataset = 'ScanNet'
    root = '/home/disk1/DATA/{}'.format(dataset)
    # root = '/media/yangfei/Repository/DATA/{}'.format(dataset)
    # model = 'BaselineSegNet'
    model = 'CRFSegNet'
    steps = 10
    repeat_num = 3
    sample_num = 8192
    num_classes = 20    # (20 categories in total, -1 is ignored)
    ignore_index = -1
    epochs = 100
    batch_size = 16
    gamma = 0.1**0.02
    lr = 1e-2
    momentum = 0.95
    weight_decay = 1e-4
    # model_path = None
    model_path = 'checkpoints/CRFSegNet_on_ScanNet_ACC.ckpt'
    # model_path = 'checkpoints/CRFSegNet_on_ScanNet.ckpt'


class Semantic3DConfig(object):
    device = 'cuda'
    device_id = 0
    dataset = 'Semantic3D'
    root = '/media/yangfei/HGST3/DATA/{}'.format(dataset)
    model_name = 'PointConvBig'
    mode = 'test'
    use_crf = True
    steps = 1
    grid_size = 0.06
    repeat_num = 3
    sample_num = 65536
    num_classes = 8    # (8 categories in total, -1 is ignored)
    ignore_index = -1
    class_weights = get_class_weights(dataset)
    epochs = 100
    batch_size = 16
    train_samples_per_epoch = batch_size * 500
    test_samples_per_epoch = batch_size * 100
    gamma = 0.1**0.02
    lr = 1e-2
    momentum = 0.95
    weight_decay = 1e-4
    prefix = '{}_on_{}_bs_{}_epochs_{}'.format(model_name, dataset, batch_size, epochs)
    model_path = 'checkpoints/{}_big_crf.ckpt'.format(prefix)
    test_point_num = [10538633, 14608690, 28931322, 24620684]


class NPM3DConfig(object):
    device = 'cuda'
    device_id = 0
    dataset = 'Paris-Lille-3D'
    root = '/home/disk1/DATA/{}'.format(dataset)
    # root = '/media/yangfei/Repository/DATA/{}'.format(dataset)
    # model = 'BaselineSegNet'
    model = 'CRFSegNet'
    steps = 10
    repeat_num = 3
    sample_num = 8192
    num_classes = 9    # (9 categories in total, -1 is ignored)
    ignore_index = -1
    epochs = 100
    batch_size = 16
    lr = 1e-2
    momentum = 0.95
    weight_decay = 1e-4
    model_path = None
    # model_path = 'checkpoints/BaselineSegNet_on_Paris-Lille-3D.ckpt'
    # model_path = 'checkpoints/CRFSegNet_on_Paris-Lille-3D.ckpt'
    # model_path = 'checkpoints/{}_on_{}.ckpt'.format(model, dataset)


