import os
import math
import numpy as np
import torch
from torch_geometric.transforms import Compose, NormalizeScale, FixedPoints
from torch_geometric.data import DataLoader
from models import conv_crf_seg as seg_models
from configure import config_s3dis as config
from mayavi import mlab

device = torch.device(config.device)
torch.cuda.set_device(config.device_id)

data_path = os.path.join(config.root, 'processed/Area_5')
test_set = os.listdir(data_path)
test_set.sort()

label_path = os.path.join(config.root, 'processed', 'label_area_5.npy')
label_dict = np.load(label_path, allow_pickle=True).item()

predict_dict = np.load('../results/S3DIS/pred_dict_s3dis_steps_5.npy', allow_pickle=True).item()

num_rooms = 68


def get_pos(dataset, point_num):
    room_pos = np.zeros((point_num, 3))
    room_rgb = np.zeros((point_num, 3))
    for file in dataset:
        data = torch.load(os.path.join(data_path, file))
        pos = data.pos.numpy()
        rgb = data.x.numpy()[:, :3]
        indices = data.indices.cpu().numpy()
        room_pos[indices] = pos
        room_rgb[indices] = rgb
    return room_pos, room_rgb


def display_rgb(pos, color, filename):
    N = pos.shape[0]
    s = np.arange(N)
    lut = np.zeros((N, 4), dtype=np.uint8)
    lut[:, :3] = (color*255).astype(np.uint8)
    lut[:, -1] = 255

    mlab.figure()
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    points3d = mlab.points3d(x, y, z, s,
                             mask_points=10,
                             colormap='jet',
                             scale_factor=0.025,
                             scale_mode='none')
    points3d.module_manager.scalar_lut_manager.lut.table = lut
    # mlab.show()
    mlab.savefig(filename, magnification=2)
    mlab.close()


def display(pos, label, filename):
    mlab.figure()
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    mlab.points3d(x, y, z, label,
                  mask_points=10,
                  colormap='jet',
                  scale_factor=0.025,
                  scale_mode='none',
                  vmin=0,
                  vmax=config.num_classes)
    mlab.show()
    mlab.savefig(filename, magnification=2)
    mlab.close()


if __name__ == '__main__':
    for room_idx in range(num_rooms):
        print('Visualization room {} ...'.format(room_idx))

        sub_set = list(filter(lambda x: '_{:02d}_'.format(room_idx) in x, test_set))
        target = label_dict[room_idx]
        pred = predict_dict[room_idx]

        pos, rgb = get_pos(dataset=sub_set, point_num=target.shape[0])

        save_path = os.path.join(config.root, 'visualization')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        filename = os.path.join(save_path, 'room_original_{:02d}.png'.format(room_idx))
        display_rgb(pos=pos, color=rgb, filename=filename)
        # filename = os.path.join(save_path, 'room_target_{:02d}.png'.format(room_idx))
        # display(pos=pos, label=target, filename=filename)
        # filename = os.path.join(save_path, 'room_pred_{:02d}.png'.format(room_idx))
        # display(pos=pos, label=pred, filename=filename)

    print('done.')
