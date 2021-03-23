# -*- coding:utf-8 -*-
import os, sys, time
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_geometric.transforms import Compose, SamplePoints, NormalizeScale, Center, FixedPoints, RandomRotate
from torch_points3d.core.data_transform import *
from datasets import Semantic3DWholeDataset
from utils import runningScore, read_ply, write_ply
import models
from configure import *

logging.getLogger().setLevel(logging.INFO)


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.cuda.set_device(cfg.device_id)

        # The room version dataset
        self.train_transform = Compose([
            RandomRotate(degrees=180, axis=2),
            RandomScaleAnisotropic(scales=[0.8, 1.2], anisotropic=True),
            RandomSymmetry(axis=[True, False, False]),
            RandomNoise(sigma=0.001),
            DropFeature(drop_proba=0.2, feature_name='rgb'),
            AddFeatsByKeys(list_add_to_x=[True, True],
                           feat_names=['pos', 'rgb'],
                           delete_feats=[False, True])

        ])
        self.test_transform = Compose([
            AddFeatsByKeys(list_add_to_x=[True, True],
                           feat_names=['pos', 'rgb'],
                           delete_feats=[False, True])
        ])

        self.dataset = Semantic3DWholeDataset(root=cfg.root,
                                              grid_size=cfg.grid_size,
                                              num_points=cfg.sample_num,
                                              train_sample_per_epoch=cfg.train_samples_per_epoch,
                                              test_sample_per_epoch=cfg.test_samples_per_epoch,
                                              train_transform=self.train_transform,
                                              test_transform=self.test_transform)

        self.dataset.create_dataloader(batch_size=cfg.batch_size,
                                       shuffle=True,
                                       num_workers=0,
                                       precompute_multi_scale=True,
                                       num_scales=5)

        self.test_probs = [np.zeros(shape=(t.data.shape[0], cfg.num_classes), dtype=np.float32)
                           for t in self.dataset.val_set.input_trees]

        self.model = getattr(models, cfg.model_name)(in_channels=6,
                                                     n_classes=cfg.num_classes,
                                                     use_crf=cfg.use_crf,
                                                     steps=cfg.steps)

        # self.optimizer = torch.optim.Adam(params=self.model.parameters(),
        #                                   lr=cfg.lr,
        #                                   weight_decay=cfg.weight_decay)
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                         lr=cfg.lr,
                                         momentum=cfg.momentum,
                                         weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.gamma)
        self.metrics = runningScore(cfg.num_classes, ignore_index=cfg.ignore_index)

    @staticmethod
    def _iou_from_confusions(confusions):
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TPFN = np.sum(confusions, axis=-1)
        TPFP = np.sum(confusions, axis=-2)

        IoU = TP / (TPFP + TPFN - TP + 1e-6)

        mask = TPFN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        IoU += mask * mIoU

        return IoU

    def train_one_epoch(self, epoch):
        self.model.train()
        self.metrics.reset()
        with Ctq(self.dataset.train_loader) as tq_loader:
            for i, data in enumerate(tq_loader):
                tq_loader.set_description('Train epoch[{}]'.format(epoch))
                data = data.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(data)
                y_target = data.y.reshape(-1) - 1
                loss = F.cross_entropy(y_pred, y_target,
                                       weight=self.cfg.class_weights.to(self.device),
                                       ignore_index=self.cfg.ignore_index)
                loss.backward()
                self.optimizer.step()
                tq_loader.set_postfix(loss=loss.item())
                self.metrics.update(y_target.cpu().numpy(), y_pred.max(dim=1)[1].cpu().numpy())

    def val_one_epoch(self, epoch):
        self.model.eval()
        self.metrics.reset()
        with Ctq(self.dataset.val_loader) as tq_loader:
            for i, data in enumerate(tq_loader):
                tq_loader.set_description('Val epoch[{}]'.format(epoch))
                data = data.to(self.device)
                y_target = data.y.reshape(-1) - 1
                with torch.no_grad():
                    y_pred = self.model(data)
                loss = F.cross_entropy(y_pred, y_target,
                                       weight=self.cfg.class_weights.to(self.device),
                                       ignore_index=self.cfg.ignore_index)
                tq_loader.set_postfix(loss=loss.item())
                self.metrics.update(y_target.cpu().numpy(), y_pred.max(dim=1)[1].cpu().numpy())

    def train(self):
        best_iu = 0
        # track parameters
        self.model.to(self.device)
        for epoch in range(self.cfg.epochs):
            logging.info('Training epoch: {}, learning rate: {}'.format(epoch, self.scheduler.get_last_lr()[0]))

            # training
            t1 = time.time()
            self.train_one_epoch(epoch)
            t2 = time.time()
            print("Average training time is {:.2f}".format((t2-t1) * 1000 / 100))
            score_dict, _ = self.metrics.get_scores()
            logging.info('Training OA: {:.2f} %, mIoU: {:.2f} %'.format(score_dict['Overall Acc'] * 100,
                                                                        score_dict['Mean IoU'] * 100))
            # validation
            t1 = time.time()
            self.val_one_epoch(epoch)
            t2 = time.time()
            print("Average test time is {:.2f}".format((t2-t1) * 1000 / 100))
            score_dict, _ = self.metrics.get_scores()
            logging.info('Test OA: {:.2f} %, mIoU: {:.2f} %'.format(score_dict['Overall Acc'] * 100,
                                                                    score_dict['Mean IoU'] * 100))
            # if best_iu <= score_dict['Mean IoU']:
            #     best_iu = score_dict['Mean IoU']
            #     self.model.save(self.cfg.model_path)     # save model
            #     logging.info('Save {} succeed, best mIoU: {:.2f} % !'.format(self.cfg.model_path, best_iu * 100))

            self.scheduler.step()
        logging.info('Training finished, best mean IoU: {:.2f} %'.format(best_iu * 100))

    def test(self, num_votes=100):
        logging.info('Test {} on {} ...'.format(self.cfg.model_name, self.cfg.dataset))
        test_smooth = 0.98
        saving_path = 'results/Semantic3D/predictions'
        os.makedirs(saving_path) if not os.path.exists(saving_path) else None

        # load model checkpoints
        self.model.load('checkpoints/PointConvBig_on_Semantic3D_bs_8_epochs_100_big_crf.ckpt')
        self.model.to(self.device)
        self.model.eval()

        epoch = 0
        last_min = -0.5
        while last_min < num_votes:
            # test one epoch
            with Ctq(self.dataset.val_loader) as tq_loader:
                for i, data in enumerate(tq_loader):
                    tq_loader.set_description('Evaluation')
                    # model inference
                    data = data.to(self.device)
                    with torch.no_grad():
                        probs = F.softmax(self.model(data), dim=-1)      # get pred probs

                    # running means for each epoch on Test set
                    point_idx = data.point_idx.cpu().numpy()    # the point idx
                    cloud_idx = data.cloud_idx.cpu().numpy()    # the cloud idx
                    probs = probs.reshape(self.cfg.batch_size, -1, self.cfg.num_classes).cpu().numpy()  # [B, N, C]
                    for b in range(self.cfg.batch_size):        # for each sample in batch
                        prob = probs[b, :, :]       # [N, C]
                        p_idx = point_idx[b, :]     # [N]
                        c_idx = cloud_idx[b][0]     # int
                        self.test_probs[c_idx][p_idx] = test_smooth * self.test_probs[c_idx][p_idx] \
                                                        + (1 - test_smooth) * prob  # running means

            # after each epoch
            new_min = np.min(self.dataset.val_set.min_possibility)
            print('Epoch {:3d} end, current min possibility = {:.2f}'.format(epoch, new_min))
            if last_min + 4 < new_min:
                print('Test procedure done, saving predicted clouds ...')
                last_min = new_min
                # projection prediction to original point cloud
                t1 = time.time()
                for i, file in enumerate(self.dataset.val_set.val_files):
                    proj_idx = self.dataset.val_set.test_proj[i]       # already get the shape
                    probs = self.test_probs[i][proj_idx, :]             # same shape with proj_idx
                    # [0 ~ 7] + 1 -> [1 ~ 8], because 0 for unlabeled
                    preds = np.argmax(probs, axis=1).astype(np.uint8) + 1
                    # saving prediction results
                    cloud_name = file.split('/')[-1]
                    # ascii_name = os.path.join(saving_path, self.dataset.test_set.ascii_files[cloud_name])
                    # np.savetxt(ascii_name, preds, fmt='%d')
                    # print('Save {:s} succeed !'.format(ascii_name))
                    filename = os.path.join(saving_path, cloud_name)
                    write_ply(filename, [preds], ['pred'])
                    print('Save {:s} succeed !'.format(filename))
                t2 = time.time()
                print('Done in {:.2f} s.'.format(t2-t1))
                return
            epoch += 1
        return

    def test_s3dis(self, num_votes=100):
        logging.info('Evaluating {} on {} ...'.format(self.cfg.model_name, self.cfg.dataset))
        test_smooth = 0.95
        # statistic label proportions in test set
        class_proportions = np.zeros(self.cfg.num_classes, dtype=np.float32)
        for i, label in enumerate(self.dataset.test_set.label_values):
            class_proportions[i] = np.sum([np.sum(labels == label) for labels in self.dataset.test_set.val_labels])

        # load model checkpoints
        self.model.load('checkpoints/RandLANet_on_S3DIS_bs_8_epochs_100_big.ckpt')
        self.model.to(self.device)
        self.model.eval()

        epoch = 0
        last_min = -0.5
        while last_min < num_votes:

            # test one epoch
            with Ctq(self.dataset.test_loader) as tq_loader:
                for i, data in enumerate(tq_loader):
                    tq_loader.set_description('Evaluation')

                    # model inference
                    data = data.to(self.device)
                    with torch.no_grad():
                        logits = self.model(data)                     # get pred
                        y_pred = F.softmax(logits, dim=-1)

                    y_pred = y_pred.cpu().numpy()
                    y_target = data.y.cpu().numpy().reshape(-1)       # get target
                    point_idx = data.point_idx.cpu().numpy()          # the point idx
                    cloud_idx = data.cloud_idx.cpu().numpy()          # the cloud idx

                    # compute batch accuracy
                    correct = np.sum(np.argmax(y_pred, axis=1) == y_target)
                    acc = correct / float(np.prod(np.shape(y_target)))      # accurate for each test batch
                    tq_loader.set_postfix(ACC=acc)

                    y_pred = y_pred.reshape(self.cfg.batch_size, -1, self.cfg.num_classes)      # [B, N, C]
                    for b in range(self.cfg.batch_size):        # for each sample in batch
                        probs = y_pred[b, :, :]     # [N, C]
                        p_idx = point_idx[b, :]     # [N]
                        c_idx = cloud_idx[b][0]       # int
                        self.test_probs[c_idx][p_idx] = test_smooth * self.test_probs[c_idx][p_idx] \
                                                       + (1 - test_smooth) * probs   # running means

            new_min = np.min(self.dataset.test_set.min_possibility)
            print('Epoch {:3d} end, current min possibility = {:.2f}'.format(epoch, new_min))

            if last_min + 1 < new_min:
                # update last_min
                last_min += 1
                # show vote results
                print('Confusion on sub clouds.')
                confusion_list = []
                num_clouds = len(self.dataset.test_set.input_labels)        # test cloud number
                for i in range(num_clouds):
                    probs = self.test_probs[i]
                    preds = self.dataset.test_set.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                    labels = self.dataset.test_set.input_labels[i]
                    confusion_list += [confusion_matrix(labels, preds, self.dataset.test_set.label_values)]

                # re-group confusions
                C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)
                # re-scale with the right number of point per class
                C *= np.expand_dims(class_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                # compute IoU
                IoUs = self._iou_from_confusions(C)
                m_IoU = np.mean(IoUs)
                s = '{:5.2f} | '.format(100 * m_IoU)
                for IoU in IoUs:
                    s += '{:5.2f} '.format(100 * IoU)
                print(s)

                if int(np.ceil(new_min) % 1) == 0:      # ???
                    print('re-project vote #{:d}'.format(int(np.floor(new_min))))
                    proj_prob_list = []

                    for i in range(num_clouds):
                        proj_idx = self.dataset.test_set.val_proj[i]
                        probs = self.test_probs[i][proj_idx, :]
                        proj_prob_list += [probs]

                    # show vote results
                    print('confusion on full cloud')
                    confusion_list = []
                    for i in range(num_clouds):
                        preds = self.dataset.test_set.label_values[np.argmax(proj_prob_list[i], axis=1)].astype(np.uint8)
                        labels = self.dataset.test_set.val_labels[i]
                        acc = np.sum(preds == labels) / len(labels)
                        print(self.dataset.test_set.input_names[i] + 'ACC:' + str(acc))
                        confusion_list += [confusion_matrix(labels, preds, self.dataset.test_set.label_values)]
                        # name = self.dataset.test_set.label_values + '.ply'
                        # write_ply(join(path, 'val_preds', name), [preds, labels], ['pred', 'label'])

                    # re-group confusions
                    C = np.sum(np.stack(confusion_list), axis=0)
                    IoUs = self._iou_from_confusions(C)
                    m_IoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * m_IoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)

                    print(s)
                    print('finished.')
                    return
            epoch += 1
            continue
        return

    def __call__(self, *args, **kwargs):
        if self.cfg.mode is 'train':
            self.train()
        elif self.cfg.mode is 'test':
            self.test()
        else:
            raise ValueError('Only `train` or `test` mode is supported!')

    def __repr__(self):
        return 'Training {} on {}, num_classes={}, batch_size={}, epochs={}, use_crf={}'.format(self.cfg.model_name,
                                                                                                self.cfg.dataset,
                                                                                                self.cfg.num_classes,
                                                                                                self.cfg.batch_size,
                                                                                                self.cfg.epochs,
                                                                                                self.cfg.use_crf)


if __name__ == '__main__':
    trainer = Trainer(cfg=Semantic3DConfig())
    print(trainer)
    print(trainer.model)
    trainer()
