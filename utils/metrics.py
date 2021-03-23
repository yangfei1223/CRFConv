# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class runningScore(object):
    def __init__(self, n_classes, ignore_index=-1):
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class) & (label_true != self.ignore_index)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        if label_trues.ndim == 1:
            self.confusion_matrix += self._fast_hist(label_trues, label_preds, self.n_classes)
        else:
            for lt, lp in zip(label_trues, label_preds):
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                'Overall Acc': acc,
                'Mean Acc': acc_cls,
                'FreqW Acc': fwavacc,
                'Mean IoU': mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class runningScoreShapeNet(object):
    def __init__(self):
        self.obj_classes = {
            'Airplane': 0,
            'Bag': 1,
            'Cap': 2,
            'Car': 3,
            'Chair': 4,
            'Earphone': 5,
            'Guitar': 6,
            'Knife': 7,
            'Lamp': 8,
            'Laptop': 9,
            'Motorbike': 10,
            'Mug': 11,
            'Pistol': 12,
            'Rocket': 13,
            'Skateboard': 14,
            'Table': 15,
        }
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29],
                            'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21],
                            'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
                            'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15],
                            'Knife': [22, 23]}
        self.category_IoU = np.array([0. for _ in range(16)], dtype=np.float32)
        self.category_num = np.array([0 for _ in range(16)], dtype=np.int32)

    def update(self, label_trues, label_preds, category):
        name = [k for (k, v) in self.obj_classes.items() if v == category][0]
        label = self.seg_classes[name]
        iu_part = 0.
        for l in label:
            locations_trues = (label_trues == l)
            locations_preds = (label_preds == l)
            i_locations = np.logical_and(locations_trues, locations_preds)
            u_locations = np.logical_or(locations_trues, locations_preds)
            i = np.sum(i_locations) + np.finfo(np.float32).eps
            u = np.sum(u_locations) + np.finfo(np.float32).eps
            iu_part += i/u
        iu = iu_part/(len(label))
        self.category_IoU[category] += iu
        self.category_num[category] += 1
        return iu

    def get_scores(self):
        pIoU = self.category_IoU.sum()/self.category_num.sum()
        per_class_pIoU = self.category_IoU/self.category_num
        mpIoU = per_class_pIoU.mean()
        cls_pIoU = {}
        for (k, v) in self.obj_classes.items():
            cls_pIoU[k] = per_class_pIoU[v]

        return (pIoU, mpIoU, cls_pIoU)


if __name__=='__main__':
    score = runningScoreShapeNet()
    print(score)