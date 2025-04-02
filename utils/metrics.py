import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve, distance_transform_edt as bwdist
import medpy.metric.binary as mmb



class cal_dice(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, y_pred, y_true):
        # smooth = 1
        smooth = 1e-5
        y_pred = y_pred > 0.5
        y_true = y_true > 0.5
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    def show(self):
        return np.mean(self.prediction)

class cal_assd(object):
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        pred = pred.squeeze() > 0.5
        gt = gt.squeeze() > 0.5
        if np.sum(pred)==0:
            pass
        else:
            score = self.cal(pred, gt)
            self.prediction.append(score)

    def cal(self, y_pred, y_true):
        return mmb.assd(y_pred,y_true)

    def show(self):
        return np.mean(self.prediction)

class cal_hd95(object):
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        pred = pred.squeeze() > 0.5
        gt = gt.squeeze() > 0.5
        if np.sum(pred)==0:
            pass
        else:
            score = self.cal(pred, gt)
            self.prediction.append(score)

    def cal(self, y_pred, y_true):
        return mmb.hd95(y_pred,y_true)

    def show(self):
        return np.mean(self.prediction)


class cal_iou(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, input, target):
        smooth = 1e-5
        input = input > 0.5
        target_ = target > 0.5
        intersection = (input & target_).sum()
        union = (input | target_).sum()

        return (intersection + smooth) / (union + smooth)
    def show(self):
        return np.mean(self.prediction)


class cal_wfm(object):
    def __init__(self, beta=1):
        self.beta = beta
        self.eps = 1e-6
        self.scores_list = []

    def update(self, pred, gt):
        assert pred.ndim == gt.ndim and pred.shape == gt.shape
        assert pred.max() <= 1 and pred.min() >= 0
        assert gt.max() <= 1 and gt.min() >= 0

        gt = gt > 0.5
        if gt.max() == 0:
            score = 0
        else:
            score = self.cal(pred, gt)
        self.scores_list.append(score)


    def cal(self, pred, gt):
        E = np.abs(pred - gt)
        dst, idst = bwdist(1 - gt, return_indices=True)

        K = fspecial_gauss(7, 5)
        Et = E.copy()
        Et[gt != 1] = Et[idst[:, gt != 1][0], idst[:, gt != 1][1]]
        EA = convolve(Et, K, mode='nearest')
        MIN_E_EA = E.copy()
        MIN_E_EA[(gt == 1) & (EA < E)] = EA[(gt == 1) & (EA < E)]

        B = np.ones_like(gt)
        B[gt != 1] = 2.0 - 1 * np.exp(np.log(1 - 0.5) / 5 * dst[gt != 1])
        Ew = MIN_E_EA * B

        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt != 1])

        R = 1 - np.mean(Ew[gt == 1])
        P = TPw / (TPw + FPw + np.finfo(np.float64).eps)
        Q = 2 * R * P / (R + P + np.finfo(np.float64).eps)

        return Q

    def show(self):
        return np.mean(self.scores_list)


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

