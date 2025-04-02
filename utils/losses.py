from typing import Sequence
from torch import Tensor
import torch
from torch.nn import Module
import torch.nn.functional as F
import torch.nn as nn
import cv2
from torch.distributions import MultivariateNormal


def kl_norm_loss(mu, sigma):
    """

    :param mu: BCK,D
    :param sigma: BCK,D
    :return:
    """
    kl = torch.mean(-0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2), dim=-1), dim=0)
    return kl

def kl_divergency(mu_p, sigma_p, mu_q, sigma_q):
    dist_p = MultivariateNormal(mu_p, sigma_p)
    dist_q = MultivariateNormal(mu_q, sigma_q)
    # Calculate the KL loss between the two distributions
    kl_loss = 1 - torch.distributions.kl_divergence(dist_p, dist_q).mean()
    return kl_loss


def asym_kl_mvn_loss(m0, S0, m1, S1):
    return (kl_divergency(m0, S0, m1, S1) + kl_divergency(m1, S1, m0, S0)) / 2


def l1_loss(a, b):
    return torch.mean(torch.abs(a - b))


def l2_loss(a, b):
    return torch.mean(torch.pow(a - b, 2))


def dice_loss(inputs, targets, softmax=False):
    if not softmax:
        inputs = F.softmax(inputs, dim=1)[:, 1]
    smooth = 1e-5

    # Flatten inputs and targets to calculate the intersection and union
    inputs = inputs.contiguous().view(-1)
    targets = targets.view(-1)

    intersection = torch.sum(inputs * targets)
    union = torch.sum(inputs) + torch.sum(targets)

    dice = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice

    return dice_loss


def ce_loss(inputs, targets):
    ce_loss = F.cross_entropy(inputs, targets)

    return ce_loss


def nll_loss(inputs, targets):
    loss = nn.NLLLoss()
    return loss(inputs, targets)


def intra_att_loss(matrix):
    """

    :param matrix: BCKK or CKK
    :return:
    """
    if len(matrix.shape) == 4:
        batch_size, c, n = matrix.size(0), matrix.size(1), matrix.size(2)
        # Calculate Frobenius norm of off-diagonal elements
    elif len(matrix.shape) == 3:
        c, n = matrix.size(0), matrix.size(1)
        batch_size = 1

    off_diag_norm = torch.norm(
        matrix * (1 - torch.eye(n).unsqueeze(0).unsqueeze(0).repeat(batch_size, c, 1, 1).cuda()),
        p='fro')

    # Calculate absolute differences between diagonal elements
    # diag_diff_avg = 0.0
    # num = 0
    # for i in range(batch_size):
    #     for j in range(c):
    #         diagonal_matrix_s = matrix[i][j]
    #         diag_elements = torch.diag(diagonal_matrix_s)
    #         diag_diff = torch.mean(torch.abs(diag_elements - diag_elements.unsqueeze(1)))
    #         diag_diff_avg += diag_diff
    #         num += 1
    # diag_diff_avg /= num
    # Combine the terms with appropriate weights
    loss = off_diag_norm
    return loss

def maskErosion(mask):
    """

    :param mask: B1HW
    :return:
    """
    mask = mask.clone().cpu().numpy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16))
    erode_mask = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        y = mask[i, 0]
        y = np.array(y, dtype='uint8')
        erosion = cv2.erode(y, kernel, 4)
        erode_mask[i] = erosion[np.newaxis]

    return erode_mask


def structure_loss(pred, mask, att='all'):
    """

    :param pred: BCHW
    :param mask: BHW
    :param att: 'interior' or 'boundary' or 'all'
    :return:
    """
    assert len(pred.shape) == 4 and len(mask.shape) == 3
    mask = mask.unsqueeze(1)

    if att == 'interior':
        weit = 1 + 3 * torch.from_numpy(maskErosion(mask)).cuda()
    elif att == 'boundary':
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    elif att == 'all':
        weit = torch.ones_like(mask).cuda()

    if pred.shape[1] == 1:
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wpixel = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

    elif pred.shape[1] == 2:
        input_softmax = pred[:, 1:2]
        inter = ((input_softmax * mask) * weit).sum(dim=(2, 3))
        union = ((input_softmax + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        wce = F.nll_loss(torch.log(pred), mask.squeeze(1).long(), reduction='none').unsqueeze(1)
        wpixel = (weit * wce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    return (wpixel + wiou).mean()

def preddivloss(clspred,mcspred,mask):
    assert len(clspred.shape) == 4 and len(mcspred.shape) == 4 and len(mask.shape) == 3
    mask = mask.unsqueeze(1)
    clspred = torch.sigmoid(clspred)
    mcspred = mcspred[:, 1:2]
    weit = 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    mse = F.mse_loss(clspred,mcspred,reduction='none')
    wmse = 1 - (weit * mse).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    return wmse.mean()





def wcedice_loss(inputs, targets, softmax=True):
    """

    :param inputs: BCHW after softmax
    :param targets: BHW (float)
    :param softmax:
    :return:
    """
    if len(targets.shape) == 3:
        targets = targets.unsqueeze(1)

    weit = 1 + 5 * torch.abs(F.avg_pool2d(targets, kernel_size=31, stride=1, padding=15) - targets)
    if not softmax:
        inputs = F.softmax(inputs, dim=1)[:, 1:2, ]
    smooth = 1e-5

    # Flatten inputs and targets to calculate the intersection and union
    # inputs = inputs.contiguous().view(-1)
    # targets = targets.view(-1)
    input_softmax = inputs[:, 1:2]
    intersection = ((input_softmax * targets) * weit).sum(dim=(2, 3))
    union = (input_softmax * weit).sum(dim=(2, 3)) + (targets * weit).sum(dim=(2, 3))
    wdice = 1 - (2. * intersection + smooth) / (union + smooth)

    wce = F.nll_loss(torch.log(inputs), targets.squeeze(1).long(), reduction='none').unsqueeze(1)
    wce = (weit * wce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    return (wce + wdice).mean()


def enforce_diversity(samples, penalty_factor):
    """

    :param samples: B,N,D,C,S,K
    :param penalty_factor:
    :return:
    """
    samples = torch.flatten(samples.permute(3, 4, 1, 0, 5, 2), start_dim=0, end_dim=1)  # C*S, N, B, K, D
    samples = F.normalize(samples,p=2,dim=-1)
    cls_num, num_samples = samples.shape[0], samples.shape[1]
    samples = samples.detach().cpu().numpy()
    distances = np.zeros((cls_num, num_samples, num_samples))

    # Calculate pairwise distances
    for c in range(cls_num):
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                distances[c, i, j] = np.mean(np.linalg.norm((samples[c, i] - samples[c, j]), axis=-1), axis=(-1, -2))
                distances[c, j, i] = distances[c, i, j]

    # Penalize samples with small distances
    # print(distances[0].shape)
    # print(distances[0])

    penalty_factor_matrx = penalty_factor*(1-np.eye(num_samples)[np.newaxis,:].repeat(cls_num,axis=0))
    # print((penalty_factor_matrx - distances)[0])
    diversity_loss = np.mean(np.maximum(0, penalty_factor_matrx - distances))

    return diversity_loss


import cv2 as cv
import numpy as np

import torch
from torch import nn

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve

"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf
"""


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.detach().cpu().numpy())).float().cuda()
        target_dt = torch.from_numpy(self.distance_field(target.detach().cpu().numpy())).float().cuda()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = (pred_error * distance).sqrt()
        loss = dt_field.mean()

        return loss


class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=3, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(
        self, pred: np.ndarray, target: np.ndarray, debug
    ) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        if debug:
            eroted, erosions = self.perform_erosion(
                pred.cpu().numpy(), target.cpu().numpy(), debug
            )
            return eroted.mean(), erosions

        else:
            eroted = torch.from_numpy(
                self.perform_erosion(pred.detach().cpu().numpy(), target.detach().cpu().numpy(), debug)
            ).float()

            loss = eroted.mean()

            return loss

class LogHausdorffDTLoss(HausdorffDTLoss):
    """
    Compute the logarithm of the Hausdorff Distance Transform Loss.

    This class computes the logarithm of the Hausdorff Distance Transform Loss, which is based on the distance transform.
    The logarithm is computed to potentially stabilize and scale the loss values, especially when the original loss
    values are very small.

    The formula for the loss is given by:
        log_loss = log(HausdorffDTLoss + 1)

    Inherits from the HausdorffDTLoss class to utilize its distance transform computation.
    """

    def forward(
            self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Compute the logarithm of the Hausdorff Distance Transform Loss.

        Args:
            input (torch.Tensor): The shape should be BNHW[D], where N is the number of classes.
            target (torch.Tensor): The shape should be BNHW[D] or B1HW[D], where N is the number of classes.

        Returns:
            torch.Tensor: The computed Log Hausdorff Distance Transform Loss for the given input and target.

        Raises:
            Any exceptions raised by the parent class HausdorffDTLoss.
        """
        log_loss: torch.Tensor = torch.log(super().forward(pred, target) + 1)
        return log_loss


from skimage import transform,measure
def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''

    out_img = np.zeros(mask.shape, dtype=np.uint8)

    for struc_id in [1]:

        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)

        props = measure.regionprops(blobs)

        if not props:
            continue

        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label

        out_img[blobs == largest_blob_label] = struc_id

    return out_img



