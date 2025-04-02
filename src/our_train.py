import os
import sys
import cv2
import datetime
import argparse
import numpy as np
import ctypes
import logging
import random
import torch.backends.cudnn as cudnn
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import ttach as tta
import json
import time

libgcc_s = ctypes.CDLL('libgcc_s.so.1')
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

import apex
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from our_model import Model
from utils.metrics import cal_dice, cal_iou
from utils.losses import structure_loss, intra_att_loss, keep_largest_connected_components, preddivloss
from utils.dataset import ISIC2018Data, ISIC2017Data
from utils.polyp_dataloader import get_loader, get_test_loader

import math
import torch.optim as optim


class CustomCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min_values):
        self.T_max = T_max
        self.eta_min_values = eta_min_values
        super(CustomCosineAnnealingLR, self).__init__(optimizer)

    def get_lr(self):
        etas = []
        for param_group, eta_min in zip(self.optimizer.param_groups, self.eta_min_values):
            t_cur = self.last_epoch
            eta_min = eta_min
            eta_max = param_group['initial_lr']
            eta = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t_cur / self.T_max))
            etas.append(eta)
        return etas


transforms = tta.Compose(
    [
        tta.Scale(scales=[1, 0.5], interpolation='bilinear', align_corners=False),
    ]
)


def cv_squared(x):
    """The squared coefficient of variation of a sample.
    Useful as a loss to encourage a positive distribution to be more uniform.
    Epsilons added for numerical stability.
    Returns 0 for an empty Tensor.
    Args:
    x: a `Tensor`.
    Returns:
    a `Scalar`.
    """
    eps = 1e-10
    # if only num_experts = 1
    if x.shape[0] == 1:
        return torch.tensor([0], device=x.device, dtype=x.dtype)
    return x.float().var() / (x.float().mean() ** 2 + eps)


def test(model, testloader, args, warmup=False):
    m_iou = cal_iou()
    m_dice = cal_dice()
    with torch.no_grad():
        for image, mask, shape, name, origin in testloader:
            image = image.cuda().float()
            preds = []
            if warmup:
                if args.tta:
                    for transformer in transforms:
                        rgb_trans = transformer.augment_image(image)
                        model_output = model.warmup(rgb_trans)
                        deaug_mask = transformer.deaugment_mask(model_output)
                        preds.append(deaug_mask)
                    pred = torch.mean(torch.stack(preds, dim=0), dim=0)
                else:
                    pred = model.warmup(image)
                if pred.shape[2] != shape[0]:
                    pred = F.interpolate(pred, size=shape, mode='bilinear', align_corners=True)[0, 0]
                pred = torch.sigmoid(pred)
            else:
                if args.tta:
                    for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()
                        rgb_trans = transformer.augment_image(image)
                        model_output = model(rgb_trans)['fuse_prob']
                        deaug_mask = transformer.deaugment_mask(model_output)
                        preds.append(deaug_mask)
                    pred = torch.mean(torch.stack(preds, dim=0), dim=0)
                else:
                    output = model(image)
                    pred = output['fuse_prob']
                if pred.shape[2] != shape[0]:
                    pred = F.interpolate(pred, size=shape, mode='bilinear', align_corners=True)[0, 0]
                pred = torch.sigmoid(pred)
            pred = pred.cpu().numpy()
            if args.postprecessing:
                pred = keep_largest_connected_components(pred)
            mask = mask.cpu().numpy()
            m_dice.update(pred, mask)
            m_iou.update(pred, mask)
    m_dice = m_dice.show()
    m_iou = m_iou.show()
    return m_dice, m_iou


def test_loss(model, testloader, warmup=False):
    loss_list = []
    with torch.no_grad():
        for image, mask, shape, name, origin in testloader:
            image = image.cuda().float()
            mask = mask.cuda().float()
            if warmup:
                clspred = model.warmup(image)
                pred = F.interpolate(clspred, size=shape, mode='bilinear', align_corners=True)
                loss = structure_loss(pred, mask, att='boundary')
            else:
                output = model(image)
                fuse_pred = output['fuse_prob']
                fuse_pred = F.interpolate(fuse_pred, size=shape, mode='bilinear', align_corners=True)
                loss = structure_loss(fuse_pred, mask, att='all')
            loss_list.append(loss.item())
    log_info = f'loss: {np.mean(loss_list):.4f}'
    print(log_info)
    return np.mean(loss_list)


def test_poylp(model, testloader, warmup=False):
    m_iou = cal_iou()
    m_dice = cal_dice()
    with torch.no_grad():
        for image, mask, name in testloader:
            image = image.cuda().float()
            mask = mask.cuda().float().squeeze(1)
            orisize = mask.shape[1:]
            if warmup:
                clspred = model.warmup(image)
                pred = F.interpolate(clspred, size=orisize, mode='bilinear', align_corners=True)[0, 0]
                pred = torch.sigmoid(pred)
            else:
                output = model(image)
                fuse_pred = output['fuse_prob']
                fuse_pred = F.interpolate(fuse_pred, size=orisize, mode='bilinear', align_corners=True)[0, 0]
                pred = torch.sigmoid(fuse_pred)
            mask = mask.cpu().numpy()
            pred = pred.cpu().numpy()
            m_dice.update(pred, mask)
            m_iou.update(pred, mask)
    m_dice = m_dice.show()
    m_iou = m_iou.show()
    return m_dice, m_iou


class Train(object):
    def __init__(self, trainloader, Model, args):
        ## dataset
        self.args = args
        self.loader = trainloader
        ## model
        self.model = Model(args)
        self.model.train(True)
        self.model.cuda()
        ## parameter
        base, head = [], []
        for name, param in self.model.named_parameters():
            if 'bkbone' in name:
                base.append(param)
            else:
                head.append(param)
        if args.opt == 'SGD':
            self.optimizer = torch.optim.SGD(
                [{'params': base, 'lr': args.bkbonelr}, {'params': head, 'lr': args.headlr}],
                momentum=args.momentum, weight_decay=args.weight_decay,
                nesterov=args.nesterov)
        elif args.opt == 'Adam':
            self.optimizer = torch.optim.AdamW(
                [{'params': base, 'lr': args.bkbonelr}, {'params': head, 'lr': args.headlr}],
                weight_decay=args.weight_decay)

        self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level='O2')

        self.scheduler = CustomCosineAnnealingLR(self.optimizer, T_max=args.T_max, eta_min_values=args.eta_min_values)

        self.snapshot_path = '{}/{}'.format(args.savepath, args.exp)

        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)
        logging.basicConfig(filename=self.snapshot_path + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

    def train(self):
        logging.info("{} iterations per epoch".format(len(self.loader)))
        global_step = 0
        best_iou = 0.0
        min_loss = float(np.inf)
        iterator = tqdm(range(self.args.epoch), ncols=70)
        size_rates = [0.75, 1, 1.25]
        for epoch in iterator:
            logging.info('epoch: {} bkbonelr: {:.4} headlr: {}'.format(epoch, self.optimizer.param_groups[0]['lr'],
                                                                       self.optimizer.param_groups[1]['lr']))
            self.scheduler.step()
            for batchidx, (image, mask) in enumerate(self.loader):
                for rate in size_rates:
                    global_step += 1
                    log_dict = {'iterNum': global_step, 'bkbonelr': self.optimizer.param_groups[0]['lr'],
                                'headlr': self.optimizer.param_groups[1]['lr']}

                    image, mask = image.cuda().float(), mask.cuda().float().squeeze(1)
                    # ---- rescale ----
                    trainsize = int(round(args.patchsize * rate / 32) * 32)
                    if rate != 1:
                        image = F.upsample(image, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                        mask = F.upsample(mask.unsqueeze(1), size=(trainsize, trainsize), mode='bilinear',
                                          align_corners=True).squeeze(1)

                    if global_step <= args.preIter:
                        clspred = self.model.warmup(image)

                        pred = F.interpolate(clspred, size=mask.shape[1:], mode='bilinear', align_corners=True)
                        loss = structure_loss(pred, mask, att='boundary')
                        logging.info(
                            'iteration:{}   loss: {}  '.format(global_step, loss.item()))

                    else:
                        output = self.model(image, mask)

                        att_m = output['att_m']  # B,D,C,S,K
                        att_v = output['att_v']  # B,D,C,S,K
                        att_m_mean = torch.mean(att_m, dim=[0, 1, 3])
                        att_v_mean = torch.mean(att_v, dim=[0, 1, 3])

                        log_dict['mean/cls0k0_mean'] = att_m_mean[0, 0]
                        log_dict['mean/cls0k1_mean'] = att_m_mean[0, 1]
                        log_dict['mean/cls0k2_mean'] = att_m_mean[0, 2]
                        log_dict['var/cls0k0_var'] = att_v_mean[0, 0]
                        log_dict['var/cls0k1_var'] = att_v_mean[0, 1]
                        log_dict['var/cls0k2_var'] = att_v_mean[0, 2]

                        if self.args.num_classes == 2:
                            log_dict['mean/cls1k0_mean'] = att_m_mean[1, 0]
                            log_dict['mean/cls1k1_mean'] = att_m_mean[1, 1]
                            log_dict['mean/cls1k2_mean'] = att_m_mean[1, 2]
                            log_dict['var/cls1k0_var'] = att_v_mean[1, 0]
                            log_dict['var/cls1k1_var'] = att_v_mean[1, 1]
                            log_dict['var/cls1k2_var'] = att_v_mean[1, 2]

                        if global_step == 1:
                            print(att_m.shape)

                        # Constraint 1: Intra-Class attribute representation independence
                        if args.intra_div != 0.0:
                            if args.decorrtype == 'attm':
                                att_m = torch.flatten(att_m, start_dim=2, end_dim=3)
                                gram_matrix = torch.einsum(
                                    "bdck,bdcq->bckq", att_m, att_m
                                )
                            elif args.decorrtype == 'queryfeat':
                                query_feat = self.model.mask2former.query_feat  #
                                feat_corr = torch.cat([module.weight.unsqueeze(0) for module in query_feat], dim=0)
                                gram_matrix = torch.einsum(
                                    "ckd,cqd->ckq", feat_corr, feat_corr
                                )
                            loss_intra_div = intra_att_loss(gram_matrix)
                            log_dict['loss/loss_intra_div'] = loss_intra_div.item()
                        else:
                            loss_intra_div = torch.tensor([0.0]).cuda()

                        # MoE
                        route_prob = output['route_prob']  # BK
                        topk_mask = output['topk_mask']  # BK
                        log_dict['pred/route_prob_max'] = torch.max(route_prob)
                        log_dict['pred/route_prob_min'] = torch.min(route_prob)
                        load_balancing_loss = cv_squared(topk_mask.sum(0))
                        log_dict['loss/load_balancing_loss'] = load_balancing_loss.item()

                        pred = output['cls_prob']
                        pred = F.interpolate(pred, size=mask.shape[1:], mode='bilinear', align_corners=True)
                        loss_cls = structure_loss(pred, mask, att='interior')
                        log_dict['loss/loss_cls'] = loss_cls.item()

                        mcs_pred = output['mcs_prob']
                        if args.num_classes == 1:
                            mcs_pred = mcs_pred[:, 0]
                            loss_mcs = structure_loss(mcs_pred, mask, att='boundary')
                        else:
                            loss_mcs = structure_loss(mcs_pred, mask, att='boundary')
                        log_dict['loss/loss_mcs'] = loss_mcs.item()

                        fuse_pred = output['fuse_prob']
                        loss_fuse = structure_loss(fuse_pred, mask, att='boundary')
                        log_dict['loss/loss_fuse'] = loss_fuse.item()

                        loss = args.clsw * loss_cls + args.mcsw * loss_mcs + args.fusew * loss_fuse + \
                               args.intra_div * loss_intra_div + args.loadbal * load_balancing_loss
                        max_probs, max_idx = torch.max(mcs_pred, dim=1)
                        a_soft_max = max_probs.detach().clone().cpu().numpy()
                        a_soft_max_cpu_30 = np.percentile(a_soft_max, 30)
                        a_soft_max_cpu_95 = np.percentile(a_soft_max, 95)
                        log_dict['pred/pred_prob_30'] = a_soft_max_cpu_30
                        log_dict['pred/pred_prob_95'] = a_soft_max_cpu_95

                        logging.info(
                            'iteration:{}   losscls: {}    lossmcs: {}   lossfuse: {}   lossdecorr: {}  lossloadbal: {}  '.format(
                                global_step,
                                loss_cls.item(),
                                loss_mcs.item(),
                                loss_fuse.item(),
                                loss_intra_div.item(),
                                load_balancing_loss.item(),
                            ))

                    self.optimizer.zero_grad()
                    with apex.amp.scale_loss(loss, self.optimizer) as scale_loss:
                        scale_loss.backward()
                    self.optimizer.step()

                    wandb.log(log_dict)
                    torch.cuda.empty_cache()

                    # if args.eval and global_step >= args.preIter and global_step % args.evalIter == 0:
                    if args.eval and global_step % args.evalIter == 0:
                        with torch.no_grad():
                            self.model.train(False)
                            if args.dataset == 'polyp':
                                mdice, miou = test_poylp(self.model, testloader=test_loader,
                                                         warmup=global_step <= self.args.preIter)
                            else:
                                mdice, miou = test(self.model, testloader=test_loader, args=args,
                                                   warmup=global_step <= self.args.preIter)
                            logging.info(
                                'iterNum: {}, mdice: {}   miou:{}'.format(global_step, mdice, miou))
                            if miou > best_iou:
                                bestEpoch = epoch
                                best_iou = miou
                                save_best_path = os.path.join(self.snapshot_path, 'best_model.pth')
                                torch.save(self.model.state_dict(), save_best_path)
                                save_path = os.path.join(self.snapshot_path,
                                                         'iter_{}_iou_{}.pth'.format(global_step, best_iou))
                                torch.save(self.model.state_dict(), save_path)

                            wandb.log({"dice/": mdice})
                            wandb.log({"iou/": miou})
                            wandb.log({'iou/best_iou': best_iou})
                            self.model.train(True)

            if (epoch + 1) % (self.args.epoch // 10) == 0:
                torch.save(self.model.state_dict(), self.snapshot_path + '/model-' + str(epoch + 1))

        logging.info("best epoch is: {}".format(bestEpoch))
        logging.info("best IoU is: {}".format(best_iou))
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ########### training configuration ######################
    parser.add_argument('--seed', type=int, default=1337, help='random seed')  # 1337
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--savepath', type=str)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--dataset', default='isic2017')  # gland  isic2017
    parser.add_argument('--datapath', tyep=str)  
    parser.add_argument('--exp', type=str, help='experiment_name') 
    parser.add_argument('--patchsize', type=int, default=352, help='patchsize for training set and test set')
    parser.add_argument('--bkbonelr', default=1e-4)  # 3e-4/4e-3/1e-4
    parser.add_argument('--headlr', default=1e-4)
    parser.add_argument('--epoch', default=100)
    parser.add_argument("--preIter", type=int, default=1000,
                        help="number of iteration to pretrain using linear head only")
    parser.add_argument('--backbone', default='pvt')  # resNet/pvt
    parser.add_argument('--batch_size', default=16)  # 16
    parser.add_argument('--weight_decay', default=5e-4)
    parser.add_argument('--opt', default='Adam')  # Adam(4e-4)/SGD(0.04)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--nesterov', default=True)
    parser.add_argument('--snapshot', default=None)

    ########### network configuration ####################### 
    parser.add_argument('--shareW', type=bool, default=True,
                        help='whether to share query weights between different images')
    parser.add_argument('--cls_attributes', type=int, default=10, help='number of attributes for each class')
    parser.add_argument('--topk', type=int, default=3, help='top k experts')
    parser.add_argument('--num_classes', type=int, default=2, help='number of class considered in mcs head')
    parser.add_argument('--num_queries', type=int, default=2,
                        help='number of subcategories, must be the multiple of num_classes')
    parser.add_argument('--dec_layers', type=int, default=9,
                        help='number of pmm layers, must be the multiple of 3')
    parser.add_argument('--tau', type=float, default=1, help='temperature hyperparamter for the pixel-class matching')
    parser.add_argument('--mask_dim', type=int, default=256, help='dimension of final generated feature')
    parser.add_argument('--sample_num', type=int, default=15, help='number of MCS')
    parser.add_argument('--fuse_mode', type=str, default='conv', help='mode to fuse cls and mcs head')

    ########### loss weight ######################
    parser.add_argument('--clsw', type=float, default=1, help='weight of cls_head loss')
    parser.add_argument('--mcsw', type=float, default=2, help='weight of mcs_head loss')
    parser.add_argument('--fusew', type=float, default=2, help='weight of fuse head')
    parser.add_argument('--intra_div', type=float, default=1, help='weight of intra_div loss')
    parser.add_argument('--loadbal', type=float, default=1, help='weight of load balancing loss')
    parser.add_argument('--decorrtype', type=str, default='queryfeat',
                        help='conduct decorrelation on attm or queryfeat?')  # attm/queryfeat

    ########### lr decay ######################
    parser.add_argument('--T_max', type=int, help='decay cycle')
    parser.add_argument('--eta_min_values', type=list, default=[0, 0], help='minimum lr when decay')

    ########### evaluation ######################
    parser.add_argument("--eval", type=bool, default=True, help="whether to evaluate")
    parser.add_argument("--evalIter", type=int, default=500, help="number of iteration to eval")
    parser.add_argument("--tta", type=bool, default=False, help="whether to use test time augmentation")
    parser.add_argument("--postprecessing", type=bool, default=False,
                        help="whether to keep the largest connected area only")

    parser.add_argument('--device', type=str, help='the device')
    parser.add_argument('--proj', type=str, default='TMImajor', help='wandb project name')
    parser.add_argument('--wandb', type=bool, default=False, help='whether use wandb')

    args = parser.parse_args()
    args.T_max = args.epoch

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if not args.wandb:
        # If you don't want your script to sync to the cloud
        os.environ['WANDB_MODE'] = 'dryrun'

    if args.exp != 'test':
        args.exp = args.exp + 'backbone{}_bkbonelr{}_headlr{}_bs{}_clsw{}_mcsw{}_fusew{}_att{}_topk{}_maskdim{}_temp{}_Tmax{}_opt{}'.format(
            args.backbone,
            args.bkbonelr,
            args.headlr,
            args.batch_size,
            args.clsw,
            args.mcsw,
            args.fusew,
            args.cls_attributes,
            args.topk,
            args.mask_dim,
            args.tau,
            args.T_max,
            args.opt
        )

    if args.dataset == 'isic2018':  
        traindata = ISIC2018Data(datapath=args.datapath, split='train', patchsize=args.patchsize)
        train_loader = DataLoader(traindata, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=4)
        testdata = ISIC2018Data(datapath=args.datapath, split='valid', patchsize=args.patchsize)
        test_loader = DataLoader(testdata, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    elif args.dataset == 'isic2017':
        traindata = ISIC2017Data(datapath=args.datapath, split='train', patchsize=args.patchsize)
        train_loader = DataLoader(traindata, batch_size=args.batch_size, pin_memory=True, shuffle=False,
                                  num_workers=4)
        testdata = ISIC2017Data(datapath=args.datapath, split='valid', patchsize=args.patchsize)
        test_loader = DataLoader(testdata, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    elif args.dataset == 'polyp':
        image_root = args.datapath + '/TrainDataset/images/'
        gt_root = args.datapath + '/TrainDataset/masks/'
        test_image_root = args.datapath + '/TestDataset/test/images/'
        test_gt_root = args.datapath + '/TestDataset/test/masks/'
        train_loader = get_loader(image_root=image_root, gt_root=gt_root, batchsize=args.batch_size,
                                  trainsize=args.patchsize, augmentation=True)
        test_loader = get_test_loader(image_root=test_image_root, gt_root=test_gt_root, batchsize=1,
                                      testsize=args.patchsize)

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    wandb.init(project=args.proj,
               name=args.exp,
               entity='isyuanyc')
    t = Train(train_loader, Model, args)
    t.train()