import os
import sys
import cv2
import argparse
import numpy as np
import ctypes
import random
import torch.backends.cudnn as cudnn
import json
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from our_model import Model
from utils.dataset import ISIC2018Data, ISIC2017Data
from utils.polyp_dataloader import get_test_loader
from utils.metrics import cal_dice, cal_iou, cal_hd95, cal_assd
from utils.losses import keep_largest_connected_components
import wandb
from PIL import Image
import ttach as tta
from tqdm import tqdm
from collections import defaultdict

transforms = tta.Compose(
    [
        tta.Scale(scales=[1, 0.5], interpolation='bilinear', align_corners=False),
    ]
)


def visualcase(name, image, gt, pred, log_dict, exp, head):
    examples = []
    save_img = image.clone().permute(0, 2, 3, 1).cpu().numpy()[0]
    save_img = (save_img - np.min(save_img)) / (np.max(save_img) - np.min(save_img))
    save_img *= 255.
    save_img = Image.fromarray(np.uint8(save_img))
    # save_img.save("img.jpg", mode="RGB")
    image = wandb.Image(save_img, caption=name)
    examples.append(image)

    mask = gt.detach().cpu().numpy()
    mask = mask[0] * 255.
    mask = Image.fromarray(np.uint8(mask))
    # mask.save("gt.jpg")
    image = wandb.Image(mask, caption=f"gt")
    examples.append(image)

    pred = (pred > 0.5) * 255.
    pred = Image.fromarray(np.uint8(pred))
    # pred.save("pred.jpg")
    image = wandb.Image(pred, caption=head)
    examples.append(image)

    log_dict[exp] = examples

classified_images = defaultdict(list)

class Test(object):
    def __init__(self, test_loader, Model, args):
        ## dataset
        self.args = args
        self.loader = test_loader
        ## model
        self.model = Model(args)
        self.model.train(False)
        self.model.cuda()
        self.m_dice = cal_dice()
        self.m_iou = cal_iou()
        self.m_assd = cal_assd()
        self.m_hd95 = cal_hd95()
        self.log_dict = {}

    def calscore(self):
        print(self.args.datapath.split('/')[-1])
        num = len(self.loader)
        # if num < 30:
        #     interval = 1
        # else:
        #     interval = num // 30
        interval=1
        iter = 0

        with torch.no_grad():
            for image, mask, shape, name, origin in tqdm(self.loader):
                mask = mask.squeeze(1)
                shape = mask.shape[1:]
                preds = []
                iter += 1
                image = image.cuda().float()
                image_shaped = F.interpolate(image, size=shape, mode='bilinear', align_corners=True)
                if args.warmup:
                    if args.tta:
                        for transformer in transforms:
                            rgb_trans = transformer.augment_image(image)
                            model_output = self.model.warmup(rgb_trans)
                            deaug_mask = transformer.deaugment_mask(model_output)
                            preds.append(deaug_mask)
                        pred = torch.mean(torch.stack(preds, dim=0), dim=0)
                    else:
                        pred = self.model.warmup(image)
                    pred = F.interpolate(pred, size=shape, mode='bilinear', align_corners=True)[0, 0]
                    pred = torch.sigmoid(pred).detach().cpu().numpy()

                else:
                    if args.tta:
                        for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()
                            rgb_trans = transformer.augment_image(image)
                            if args.head == 'cls':
                                model_output = self.model(rgb_trans)['cls_prob']
                            elif args.head == 'mcs':
                                model_output = self.model(rgb_trans)['mcs_prob']
                            elif args.head == 'fuse':
                                model_output = self.model(rgb_trans)['fuse_prob']
                            deaug_mask = transformer.deaugment_mask(model_output)
                            preds.append(deaug_mask)
                        pred = torch.mean(torch.stack(preds, dim=0), dim=0)
                        if args.head == 'cls' or args.head == 'fuse':
                            pred = F.interpolate(pred, size=shape, mode='bilinear', align_corners=True)[0, 0]
                            pred = torch.sigmoid(pred).detach().cpu().numpy()
                        else:
                            pred = F.interpolate(pred, size=shape, mode='bilinear', align_corners=True)[0, 1]
                            pred = pred.detach().cpu().numpy()
                    else:
                        outdict = self.model(image)
    

                        if args.head == 'cls':
                            pred = outdict['cls_prob']
                            pred = F.interpolate(pred, size=shape, mode='bilinear', align_corners=True)[0, 0]
                            pred = torch.sigmoid(pred).detach().cpu().numpy()
                        elif args.head == 'fuse':
                            pred = outdict['fuse_prob']
                            pred = F.interpolate(pred, size=shape, mode='bilinear', align_corners=True)[0, 0]
                            pred = torch.sigmoid(pred).detach().cpu().numpy()
                        elif args.head == 'mcs':
                            pred = outdict['mcs_prob']
                            pred = F.interpolate(pred, size=shape, mode='bilinear', align_corners=True)[0, 1]
                            pred = pred.detach().cpu().numpy()
                if args.postprecessing:
                    pred = keep_largest_connected_components(pred)
                if args.wandb and iter % interval == 0:
                    visualcase(name=name, image=image_shaped, gt=mask, pred=pred, log_dict=self.log_dict,
                               exp=self.args.logexp,
                               head=self.args.head)
                    wandb.log(self.log_dict)

                mask = mask.cpu().numpy()
                self.m_dice.update(pred, mask)
                self.m_iou.update(pred, mask)
                self.m_assd.update(pred, mask)
                self.m_hd95.update(pred, mask)

                if self.args.saveres:
                    if not os.path.exists(self.args.savepath):
                        os.makedirs(self.args.savepath)
                    cv2.imwrite(self.args.savepath + '/' + name[0], np.round(pred * 255))

        m_dice = self.m_dice.show()
        m_iou = self.m_iou.show()
        m_assd = self.m_assd.show()
        m_hd95 = self.m_hd95.show()
        self.log_dict['mdice'] = m_dice
        self.log_dict['m_iou'] = m_iou
        self.log_dict['m_assd'] = m_assd
        self.log_dict['m_hd95'] = m_hd95
        wandb.log(self.log_dict)
        print('dataset: {} mDice: {:.4f} mIOU: {:.4f}   mAssd: {:.4f} mHD95: {:.4f}'.format(
            self.args.datapath.split('/')[-1], m_dice, m_iou, m_assd, m_hd95))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--saveres', type=bool, default=False, help='whether to save results')
    parser.add_argument('--savepath', type=str, help='path of saved results')
    parser.add_argument('--dataset', default='isic2017') 
    parser.add_argument('--datapath', tyep=str)  
    parser.add_argument('--mode', default='test')
    parser.add_argument('--num_workers', default=4)
    parser.add_argument('--snapshot',type=str, help='model checkpoint path')
    parser.add_argument('--exp', default='pvt')
    parser.add_argument('--logexp', default='PH2')
    parser.add_argument('--patchsize', type=int, default=352, help='patchsize for training set and test set')
    parser.add_argument('--backbone', default='pvt')  # resNet/res2Net/pvt
    parser.add_argument('--warmup', default=False)
    parser.add_argument('--head', default='fuse')  # cls/mcs/fuse
    parser.add_argument('--tta', default=False, help='whether to use test time augmentation')  # cls/mcs
    parser.add_argument("--postprecessing", type=bool, default=False,
                        help="whether to keep the largest connected area only")
    parser.add_argument('--shareW', type=bool, default=True,
                        help='whether to share query weights between different images')
    parser.add_argument('--cls_attributes', type=int, default=10, help='number of attributes for each class')
    parser.add_argument('--topk', type=int, default=3, help='top k experts')
    parser.add_argument('--num_classes', type=int, default=2, help='number of class considered in mcs head')
    parser.add_argument('--num_queries', type=int, default=2,
                        help='number of subcategories, must be the 3 multiple of num_classes')
    parser.add_argument('--tau', type=float, default=1,
                        help='temperature hyperparamter for the pixel-class matching')
    parser.add_argument('--mask_dim', type=int, default=256, help='dimension of final generated feature')
    parser.add_argument('--sample_num', type=int, default=15, help='number of MCS')
    parser.add_argument('--device', type=str, help='the device')
    parser.add_argument('--wandb', type=bool, default=False, help='whether use wandb')
    parser.add_argument('--proj', type=str, default='compareVis', help='wandb project name') #compareVis
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # there exists Motel Carlo Sampling, must set a seed to fix test results
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    args.exp = args.exp + '_head{}_tta{}_postprecessing{}'.format(
        args.head,
        args.tta,
        args.postprecessing,
    )

    if not args.wandb:
        # If you don't want your script to sync to the cloud
        os.environ['WANDB_MODE'] = 'dryrun'

    wandb.init(project=args.proj,
               name=args.exp,
               entity='isyuanyc')

    if args.dataset == 'isic2018':
        testdata = ISIC2018Data(datapath=args.datapath, split='valid', patchsize=args.patchsize)
        test_loader = DataLoader(testdata, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    elif args.dataset == 'isic2017':
        testdata = ISIC2017Data(datapath=args.datapath, split='valid', patchsize=args.patchsize)
        test_loader = DataLoader(testdata, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)


    t = Test(test_loader, Model, args)
    t.calscore()

    wandb.finish()
