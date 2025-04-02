# coding=utf-8

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.mask2former_decoder import MultiScaleMaskedTransformerDecoder
from contrastive_model.lib.pvtv2 import pvt_v2_b2
from modeling.unet import U_Net
from modeling.resnet import resnet50
from PIL import Image
from scipy.optimize import linear_sum_assignment
import time


def erosion_to_dilate(output):
    """

    :param output: BHW
    :return:
    """
    z = np.where(output > 0.5, 1.0, 0.0)  # covert segmentation result
    kernel = np.ones((4, 4), np.uint8)  # kernal matrix
    mask_bg = np.zeros_like(z)  # result array
    mask_fg = np.zeros_like(z)  # result array
    mask_boundary = np.zeros_like(z)  # result array
    for i in range(output.shape[0]):
        y = z[i]
        dilate = np.array(y, dtype='uint8')
        erosion = np.array(y, dtype='uint8')
        erosion = cv2.erode(erosion, kernel, 4)
        dilate = cv2.dilate(dilate, kernel, 4)
        mask1 = np.ones_like(dilate) - dilate
        mask2 = erosion
        boundary = dilate - erosion
        mask_bg[i] = mask1
        mask_fg[i] = mask2
        mask_boundary[i] = boundary
    return mask_bg, mask_fg, mask_boundary


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        if args.backbone == 'resNet':
            self.bkbone = resnet50()
            path = '../res/resnet50-19c8e357.pth'
            save_model = torch.load(path)
            model_dict = self.bkbone.state_dict()
            state_dict = {
                k: v
                for k, v in save_model.items() if k in model_dict.keys()
            }
            model_dict.update(state_dict)
            print("load pretrained resnet model.....")
            self.bkbone.load_state_dict(model_dict)

        elif args.backbone == 'pvt':
            self.bkbone = pvt_v2_b2(img_size=args.patchsize)
            path = '../contrastive_model/lib/pvt_v2_b2.pth'
            save_model = torch.load(path)
            model_dict = self.bkbone.state_dict()
            state_dict = {
                k: v
                for k, v in save_model.items() if k in model_dict.keys()
            }
            model_dict.update(state_dict)
            print("load pretrained pvt_v2_b2 model.....")
            self.bkbone.load_state_dict(model_dict)

        elif args.backbone == 'unet':
            self.bkbone = U_Net()

        if args.backbone == 'res2Net' or args.backbone == 'resNet':
            self.linear5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
            self.linear4 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
            self.linear3 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        elif args.backbone == 'pvt':
            self.linear5 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
            self.linear4 = nn.Sequential(nn.Conv2d(320, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
            self.linear3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))

        self.predict = nn.Conv2d(64 * 3, 1, kernel_size=1, stride=1, padding=0)

        self.featconv = nn.Conv2d(64 * 3, args.mask_dim, kernel_size=1, stride=1, padding=0)

        self.mask2former = MultiScaleMaskedTransformerDecoder(in_channels=64, num_queries=args.num_queries,
                                                              num_classes=args.num_classes,
                                                              cls_attributes=args.cls_attributes,
                                                              dec_layers=args.dec_layers,
                                                              mask_dim=args.mask_dim,
                                                              shareW=args.shareW)
        self.fuse_mode = args.fuse_mode
        self.fuse = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

        # Multiple of Experts (MoE)
        # self.switch = nn.Linear(64 * 4, args.cls_attributes)
        self.switch1 = nn.Sequential(
            nn.Linear(64 * 4, 64 * 2),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.Linear(32, args.cls_attributes),
        )

        self.fc1 = nn.Linear(in_features=args.cls_attributes * args.mask_dim, out_features=args.mask_dim)
        self.fc2 = nn.Linear(in_features=args.mask_dim, out_features=args.mask_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.mask_norm = nn.LayerNorm(args.num_classes)
        self.feat_norm = nn.LayerNorm(args.mask_dim)

        self.initialize()

    def warmup(self, x):
        out2, out3, out4, out5 = self.bkbone(x.to(self.predict.weight.dtype))
        out5 = self.linear5(out5)  # OS32
        out4 = self.linear4(out4)  # OS16
        out3 = self.linear3(out3)  # OS8
        out5 = F.interpolate(out5, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out4 = F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=True)
        predfeat = torch.cat([out5, out4 * out5, out3 * out4 * out5], dim=1)
        clspred = self.predict(predfeat)
        return clspred

    def forward(self, img, gt=None, shape=None):
        """

        :param img: BCHW
        :param gt:
        :param shape:
        :return:
        """

        returndict = {}
        multi_scale_feats = []
        # pixel encoder
        out2, out3, out4, out5 = self.bkbone(img)

        # pixel decoder
        out5 = self.linear5(out5)  # OS32
        out4 = self.linear4(out4)  # OS16
        out3 = self.linear3(out3)  # OS8
        out2 = self.linear2(out2)  # OS4
        multi_scale_feats.append(out5)
        multi_scale_feats.append(out4)
        multi_scale_feats.append(out3)

        out5_3 = F.interpolate(out5, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out4_3 = F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=True)
        predfeat = torch.cat([out5_3, out4_3 * out5_3, out3 * out4_3 * out5_3], dim=1)

        cls_pred = self.predict(predfeat)
        returndict['cls_prob'] = cls_pred  # os8

        # use linear head output to generate mask for mask attention in the transformer decoder layers
        pred = F.interpolate(cls_pred, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=True)[:, 0]
        pred[torch.where(pred > 0)] /= (pred > 0).float().mean()
        pred[torch.where(pred < 0)] /= (pred < 0).float().mean()
        pred = torch.sigmoid(pred).detach().cpu().numpy()
        mask_bg, mask_fg, mask_boundary = erosion_to_dilate(pred)
        mask_bg, mask_fg, mask_boundary = torch.from_numpy(mask_bg).cuda(), torch.from_numpy(
            mask_fg).cuda(), torch.from_numpy(mask_boundary).cuda()  # BHW

        mask_bg, mask_fg = 1 - mask_bg, 1 - mask_fg  # attn-mask: ``True`` value indicates that the corresponding position is not allowed to attend
        mask_bg, mask_fg = mask_bg.unsqueeze(1).repeat(1, self.args.cls_attributes, 1, 1), \
                           mask_fg.unsqueeze(1).repeat(1, self.args.cls_attributes, 1, 1)  # BQHW

        mask_bg, mask_fg = mask_bg.detach(), mask_fg.detach()
        mask = [mask_bg, mask_fg]
        mean_embed, var_embed = self.mask2former(multi_scale_feats, mask)  # B,D,C,S,K
        returndict['att_m'] = mean_embed
        returndict['att_v'] = var_embed

        # MoE routing network
        avg_out2 = F.adaptive_avg_pool2d(out2, 1).squeeze(-1).squeeze(-1)
        avg_out3 = F.adaptive_avg_pool2d(out3, 1).squeeze(-1).squeeze(-1)
        avg_out4 = F.adaptive_avg_pool2d(out4, 1).squeeze(-1).squeeze(-1)
        avg_out5 = F.adaptive_avg_pool2d(out5, 1).squeeze(-1).squeeze(-1)
        poolfeat = torch.cat((avg_out5, avg_out4, avg_out3, avg_out2), dim=-1)
        # route_value = self.switch(poolfeat)  # B,K
        route_value = self.switch1(poolfeat)  # B,K
        topk_indices, topk_mask = self.get_topk_mask(route_value, k=self.args.topk)
        # return topk_indices
        returndict['topk_indices'] = topk_indices  # B,k
        returndict['topk_mask'] = topk_mask  # B,K
        topk_route_value = topk_mask * route_value  # B,K
        route_prob = torch.where(topk_route_value == 0, topk_route_value.new_tensor(float('-inf')),
                                 topk_route_value)
        route_prob = F.softmax(route_prob, dim=-1)

        returndict['route_prob'] = route_prob
        
        epsilon = torch.randn(self.args.sample_num, *mean_embed.shape, device=mean_embed.device)
        samples = mean_embed.unsqueeze(0) + var_embed.unsqueeze(0) * epsilon  # 
        samples = samples.permute(1,0,2,3,4,5)
        returndict['samples'] = samples
        # BNDCK-> BNDC
        sample = samples.permute(0, 1, 3, 4, 2, 5)  # B,N,D,C,S,K-> B,N,C,S,D,K
        # # MoE
        _, N, C, S, D, _ = sample.shape

        sample = sample * (route_prob[:, None, None, None, None, :].repeat(1, N, C, S, D, 1))  # B,N,C,S,D,K
        sample = torch.flatten(sample, start_dim=4)  # B,N,C,S,D*K
        sample = self.fc1(sample.to(self.fc1.weight.dtype))  # # B,N,C,S,D
        sample = self.LeakyReLU(sample)
        cls_feats = self.fc2(sample)  # # B,N,C,S,D

        up_seg_feat = self.featconv(predfeat)  # bdh'w' (h'=OS8)
        up_seg_feat = F.interpolate(up_seg_feat, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=True)
        seg_feats = up_seg_feat.permute(0, 2, 3, 1)  # bdhw->bhwd
        seg_feats = self.feat_norm(seg_feats)
        seg_feats = F.normalize(seg_feats, p=2, dim=-1)  # cosine sim norm
        cls_feats = F.normalize(cls_feats, p=2, dim=-1)  # cosine sim norm
        
        cls_pixel_sim = torch.einsum(
            "bhwd,bncsd->bncshw", seg_feats, cls_feats.to(seg_feats.dtype)
        )

        cls_pixel_sim = torch.amax(cls_pixel_sim, dim=3)  # bnchw
        cls_pixel_sim = cls_pixel_sim.permute(0, 1, 3, 4, 2)  # bnchw->bnhwc
        cls_pixel_sim = self.mask_norm(cls_pixel_sim)
        cls_pixel_sim = cls_pixel_sim.permute(0, 1, 4, 2, 3)  # bnhwc->bnchw
        numerator = torch.exp(cls_pixel_sim / self.args.tau)
        denominator = torch.sum(numerator, dim=2, keepdim=True)
        prob_c = numerator / denominator
        prob = torch.mean(prob_c, dim=1)  # bchw
        returndict['mcs_prob'] = prob

        # cls and mcs head fusion
        fuse_mcs_feat = torch.mean(cls_pixel_sim, dim=1)  # b2hw
        fuse_cls_feat = F.interpolate(cls_pred, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=True)
        if self.fuse_mode == 'conv':
            fuse_prob = self.fuse(torch.cat((fuse_mcs_feat, fuse_cls_feat), dim=1))
        elif self.fuse_mode == 'avg':
            mcs_prob = prob[:,1:2,:,:]
            cls_prob = F.sigmoid(fuse_cls_feat)
            fuse_prob = (mcs_prob+cls_prob)/2
        elif self.fuse_mode == 'max':
            mcs_prob = prob[:,1:2,:,:]
            cls_prob = F.sigmoid(fuse_cls_feat)
            fuse_prob = torch.maximum(mcs_prob,cls_prob)

        returndict['fuse_prob'] = fuse_prob

        return returndict

    def initialize(self):
        if self.args.snapshot:
            print("load pretrained encoder decoder model.....")
            pretrained_dict = torch.load(self.args.snapshot)
            model_dict = self.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if ('bkbone' in k or 'linear3' in k or'linear4' in k or'linear5' in k or'predict' in k)}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            self.weight_init(self)

    def weight_init(self, module):
        for n, m in module.named_children():
            print('initialize: ' + n)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Sequential):
                self.weight_init(m)
            elif isinstance(m, (nn.ReLU, nn.PReLU, nn.LeakyReLU, nn.Dropout, nn.Softmax)):
                pass
            else:
                pass

    def get_topk_mask(self, x, k):
        # Get the top k values and indices along the last dimension
        topk_values, topk_indices = torch.topk(x, k, dim=-1)

        # Create a mask with zeros
        mask = torch.zeros_like(x, dtype=torch.float32)

        # Set the elements corresponding to the top k indices to 1
        mask.scatter_(-1, topk_indices, 1)

        return topk_indices, mask
