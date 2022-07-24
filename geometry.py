"""Provides functions that manipulate boxes and points"""
import math

import torch.nn.functional as F
import torch.nn as nn
import torch

import utils

class Geometry(object):

    @classmethod
    def initialize(cls, imside, device):
        cls.imside = imside
        jump_size = 8  # 8
        rfside = 15  # 15
        cls.upsample_size = [int(imside / jump_size)] * 2  # imside_ch
        cls.rfs = cls.receptive_fields(rfside, jump_size, cls.upsample_size).to(device)
        cls.rf_center = cls.center(cls.rfs)

        cls.spatial_side = cls.upsample_size[0]

        cls.grid_x = torch.linspace(-1, 1, cls.spatial_side).to(device)
        cls.grid_x = cls.grid_x.view(1, -1).repeat(cls.spatial_side, 1).view(1, 1, -1)

        cls.grid_y = torch.linspace(-1, 1, cls.spatial_side).to(device)
        cls.grid_y = cls.grid_y.view(-1, 1).repeat(1, cls.spatial_side).view(1, 1, -1)

        cls.x = torch.arange(0, cls.spatial_side).float().to(device)
        cls.y = torch.arange(0, cls.spatial_side).float().to(device)

        cls.unnorm = UnNormalize()

        cls.grid = torch.stack(list(reversed(torch.meshgrid(torch.linspace(-1, 1, cls.spatial_side),
                                                            torch.linspace(-1, 1, cls.spatial_side))))).permute(1, 2, 0).to(device)

    @classmethod
    def normalize_kps(cls, kps):
        kps = kps.clone().detach()
        kps[kps != -2] -= (cls.imside // 2)
        kps[kps != -2] /= (cls.imside // 2)
        return kps

    @classmethod
    def unnormalize_kps(cls, kps):
        kps = kps.clone().detach()
        kps[kps != -2] *= (cls.imside // 2)
        kps[kps != -2] += (cls.imside // 2)
        return kps

    @classmethod
    def attentive_indexing(cls, kps, thres=0.1):
        r"""kps: normalized keypoints x, y (N, 2)
            returns attentive index map(N, spatial_side, spatial_side)
        """
        nkps = kps.size(0)
        kps = kps.view(nkps, 1, 1, 2)

        eps = 1e-5
        attmap = (cls.grid.unsqueeze(0).repeat(nkps, 1, 1, 1) - kps).pow(2).sum(dim=3)
        attmap = (attmap + eps).pow(0.5)
        attmap = (thres - attmap).clamp(min=0).view(nkps, -1)
        attmap = attmap / (attmap.sum(dim=1, keepdim=True) + eps)
        attmap = attmap.view(nkps, cls.spatial_side, cls.spatial_side)

        return attmap

    @classmethod
    def apply_gaussian_kernel(cls, corr, sigma=10):
        bsz, side, side = corr.size()

        center = corr.max(dim=2)[1]
        center_y = center // cls.spatial_side
        center_x = center % cls.spatial_side

        y = cls.y.view(1, 1, cls.spatial_side).repeat(bsz, center_y.size(1), 1) - center_y.unsqueeze(2)
        x = cls.x.view(1, 1, cls.spatial_side).repeat(bsz, center_x.size(1), 1) - center_x.unsqueeze(2)

        y = y.unsqueeze(3).repeat(1, 1, 1, cls.spatial_side)
        x = x.unsqueeze(2).repeat(1, 1, cls.spatial_side, 1)

        gauss_kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))
        filtered_corr = gauss_kernel * corr.view(bsz, -1, cls.spatial_side, cls.spatial_side)
        filtered_corr = filtered_corr.view(bsz, side, side)

        return filtered_corr

    @classmethod
    def transfer_kps_diff(cls, confidence_ts, src_kps, n_pts, normalized, is_train=False):
        r"""Transfer keypoints by weighted average"""
        if is_train:
            thres = 0.1
        else:
            thres = 0.05
        if not normalized:
            src_kps = Geometry.normalize_kps(src_kps)
        confidence_ts = cls.apply_gaussian_kernel(confidence_ts)

        pdf = F.softmax(confidence_ts, dim=2)
        prd_x = (pdf * cls.grid_x).sum(dim=2)
        prd_y = (pdf * cls.grid_y).sum(dim=2)

        prd_kps = []
        for idx, (x, y, src_kp, np) in enumerate(zip(prd_x, prd_y, src_kps, n_pts)):
            max_pts = src_kp.size()[1]
            prd_xy = torch.stack([x, y]).t()

            src_kp = src_kp[:, :np].t()
            attmap = cls.attentive_indexing(src_kp,thres).view(np, -1)
            prd_kp = (prd_xy.unsqueeze(0) * attmap.unsqueeze(-1)).sum(dim=1).t()
            pads = (torch.zeros((2, max_pts - np)).to(prd_kp.device) - 2)
            prd_kp = torch.cat([prd_kp, pads], dim=1)
            prd_kps.append(prd_kp)

        return torch.stack(prd_kps)
   
    @staticmethod
    def center(box):
        r"""Calculates center (x, y) of box (N, 4)"""
        x_center = box[:, 0] + (box[:, 2] - box[:, 0]) // 2
        y_center = box[:, 1] + (box[:, 3] - box[:, 1]) // 2
        return torch.stack((x_center, y_center)).t().to(box.device)

    @staticmethod
    def receptive_fields(rfsz, jsz, feat_size):
        r"""Returns a set of receptive fields (N, 4)"""
        width = feat_size[1]
        height = feat_size[0]

        feat_ids = torch.tensor(list(range(width))).repeat(1, height).t().repeat(1, 2)
        feat_ids[:, 0] = torch.tensor(list(range(height))).unsqueeze(1).repeat(1, width).view(-1)

        box = torch.zeros(feat_ids.size()[0], 4)
        box[:, 0] = feat_ids[:, 1] * jsz - rfsz // 2
        box[:, 1] = feat_ids[:, 0] * jsz - rfsz // 2
        box[:, 2] = feat_ids[:, 1] * jsz + rfsz // 2
        box[:, 3] = feat_ids[:, 0] * jsz + rfsz // 2

        return box

    @staticmethod
    def gaussian2d(side=7):
        r"""Returns 2-dimensional gaussian filter"""
        dim = [side, side]

        siz = torch.LongTensor(dim)
        sig_sq = (siz.float() / 2 / 2.354).pow(2)
        siz2 = (siz - 1) / 2

        x_axis = torch.arange(-siz2[0], siz2[0] + 1).unsqueeze(0).expand(dim).float()
        y_axis = torch.arange(-siz2[1], siz2[1] + 1).unsqueeze(1).expand(dim).float()

        gaussian = torch.exp(-(x_axis.pow(2) / 2 / sig_sq[0] + y_axis.pow(2) / 2 / sig_sq[1]))
        gaussian = gaussian / gaussian.sum()

        return gaussian

    @staticmethod
    def neighbours(box, kps):
        r"""Returns boxes in one-hot format that covers given keypoints"""
        box_duplicate = box.unsqueeze(2).repeat(1, 1, len(kps.t())).transpose(0, 1)
        kps_duplicate = kps.unsqueeze(1).repeat(1, len(box), 1)

        xmin = kps_duplicate[0].ge(box_duplicate[0])
        ymin = kps_duplicate[1].ge(box_duplicate[1])
        xmax = kps_duplicate[0].le(box_duplicate[2])
        ymax = kps_duplicate[1].le(box_duplicate[3])

        nbr_onehot = torch.mul(torch.mul(xmin, ymin), torch.mul(xmax, ymax)).t()
        n_neighbours = nbr_onehot.sum(dim=1)

        return nbr_onehot, n_neighbours

    @staticmethod
    def interpolate4d(tensor4d, size):
        bsz, h1, w1, h2, w2 = tensor4d.size()
        tensor4d = tensor4d.view(bsz, h1, w1, -1).permute(0, 3, 1, 2)
        tensor4d = F.interpolate(tensor4d, size, mode='bilinear', align_corners=True)
        tensor4d = tensor4d.view(bsz, h2, w2, -1).permute(0, 3, 1, 2)
        tensor4d = F.interpolate(tensor4d, size, mode='bilinear', align_corners=True)
        tensor4d = tensor4d.view(bsz, size[0], size[0], size[0], size[0])
        return tensor4d


    @classmethod
    def transfer_kps_dhpf(cls, correlation_matrix, kps, n_pts, normalized):
        r"""Transfer keypoints by nearest-neighbour assignment"""

        max_pts = 40

        prd_kps = []
        for ct, kpss, np in zip(correlation_matrix, kps, n_pts):

            # 1. Prepare geometries & argmax target indices
            kp = kpss.narrow_copy(1, 0, np)
            _, trg_argmax_idx = torch.max(ct, dim=1)
            geomet = cls.rfs[:, :2].unsqueeze(0).repeat(len(kp.t()), 1, 1)

            # 2. Retrieve neighbouring source boxes that cover source key-points
            src_nbr_onehot, n_neighbours = cls.neighbours(cls.rfs, kp)

            # 3. Get displacements from source neighbouring box centers to each key-point
            src_displacements = kp.t().unsqueeze(1).repeat(1, len(cls.rfs), 1) - geomet
            src_displacements = src_displacements * src_nbr_onehot.unsqueeze(2).repeat(1, 1, 2).float()

            # 4. Transfer the neighbours based on given correlation matrix
            vector_summator = torch.zeros_like(geomet)
            src_idx = src_nbr_onehot.nonzero()

            trg_idx = trg_argmax_idx.index_select(dim=0, index=src_idx[:, 1])
            vector_summator[src_idx[:, 0], src_idx[:, 1]] = geomet[src_idx[:, 0], trg_idx]
            vector_summator += src_displacements
            prd = (vector_summator.sum(dim=1) / n_neighbours.unsqueeze(1).repeat(1, 2).float()).t()

            # 5. Concatenate pad-points
            pads = (torch.zeros((2, max_pts - np)).to(prd.device) - 1)
            prd = torch.cat([prd, pads], dim=1)
            prd_kps.append(prd)

        return torch.stack(prd_kps)


    @classmethod
    def cosine_similarity(cls, src_feats, trg_feats):
        correlations = []
        for src_feat, trg_feat in zip(src_feats, trg_feats):
            src_feat = utils.l2normalize(src_feat, dim=1)
            trg_feat = utils.l2normalize(trg_feat, dim=1)
            src_feat = F.interpolate(src_feat, 15, mode='bilinear', align_corners=True)
            trg_feat = F.interpolate(trg_feat, 15, mode='bilinear', align_corners=True)

            bsz, nch, side, side = src_feat.size()
            src_feat = src_feat.view(bsz, nch, -1).transpose(1, 2)
            trg_feat = trg_feat.view(bsz, nch, -1)
            corr = torch.bmm(src_feat, trg_feat)
            correlations.append(corr.view(bsz, 1, side, side, side, side))
        return correlations


class UnNormalize:
    r"""Image unnormalization"""
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image):
        img = image.clone()
        for im_channel, mean, std in zip(img, self.mean, self.std):
            im_channel.mul_(std).add_(mean)
        return img


