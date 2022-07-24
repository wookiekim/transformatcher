r"""Strong supervision"""
from abc import ABC, abstractmethod

import math
import torch.nn.functional as F
import numpy as np
import torch

from geometry import Geometry

class Objective:

    @classmethod
    def initialize(cls, alpha):
        cls.softmax = nn.Softmax(dim=1)
        cls.alpha = alpha
        cls.eps = 1e-5
        cls.smoothL1 = nn.SmoothL1Loss()

    @classmethod
    def kps_regression(cls, prd_kps, trg_kps, npts):
        l2dist = (prd_kps - trg_kps).pow(2).sum(dim=1)
        loss = []
        for dist, npt in zip(l2dist, npts):
            loss.append(dist[:npt].mean())
        return torch.stack(loss).mean()

    @classmethod
    def information_entropy(cls, correlation_matrix):
        r"""Computes information entropy of all candidate matches"""
        bsz, side, side = correlation_matrix.size()
        side = int(math.sqrt(side))
        s = side // 4
        correlation_matrix = Geometry.interpolate4d(correlation_matrix.view(bsz, side, side, side, side),
                                                    [s, s]).view(bsz, s ** 2, s ** 2)

        src_pdf = torch.nn.functional.softmax(correlation_matrix, dim=2)
        trg_pdf = torch.nn.functional.softmax(correlation_matrix.transpose(1, 2), dim=2)

        src_ent = (-(src_pdf * torch.log2(src_pdf + cls.eps)).sum(dim=2)).view(bsz, -1)
        trg_ent = (-(trg_pdf * torch.log2(trg_pdf + cls.eps)).sum(dim=2)).view(bsz, -1)
        score_net = (src_ent + trg_ent).mean(dim=1) / 2

        return score_net.mean()


class StrongSupStrategy:
    def get_image_pair(self, batch, *args):
        r"""Returns (semantically related) pairs for strongly-supervised training"""
        return batch['src_img'], batch['trg_img']

    def get_correlation(self, correlation_matrix):
        r"""Returns correlation matrices of 'ALL PAIRS' in a batch"""
        return correlation_matrix

    def compute_loss(self, correlation_matrix, *args):
        r"""Strongly-supervised matching loss (L_{match})"""
        prd_kps = args[0]
        gt_kps = Geometry.normalize_kps(args[1])
        npts = args[2]

        loss_net = Objective.kps_regression(prd_kps, gt_kps, npts)

        return loss_net