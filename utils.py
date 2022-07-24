r"""Some helper functions"""
import random

import numpy as np
import torch.nn.functional as F
import torch


def fix_randseed(seed):
    r"""Fixes random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    r"""Computes average of a list"""
    return sum(x) / len(x) if len(x) > 0 else 0.0


def where(predicate):
    r"""Predicate must be a condition on nd-tensor"""
    matching_indices = predicate.nonzero()
    if len(matching_indices) != 0:
        matching_indices = matching_indices.t().squeeze(0)
    return matching_indices


def equal(a, b, eps=1e-9):
    return len(((a.view(-1) - b.view(-1)) > eps).nonzero()) == 0


def feat_normalize(x, interp_size):
    r"""L2-normalizes given 2D feature map after interpolation"""
    x = F.interpolate(x, interp_size, mode='bilinear', align_corners=True)
    return x.pow(2).sum(1).view(x.size(0), -1)


def l2normalize(x, dim):
    r"""L1-normalization"""
    return x / x.norm(p=2, dim=dim, keepdim=True)