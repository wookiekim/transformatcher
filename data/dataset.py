r""" Superclass for semantic correspondence datasets """
import random
import os

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class CorrespondenceDataset(Dataset):
    r""" Parent class of PFPascal, PFWillow, and SPair """
    def __init__(self, benchmark, datapath, thres, split, augment, imside):
        r""" CorrespondenceDataset constructor """
        super(CorrespondenceDataset, self).__init__()

        # {Directory name, Layout path, Image path, Annotation path, PCK threshold}
        self.metadata = {
            'pfwillow': ('PF-WILLOW',
                         'test_pairs.csv',
                         '',
                         '',
                         'bbox'),
            'pfpascal': ('PF-PASCAL',
                         '_pairs.csv',
                         'JPEGImages',
                         'Annotations',
                         'img'),
            'spair':    ('SPair-71k',
                         'Layout/large', # large
                         'JPEGImages',
                         'PairAnnotation',
                         'bbox')
        }

        # Directory path for train, val, or test splits
        base_path = os.path.join(os.path.abspath(datapath), self.metadata[benchmark][0])
        if benchmark == 'pfpascal':
            self.spt_path = os.path.join(base_path, split+'_pairs.csv')
        elif benchmark == 'spair':
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1], split+'.txt')
        else:
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1])

        # Directory path for images
        self.img_path = os.path.join(base_path, self.metadata[benchmark][2])

        # Directory path for annotations
        if benchmark == 'spair':
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3], split)
        else:
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3])
        self.augment = augment

        # Miscellaneous
        self.max_pts = 40
        self.split = split
        self.img_size = imside
        self.benchmark = benchmark
        self.range_ts = torch.arange(self.max_pts)
        self.thres = self.metadata[benchmark][4] if thres == 'auto' else thres

        if self.augment and self.split == 'trn':
            self.transform = A.Compose([
                A.ToGray(p=0.2),
                A.Posterize(p=0.2),
                A.Equalize(p=0.2),
                A.augmentations.transforms.Sharpen(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.Solarize(p=0.2),
                A.ColorJitter(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                A.pytorch.transforms.ToTensorV2(),
            ])
        else:
            self.transform = transforms.Compose([transforms.Resize((self.img_size, self.img_size)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])

        # To get initialized in subclass constructors
        self.train_data = []
        self.src_imnames = []
        self.trg_imnames = []
        self.cls = []
        self.cls_ids = []
        self.src_kps = []
        self.trg_kps = []

    def __len__(self):
        r""" Returns the number of pairs """
        return len(self.train_data)

    def __getitem__(self, idx):
        r""" Constructs and return a batch """

        # Image name
        batch = dict()
        batch['src_imname'] = self.src_imnames[idx]
        batch['trg_imname'] = self.trg_imnames[idx]

        # Object category
        batch['category_id'] = self.cls_ids[idx]
        batch['category'] = self.cls[batch['category_id']]

        # Image as numpy (original width, original height)
        src_pil = self.get_image(self.src_imnames, idx)
        trg_pil = self.get_image(self.trg_imnames, idx)
        batch['src_imsize'] = src_pil.size
        batch['trg_imsize'] = trg_pil.size

        # Key-points (re-scaled)
        batch['src_kps'], num_pts = self.get_points(self.src_kps, idx, src_pil.size)
        batch['trg_kps'], _ = self.get_points(self.trg_kps, idx, trg_pil.size)
        batch['n_pts'] = torch.tensor(num_pts)

        # Image as tensor
        if self.augment and self.split == 'trn':
            batch['src_img'] = self.transform(image=np.array(src_pil))['image']
            batch['trg_img'] = self.transform(image=np.array(trg_pil))['image']
            batch['src_img'], batch['src_kps'], batch['src_bbox'] = self.random_crop(batch['src_img'], batch['src_kps'], batch['n_pts'], self.src_bbox[idx].clone())
            batch['trg_img'], batch['trg_kps'], batch['trg_bbox'] = self.random_crop(batch['trg_img'], batch['trg_kps'], batch['n_pts'], self.trg_bbox[idx].clone())
        else:
            batch['src_img'] = self.transform(src_pil)
            batch['trg_img'] = self.transform(trg_pil)
            if self.benchmark != 'pfwillow':
                batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize'])
                batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize'])

        # Total number of pairs in training split
        batch['datalen'] = len(self.train_data)

        return batch

    def random_crop(self, img, kps, n_pts, bbox, pad=20, p=0.5):
        r""" Random crop using keypoint annnotations """
        target_size = (self.img_size,) * 2

        _, h, w = img.shape
        if random.uniform(0, 1) > p:
            resized_img = torchvision.transforms.functional.resize(img, target_size)
            bbox[0::2] *= (self.img_size / w)
            bbox[1::2] *= (self.img_size / h)
            return resized_img, kps, bbox

        kps = kps.t()
        kps_org = kps.clone()
        kps_org[:, 0][:n_pts] *= w / target_size[1]
        kps_org[:, 1][:n_pts] *= h / target_size[0]

        # If there is only one keypoint, crop large area. Otherwise, apply given padding.
        if n_pts == 1:
            x, y = kps_org[:n_pts][0]
            left_pad = int(x // 2)
            right_pad = int((w - x) // 2)
            top_pad = int(y // 2)
            bottom_pad = int((h - y) // 2)
        else:
            left_pad, right_pad, top_pad, bottom_pad = pad, pad, pad, pad

        left = random.randint(0, max(0, kps_org[:, 0][:n_pts].min().int() - left_pad))
        top = random.randint(0, max(0, kps_org[:, 1][:n_pts].min().int() - top_pad))
        width = random.randint(min(w, kps_org[:, 0][:n_pts].max().int() + right_pad), w) - left
        height = random.randint(min(h, kps_org[:, 1][:n_pts].max().int() + bottom_pad), h) - top

        cropped_img = torchvision.transforms.functional.resized_crop(img, top, left, height, width, size=target_size)

        bbox[0::2] -= left
        bbox[1::2] -= top
        bbox[0::2] *= (self.img_size / width)
        bbox[1::2] *= (self.img_size / height)
        bbox = bbox.clamp(min=0, max=self.img_size - 1)

        resized_kps = torch.zeros_like(kps, dtype=torch.float)
        resized_kps[:, 0] = (kps_org[:, 0] - left) * (target_size[1] / width)
        resized_kps[:, 1] = (kps_org[:, 1] - top) * (target_size[0] / height)
        resized_kps = torch.clamp(resized_kps, 0, target_size[0] - 1)
        resized_kps[kps_org == -2.0] = -2.0
        resized_kps = resized_kps.t()

        return cropped_img, resized_kps, bbox

    def get_bbox(self, bbox_list, idx, imsize):
        r""" Return object bounding-box """
        bbox = bbox_list[idx].clone()
        bbox[0::2] *= (self.img_size / imsize[0])
        bbox[1::2] *= (self.img_size / imsize[1])

        return bbox

    def get_image(self, imnames, idx):
        r""" Reads PIL image from path """
        path = os.path.join(self.img_path, imnames[idx])
        return Image.open(path).convert('RGB')

    def get_pckthres(self, batch, imsize):
        r""" Computes PCK threshold """
        if self.thres == 'bbox':
            bbox = batch['trg_bbox'].clone()
            bbox_w = (bbox[2] - bbox[0])
            bbox_h = (bbox[3] - bbox[1])
            pckthres = torch.max(bbox_w, bbox_h)
        elif self.thres == 'img':
            imsize_t = batch['trg_img'].size()
            pckthres = torch.tensor(max(imsize_t[1], imsize_t[2]))
        else:
            raise Exception('Invalid pck threshold type: %s' % self.thres)
        return pckthres.float()

    def get_points(self, pts_list, idx, org_imsize):
        r""" Returns key-points of an image """
        xy, n_pts = pts_list[idx].size()
        pad_pts = torch.zeros((xy, self.max_pts - n_pts)) - 2
        x_crds = pts_list[idx][0] * (self.img_size / org_imsize[0])
        y_crds = pts_list[idx][1] * (self.img_size / org_imsize[1])
        kps = torch.cat([torch.stack([x_crds, y_crds]), pad_pts], dim=1)

        return kps, n_pts
