import cv2
import os
from skimage.io import imread
import tifffile
import torch
from torch.utils.data import Dataset
from utils import utils
import numpy as np
from loguru import logger

class XViewDataset(Dataset):
    "Dataset for xView"

    def __init__(self, pairs, mode, bldg_polys, return_geo=False):
        """
        :param pre_chips: List of pre-damage chip filenames
        :param post_chips: List of post_damage chip filenames
        :param transform: PyTorch transforms to be used on each example
        """
        self.pairs = pairs
        self.return_geo=return_geo
        self.mode = mode
        self.bldg_polys = bldg_polys


    def __len__(self):
        return(len(self.pairs))

    def __getitem__(self, idx, return_img=False):
        fl = self.pairs[idx]

        pre_image = cv2.imread(str(fl.opts.in_pre_path), cv2.IMREAD_COLOR)
        logger.trace(f'Pre image shape before GRAY2RGB: {pre_image.shape}')
        if self.bldg_polys:
            logger.trace(pre_image)
            pre_image = cv2.cvtColor(pre_image,cv2.COLOR_GRAY2RGB)
            logger.trace(f'Pre image shape after GRAY2RGB: {pre_image.shape}')
        post_image = cv2.imread(str(fl.opts.in_post_path), cv2.IMREAD_COLOR)

        if self.mode == 'cls':
            img = np.concatenate([pre_image, post_image], axis=2)
        elif self.mode == 'loc':
            img = pre_image
        else:
            raise ValueError('Incorrect mode!  Must be cls or loc')
            
        img = utils.preprocess_inputs(img)

        inp = []
        inp.append(img)
        inp.append(img[::-1, ...])
        inp.append(img[:, ::-1, ...])
        inp.append(img[::-1, ::-1, ...])
        inp = np.asarray(inp, dtype='float')
        inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
        
        out_dict = {}
        out_dict['in_pre_path'] = str(fl.opts.in_pre_path)
        out_dict['in_post_path'] = str(fl.opts.in_post_path)
        if return_img:
            out_dict['pre_image'] = pre_image
            out_dict['post_image'] = post_image
        out_dict['img'] = inp
        out_dict['idx'] = idx
        out_dict['out_cls_path'] = str(fl.opts.out_cls_path)
        out_dict['out_loc_path'] = str(fl.opts.out_loc_path)
        out_dict['out_overlay_path'] = str(fl.opts.out_overlay_path)
        out_dict['is_vis'] = fl.opts.is_vis

        return out_dict
