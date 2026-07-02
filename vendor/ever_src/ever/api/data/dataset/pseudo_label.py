import glob
import os
from concurrent.futures import ProcessPoolExecutor
from functools import reduce

import numpy as np
import torch
from skimage.io import imread, imsave
from torch.utils.data import Dataset


class PseudoMaskDataset(Dataset):
    """
    dataset = PseudoMaskDataset(...)

    # the mode for training using images and pseudo labels
    dataset.to(PseudoMaskDataset.IMAGE_MASK)

    # the mode for predicting pseudo labels
    dataset.to(PseudoMaskDataset.IMAGE_ONLY)

    for image, ann in loader:
        y_pred = model(image)
        dataset.save_pseudo_mask(ann['filename'], y_pred)

    """
    IMAGE_ONLY = 'image_only'
    IMAGE_MASK = 'image_mask'
    SUFFIX = '_pseudo_mask'
    VERSIONED_SUFFIX = '_pseudo_mask_{version}'

    def __init__(self, image_dir, mask_dir=None, image_ext=('*.tif', '*.png', '*.jpg'), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.im_ext = image_ext
        self.transform = transform
        self.state = PseudoMaskDataset.IMAGE_ONLY

        self.im_fp_list = self.load_filenames(image_dir)

        self.ppe = ProcessPoolExecutor(max_workers=4)

    def __getitem__(self, idx):
        im_fp = self.im_fp_list[idx]

        im = imread(im_fp).astype(np.float32)

        ann = dict(filename=im_fp)

        blob = dict(image=im)
        if self.state == PseudoMaskDataset.IMAGE_MASK:
            ann.update(dict(
                mask=imread(self.mask_list[idx]).astype(np.int64)
            ))
            blob['mask'] = ann['mask']

        if self.transform:
            blob = self.transform(**blob)
            im = blob['image']
            if 'mask' in blob:
                ann['mask'] = blob['mask']

        return im, ann

    def __len__(self):
        return len(self.im_fp_list)

    def load_filenames(self, image_dir):
        return reduce(lambda a, b: a + b, [glob.glob(os.path.join(image_dir, ext)) for ext in self.im_ext])

    def to(self, mode: str):
        if mode == self.state:
            return self
        if PseudoMaskDataset.IMAGE_MASK == mode:
            self.mask_list = []
            del self.ppe
            self.ppe = None
            for fp in self.im_fp_list:
                basename = os.path.basename(fp)
                for ext in self.im_ext:
                    basename = basename.replace(ext[1:], '.png')
                mask_fp = os.path.join(self.mask_dir, basename)
                if os.path.exists(mask_fp):
                    self.mask_list.append(mask_fp)
                else:
                    raise FileNotFoundError(mask_fp)

        self.state = mode
        return self

    def save_hard_pseudo_mask(self, filename, pseudo_mask):
        basename = os.path.basename(filename)
        for ext in self.im_ext:
            basename = basename.replace(ext[1:], '.png')
        save_path = os.path.join(self.mask_dir, basename)

        if isinstance(pseudo_mask, torch.Tensor):
            pseudo_mask = pseudo_mask.cpu().numpy()

        pseudo_mask = pseudo_mask.astype(np.uint8, copy=False)
        pseudo_mask = np.squeeze(pseudo_mask)

        self.ppe.submit(imsave, save_path, pseudo_mask, check_contrast=False)

    def __del__(self):
        if self.ppe:
            self.ppe.shutdown()
