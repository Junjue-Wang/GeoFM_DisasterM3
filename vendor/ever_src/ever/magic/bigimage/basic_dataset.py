import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tifffile
from skimage.io import imread, imsave
from torch.utils.data import Dataset
from tqdm import tqdm

from .sliding_window import sliding_window

__all__ = ['ImageDirDataset',
           'BigImagePatchData',
           'BigImageMaskPatchData',
           'ParallelBigImagePatchDataset']


def _imread(fp):
    if fp.endswith('tif'):
        return tifffile.imread(fp)
    elif fp.endswith('*.jpg') or fp.endswith('*.png'):
        return imread(fp)
    else:
        raise ValueError('Unsupported type')


class ImageDirDataset(Dataset):
    def __init__(self, image_dir, exts=('*.tif', '*.jpg', '*.png'), transforms=None):
        super(ImageDirDataset, self).__init__()
        self.image_paths = sum([glob.glob(os.path.join(image_dir, ext)) for ext in exts])
        self.transforms = transforms

    def __getitem__(self, idx):
        fp = self.image_paths[idx]

        if self.transforms:
            im = _imread(fp)
            out = self.transforms(**dict(image=im))

        out.update(dict(image_name=os.path.basename(fp)))
        return out

    def __len__(self):
        return len(self.image_paths)


class BigImageMaskPatchData(Dataset):
    def __init__(self, image_path, mask_path, kernel_size, stride, transforms=None):
        self.image = tifffile.imread(image_path, out='memmap')
        self.mask = tifffile.imread(mask_path, out='memmap')
        assert self.image.shape[0] == self.mask.shape[0]
        assert self.image.shape[1] == self.mask.shape[1]
        h, w = self.image.shape[:2]
        self.boxes = sliding_window((h, w), kernel_size, stride)
        self.transforms = transforms

    def __getitem__(self, idx):
        box = self.boxes[idx]
        xmin, ymin, xmax, ymax = box
        img = np.asarray(self.image[ymin:ymax, xmin:xmax])
        mask = np.asarray(self.mask[ymin:ymax, xmin:xmax])
        if self.transforms:
            blob = self.transforms(**dict(image=img, mask=mask))
            img = blob['image']
            mask = blob['mask']
        return img, mask, box

    def __len__(self):
        return self.boxes.shape[0]


class BigImagePatchData(Dataset):
    def __init__(self, image, kernel_size, stride, transforms=None):
        h, w = image.shape[:2]
        self.boxes = sliding_window((h, w), kernel_size, stride)
        self.image = image
        self.transforms = transforms

    def __getitem__(self, idx):
        box = self.boxes[idx]
        xmin, ymin, xmax, ymax = box
        # img = np.asarray(self.image[ymin:ymax, xmin:xmax].cpu())  # expects tensor; breaks when image is numpy
        img = np.asarray(self.image[ymin:ymax, xmin:xmax])
        if self.transforms:
            img = self.transforms(**dict(image=img))['image']
        return img, box

    def __len__(self):
        return self.boxes.shape[0]

    def to_tiff(self, out_dir, filter_fn=None, prefix='', parallel=None, name_func=None):
        if parallel:
            ppe = ProcessPoolExecutor(max_workers=parallel['max_workers'])

        _filter_fn = (lambda img, box: True) if filter_fn is None else filter_fn
        for i in tqdm(range(len(self)), desc='saving tiff patches'):
            img, box = self[i]
            if _filter_fn(img, box):

                if name_func is None:
                    name = f'{prefix}{i}.png'
                else:
                    name = name_func(prefix, box, i)

                if parallel:
                    ppe.submit(tifffile.imsave, os.path.join(out_dir, name), img)
                else:
                    tifffile.imsave(os.path.join(out_dir, name), img)
            else:
                continue
        if parallel:
            ppe.shutdown()

    def to_png(self, out_dir, filter_fn=None, prefix='', parallel=None, name_func=None):
        if parallel:
            ppe = ProcessPoolExecutor(max_workers=parallel['max_workers'])
        _filter_fn = (lambda img, box: True) if filter_fn is None else filter_fn
        for i in tqdm(range(len(self)), desc='saving png patches'):
            img, box = self[i]
            if _filter_fn(img, box):
                assert img.shape[2] == 1 or img.shape[2] == 3 or img.shape[2] == 4

                if name_func is None:
                    name = f'{prefix}{i}.png'
                else:
                    name = name_func(prefix, box, i)

                if parallel:
                    ppe.submit(imsave, os.path.join(out_dir, name), img)
                else:
                    imsave(os.path.join(out_dir, name), img)
            else:
                continue

        if parallel:
            ppe.shutdown()

    def filtered_size(self, filter_fn=None):
        if filter_fn is None:
            return len(self)
        cnt = 0
        for i in range(len(self)):
            img, box = self[i]
            if filter_fn(img, box):
                cnt += 1

        return cnt


class ParallelBigImagePatchDataset(Dataset):
    def __init__(self, datasets):
        N = len(datasets[0])
        assert isinstance(datasets[0], BigImagePatchData)
        assert all([len(d) == N for d in datasets]), 'All datasets must have the same length.'
        self.datasets = datasets

    def __getitem__(self, idx):
        imgs = []
        box = None
        for d in self.datasets:
            img, box = d[idx]
            imgs.append(img)
        if imgs[0].ndim == 3:
            img = np.concatenate(imgs, axis=2)
        else:
            img = np.stack(imgs, axis=2)
        return img, box

    def __len__(self):
        return len(self.datasets[0])
