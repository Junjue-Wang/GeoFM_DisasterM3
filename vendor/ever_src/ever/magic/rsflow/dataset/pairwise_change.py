import os
import glob
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
import numpy as np


class PairwiseChangeDataset(Dataset):
    def __init__(self, t1_tif_dir, t2_tif_dir, transforms=None):
        self.t1_fps = glob.glob(os.path.join(t1_tif_dir, '*.tif'))
        self.t2_fps = [os.path.join(t2_tif_dir, os.path.basename(fp)) for fp in self.t1_fps]
        self.transforms = transforms

    def __getitem__(self, idx):
        img1 = imread(self.t1_fps[idx])
        img2 = imread(self.t2_fps[idx])
        if img1.ndim == 3:
            img = np.concatenate([img1, img2], axis=2)
        else:
            img = np.stack([img1, img2], axis=2)
        if self.transforms:
            img = self.transforms(image=img)['image']
        return img, self.t1_fps[idx], self.t2_fps[idx]

    def __len__(self):
        return len(self.t1_fps)


def gen_dataloader(t1_tif_dir, t2_tif_dir, batch_size, num_workers, transforms=None):
    dataset = PairwiseChangeDataset(t1_tif_dir, t2_tif_dir, transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
