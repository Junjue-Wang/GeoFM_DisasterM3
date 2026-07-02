import os
import glob
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread


class SegmentationDataset(Dataset):
    def __init__(self, tif_dir, transforms=None):
        self.fps = glob.glob(os.path.join(tif_dir, '*.tif'))
        self.transforms = transforms

    def __getitem__(self, idx):
        img = imread(self.fps[idx])
        if self.transforms:
            img = self.transforms(image=img)['image']
        return img, self.fps[idx]

    def __len__(self):
        return len(self.fps)


def gen_dataloader(tif_dir, batch_size, num_workers, transforms=None):
    dataset = SegmentationDataset(tif_dir, transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
