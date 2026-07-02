from collections import namedtuple

import torch
import torch.distributed as dist
import torch.nn as nn
from ever.api.data.distributed import DistributedNonOverlapSeqSampler
from ever.core import dist as er_dist, to
from ever.interface.transform_base import Transform
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from tqdm import tqdm

from ..basic_dataset import BigImagePatchData, ParallelBigImagePatchDataset

__all__ = ['PatchGenerator', 'autopatch', 'AutoPatchConfig']


class AutoPatchConfig(namedtuple('_AutoPatchConfig',
                                 ['kernel_size',
                                  'stride',
                                  'distributed',
                                  'batch_size',
                                  'num_workers',
                                  'progress_bar',
                                  'local_rank'])):
    def __new__(cls,
                kernel_size,
                stride,
                distributed=False,
                batch_size=1,
                num_workers=0,
                progress_bar=True,
                local_rank=0):
        return super().__new__(cls,
                               kernel_size,
                               stride,
                               distributed,
                               batch_size,
                               num_workers,
                               progress_bar,
                               local_rank)


class PatchGenerator(DataLoader):
    def __init__(self, image, kernel_size, stride, batch_size, transforms=None, distributed=False, num_workers=0):
        if isinstance(image, list) or isinstance(image, tuple):
            datasets = [BigImagePatchData(im, kernel_size, stride, transforms) for im in image]
            dataset = ParallelBigImagePatchDataset(datasets)
        else:
            dataset = BigImagePatchData(image, kernel_size, stride, transforms)
        if distributed:
            sampler = DistributedNonOverlapSeqSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        super(PatchGenerator, self).__init__(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)


class autopatch(nn.Module):
    """
    model = nn.Sequential(
        nn.Conv2d(7, 1, 3, 1, 1)
    )
    import numpy as np
    from ever.magic.transform import segm

    image = np.ones([4000, 4000, 7])

    model = autopatch(model,
                   config=AutoPatchConfig((512, 512), 256, distributed=False,
                                          batch_size=2, progress_bar=True),
                   preprocess_fn=lambda x: x.permute(0, 3, 1, 2).float(),
                   ensemble_transforms=[segm.HorizontalFlip(), segm.VerticalFlip(), segm.Identity()],
                   ensemble_fn=lambda out_list: torch.stack(out_list, dim=0).mean(dim=0),
                   merge_fn=lambda out_list: out_list):
    out = model(image)
    """

    def __init__(self,
                 module: nn.Module,
                 config: AutoPatchConfig,
                 preprocess_fn=lambda x: x,
                 merge_fn=lambda output_list: output_list,
                 ensemble_transforms=None,
                 ensemble_fn=None):
        super(autopatch, self).__init__()
        self.module = module
        self.cfg = config

        self.preprocess_fn = preprocess_fn
        self.ensemble_transforms = ensemble_transforms
        if ensemble_transforms is not None and len(ensemble_transforms) > 0:
            assert all([isinstance(t, Transform) for t in ensemble_transforms])
            assert ensemble_fn is not None
        self.ensemble_fn = ensemble_fn
        self.merge_fn = merge_fn

        if config.distributed:
            # init nccl
            torch.cuda.set_device(config.local_rank)
            dist.init_process_group(
                backend="nccl", init_method="env://"
            )

    def forward(self, image, buffer_device='cpu'):
        cfg = self.cfg
        buffer_device = torch.device(buffer_device)
        gpu_device = torch.device('cuda')
        dataloader = PatchGenerator(image,
                                    kernel_size=cfg.kernel_size,
                                    stride=cfg.stride,
                                    batch_size=cfg.batch_size,
                                    distributed=cfg.distributed,
                                    num_workers=cfg.num_workers)
        if cfg.progress_bar:
            dataloader = tqdm(dataloader)
        out_list = []
        with torch.no_grad():
            for im, box in dataloader:
                im = to.to_device(im, gpu_device)
                pim = self.preprocess_fn(im)
                if self.ensemble_transforms is not None and len(self.ensemble_transforms) > 0:
                    trans_outs = [to.to_device(self.module(t.transform(pim)), buffer_device) for t in
                                  self.ensemble_transforms]
                    outs = [t.inv_transform(tout) for t, tout in zip(self.ensemble_transforms, trans_outs)]
                    out = self.ensemble_fn(outs)
                else:
                    out = self.module(pim)

                out = to.to_device(out, buffer_device)
                if cfg.distributed:
                    multideive_blob = er_dist.all_gather((out, box))
                    for blob in multideive_blob:
                        _out, _box = blob
                        for o, b in zip(_out, _box):
                            out_list.append((o, b))
                else:
                    for o, b in zip(out, box):
                        out_list.append((o, b))
            ret = self.merge_fn(out_list)
        return ret
