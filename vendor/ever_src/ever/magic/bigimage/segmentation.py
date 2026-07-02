import torch
from ever.core import to
from ever.core.device import auto_device
from .autopatch.core import autopatch, PatchGenerator
from .autopatch.merge_ops import average_merge_fn, AverageMemmapMerge
from .autopatch.preprocess_fn import mean_std_normalization_totensor
from tqdm import tqdm

__all__ = ['AutoPatchSegm',
           'AutoPatchMapping']


class AutoPatchSegm(autopatch):
    def __init__(self,
                 module,
                 config,
                 image_size,
                 preprocess_fn=mean_std_normalization_totensor(),
                 ensemble_transforms=None,
                 ensemble_fn=None):
        super(AutoPatchSegm, self).__init__(module, config,
                                            preprocess_fn=preprocess_fn,
                                            merge_fn=average_merge_fn(*image_size),
                                            ensemble_transforms=ensemble_transforms,
                                            ensemble_fn=ensemble_fn)


class AutoPatchMapping(autopatch):
    def __init__(self,
                 module,
                 config,
                 output_path,
                 preprocess_fn=mean_std_normalization_totensor(),
                 ensemble_transforms=None,
                 ensemble_fn=None,
                 merge_class=AverageMemmapMerge
                 ):
        super(AutoPatchMapping, self).__init__(module,
                                               config,
                                               preprocess_fn=preprocess_fn,
                                               merge_fn=None,
                                               ensemble_transforms=ensemble_transforms,
                                               ensemble_fn=ensemble_fn)
        self.output_path = output_path
        self.merge_class = merge_class

    def forward(self, image, raw_out=False):
        memmap_merge = None
        cfg = self.cfg
        cpu_device = torch.device('cpu')
        dataloader = PatchGenerator(image,
                                    kernel_size=cfg.kernel_size,
                                    stride=cfg.stride,
                                    batch_size=cfg.batch_size,
                                    distributed=cfg.distributed)
        if cfg.progress_bar:
            dataloader = tqdm(dataloader)
        device = auto_device()
        with torch.no_grad():
            for im, box in dataloader:
                im = to.to_device(im, device)
                pim = self.preprocess_fn(im)
                if self.ensemble_transforms is not None and len(self.ensemble_transforms) > 0:
                    trans_outs = [to.to_device(self.module(t.transform(pim)), cpu_device) for t in
                                  self.ensemble_transforms]
                    outs = [t.inv_transform(tout) for t, tout in zip(self.ensemble_transforms, trans_outs)]
                    out = self.ensemble_fn(outs)
                else:
                    out = self.module(pim)

                out = to.to_device(out, cpu_device)

                if memmap_merge is None:
                    c = out.size(1)
                    dtype = out.numpy().dtype
                    if isinstance(image, list) or isinstance(image, tuple):
                        h = image[0].shape[0]
                        w = image[0].shape[1]
                    else:
                        h = image.shape[0]
                        w = image.shape[1]
                    memmap_shape = (h, w, c)
                    memmap_merge = self.merge_class(self.output_path, memmap_shape, dtype)

                out = out.permute(0, 2, 3, 1)
                for o, b in zip(out, box):
                    memmap_merge.merge(o, b)

            memmap_merge.normalize(raw_out)
