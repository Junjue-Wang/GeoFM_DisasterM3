import torch
from functools import partial
import tifffile


def merge_segmentation_fn(image_height, image_width):
    def _merge(h, w, out_list):
        pred_list, win_list = list(zip(*out_list))

        num_classes = pred_list[0].size(0)
        device = pred_list[0].device
        res_img = torch.zeros(num_classes, h, w, dtype=torch.float32, device=device)

        for pred, win in zip(pred_list, win_list):
            res_img[:, win[1]:win[3], win[0]: win[2]] = pred

        return res_img

    return partial(_merge, image_height, image_width)


class OverwriteMemmapMerge(object):
    def __init__(self, filename, memmap_shape, dtype):
        assert len(memmap_shape) == 3
        self.h = memmap_shape[0]
        self.w = memmap_shape[1]
        self.c = memmap_shape[2]
        self.filename = filename

        self.mem_prob = tifffile.memmap(self.filename, shape=memmap_shape, dtype=dtype)

    def merge(self, data, box):
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        xmin, ymin, xmax, ymax = box

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        self.mem_prob[ymin:ymax, xmin:xmax] = data

    def normalize(self, raw_out=False):
        return
