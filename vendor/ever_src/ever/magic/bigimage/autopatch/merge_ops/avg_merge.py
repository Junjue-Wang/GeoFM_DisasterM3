import torch
from functools import partial
import tifffile
import tempfile
import numpy as np
from ever.magic.bigimage.sliding_window import sliding_window


def average_merge_fn(image_height, image_width):
    def _merge(h, w, out_list):
        pred_list, win_list = list(zip(*out_list))

        num_classes = pred_list[0].size(0)
        device = pred_list[0].device
        res_img = torch.zeros(num_classes, h, w, dtype=torch.float32, device=device)
        res_count = torch.zeros(h, w, dtype=torch.float32, device=device)

        for pred, win in zip(pred_list, win_list):
            res_count[win[1]:win[3], win[0]: win[2]] += 1
            res_img[:, win[1]:win[3], win[0]: win[2]] += pred

        avg_res_img = res_img / res_count.unsqueeze_(0)

        return avg_res_img

    return partial(_merge, image_height, image_width)


class AverageMemmapMerge(object):
    def __init__(self, filename, memmap_shape, dtype):
        assert len(memmap_shape) == 3
        self.h = memmap_shape[0]
        self.w = memmap_shape[1]
        self.c = memmap_shape[2]
        self.filename = filename
        self.tmp_count_fp = tempfile.TemporaryFile()
        self.mem_count = np.memmap(self.tmp_count_fp, shape=(self.h, self.w, 1), dtype=dtype, mode='w+')

        self.tmp_prob_fp = tempfile.TemporaryFile()
        self.mem_prob = np.memmap(self.tmp_prob_fp, shape=memmap_shape, dtype=np.float32, mode='w+')

    def merge(self, prob, box):
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        xmin, ymin, xmax, ymax = box

        if isinstance(prob, torch.Tensor):
            prob = prob.cpu().numpy()
        self.mem_prob[ymin:ymax, xmin:xmax] += prob
        self.mem_count[ymin:ymax, xmin:xmax] += 1.

    def normalize(self, raw_out=False):
        if raw_out:
            mem_mask = tifffile.memmap(self.filename, shape=(self.h, self.w, self.c), dtype=np.float32)
        else:
            mem_mask = tifffile.memmap(self.filename, shape=(self.h, self.w), dtype=np.uint8)
        for i, box in enumerate(sliding_window((self.h, self.w), (1024, 1024), 1024)):
            xmin, ymin, xmax, ymax = box
            local_prob = np.asarray(self.mem_prob[ymin:ymax, xmin:xmax], dtype=np.float32)
            local_count = np.asarray(self.mem_count[ymin:ymax, xmin:xmax])
            avg_prob = local_prob / local_count
            if not raw_out:
                if local_prob.shape[2] == 1:
                    mask = (avg_prob > 0.5).squeeze(2).astype(np.uint8, copy=False)
                else:
                    mask = avg_prob.argmax(axis=2).astype(np.uint8, copy=False)
            else:
                mask = avg_prob
            mem_mask[ymin:ymax, xmin:xmax] = mask
            if i % 100 == 0:
                mem_mask.flush()
        mem_mask.flush()

    def __del__(self):
        self.tmp_count_fp.close()
        self.tmp_prob_fp.close()
