import torch
from PIL import Image
import numpy as np


def viz_binary_tp_fp_fn(y_pred, y_true):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()

    mix = y_true * 2 + y_pred

    # tp = mix == 3, green
    # fp = mix == 1, red
    # fn = mix == 2, blue

    mix = Image.fromarray(mix)

    mix.putpalette([0, 0, 0,
                    255, 0, 0,
                    0, 0, 255,
                    0, 255, 0])
    mix = mix.convert('RGB')
    mix = np.array(mix)
    return mix
