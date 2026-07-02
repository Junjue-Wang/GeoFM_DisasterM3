import torch
import numpy as np


def mean_std_normalization_totensor(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
    _mean = mean
    _std = std

    def _mean_std_normalization_totensor(NHWC_image):
        mean = _mean
        std = _std
        if isinstance(NHWC_image, torch.Tensor):
            pass
        elif isinstance(NHWC_image, np.ndarray):
            NHWC_image = torch.from_numpy(NHWC_image)

        NCHW_image = NHWC_image.permute(0, 3, 1, 2).float()

        dtype = NCHW_image.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=NCHW_image.device)
        std = torch.as_tensor(std, dtype=dtype, device=NCHW_image.device)
        return NCHW_image.sub(mean[None, :, None, None]).div(std[None, :, None, None])

    return _mean_std_normalization_totensor



