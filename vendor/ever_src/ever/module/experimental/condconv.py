# ref https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py
import torch
import torch.nn as nn


class CondConv2d(nn.Conv2d):
    pass
