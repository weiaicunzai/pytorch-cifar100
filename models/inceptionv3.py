""" inceptionv3 in pytorch


[1] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna

    Rethinking the Inception Architecture for Computer Vision
    https://arxiv.org/abs/1512.00567v3
"""

import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
