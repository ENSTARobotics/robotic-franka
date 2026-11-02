"""mini-Xception 48x48 in PyTorch, equivalent to the Keras version of Joab's ROB311 lab (TD4).

Original architecture and weights: Octavio Arriaga et al., MIT
(https://github.com/oarriaga/face_classification); head adapted to 48x48 by Joab.
The weights are converted once from Keras with convert_weights.py -> models/mini_xception_48.pt.

Details so that it matches Keras:
- Keras's BatchNormalization uses epsilon 1e-3 (PyTorch, 1e-5 by default).
- Keras's "same" padding in MaxPool 3x3 stride 2 is asymmetric (the extra goes right/bottom).
"""
import torch
from torch import nn
import torch.nn.functional as F

BN_EPS = 1e-3


def bn(channels):
    return nn.BatchNorm2d(channels, eps=BN_EPS)


class SeparableConv(nn.Module):
    """Keras's SeparableConv2D: depthwise 3x3 (same padding) + pointwise 1x1, no bias."""

    def __init__(self, inputs, outputs):
        super().__init__()
        self.depthwise = nn.Conv2d(inputs, inputs, 3, padding=1, groups=inputs, bias=False)
        self.pointwise = nn.Conv2d(inputs, outputs, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def maxpool_same(x, k=3, s=2):
    """Keras's MaxPooling2D(3, strides=2, padding="same")."""
    pads = []
    for n in (x.shape[3], x.shape[2]):              # F.pad: width first, then height
        out = -(-n // s)
        total = max((out - 1) * s + k - n, 0)
        pads += [total // 2, total - total // 2]
    return F.max_pool2d(F.pad(x, pads, value=float("-inf")), k, s)


class Block(nn.Module):
    def __init__(self, inputs, filters):
        super().__init__()
        self.residual = nn.Conv2d(inputs, filters, 1, stride=2, bias=False)
        self.bn_residual = bn(filters)
        self.sep1, self.bn1 = SeparableConv(inputs, filters), bn(filters)
        self.sep2, self.bn2 = SeparableConv(filters, filters), bn(filters)

    def forward(self, x):
        residual = self.bn_residual(self.residual(x))
        x = F.relu(self.bn1(self.sep1(x)))
        x = maxpool_same(self.bn2(self.sep2(x)))
        return x + residual


class MiniXception(nn.Module):
    """Input (N, 1, 48, 48) normalized to [-1, 1]; output: probabilities of 7 classes."""

    def __init__(self):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(1, 8, 3, bias=False), bn(8)
        self.conv2, self.bn2 = nn.Conv2d(8, 8, 3, bias=False), bn(8)
        self.blocks = nn.ModuleList([Block(i, f) for i, f in ((8, 16), (16, 32), (32, 64), (64, 128))])
        self.final = nn.Conv2d(128, 7, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        for block in self.blocks:
            x = block(x)
        return F.softmax(self.final(x).mean(dim=(2, 3)), dim=1)


def load(path, device):
    model = MiniXception()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()
