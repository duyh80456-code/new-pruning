"""SOLAR (WACV 2026) as its authors run it, on our CIFAR MobileNetV2

The SOLAR recipe of models/s_resnet_solar.py (Slimmable Networks base, a
private BatchNorm per width, one head per width) on the backbone of
models/us_mobilenet_v2_cifar.py with slim_last, the MobileNetV2 our SOLAR
rows used: 3x3 stride 1 stem, the second stage at stride 1, and the last
1x1 convolution to 1280 slimmed with the width, so width w ends in
1280 x w channels and its head reads exactly those. The inverted residual
is upstream's s-net block (models/s_mobilenet_v2.py), channel counts
make_divisible as there.
"""
import math

import torch.nn as nn

from .s_mobilenet_v2 import InvertedResidual
from .slimmable_ops import SwitchableBatchNorm2d, SlimmableConv2d
from .slimmable_ops import make_divisible
from utils.config import FLAGS


class Model(nn.Module):
    def __init__(self, num_classes=100, input_size=32):
        super(Model, self).__init__()
        if not getattr(FLAGS, 'slim_last', False):
            raise ValueError('s_mobilenet_v2_solar is the slim_last model; '
                             'set slim_last: True')
        # t, c, n, s; ImageNet's second stride of 2 becomes 1
        block_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        widths = FLAGS.width_mult_list
        channels = [make_divisible(32 * w) for w in widths]
        features = [nn.Sequential(
            SlimmableConv2d([3 for _ in widths], channels, 3, 1, 1,
                            bias=False),
            SwitchableBatchNorm2d(channels),
            nn.ReLU6(inplace=True),
        )]
        for t, c, n, s in block_setting:
            outp = [make_divisible(c * w) for w in widths]
            for i in range(n):
                features.append(InvertedResidual(
                    channels, outp, s if i == 0 else 1, t))
                channels = outp
        self.outp = [make_divisible(1280 * w) for w in widths]
        features.append(nn.Sequential(
            SlimmableConv2d(channels, self.outp, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(self.outp),
            nn.ReLU6(inplace=True),
        ))
        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)

        self.heads = nn.ModuleList(
            [nn.Linear(c, num_classes) for c in self.outp])
        self.width_mult = max(widths)
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.heads[FLAGS.width_mult_list.index(self.width_mult)](x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
