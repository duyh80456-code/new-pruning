"""AutoSlim step 3 on CIFAR-100 ResNet-50: the searched configurations
trained as one slimmable network

As the authors' models/autoslim_resnet.py (JiahuiYu/slimmable_networks):
one slimmable network over a fixed list of configurations, each with its
own channel count in every layer, a private BatchNorm per configuration
(SwitchableBatchNorm2d), one classifier whose input is sliced to the
configuration's feature size, and every configuration trained on every
batch (train.py's s-net branch, slimmable_training without
universally_slimmable_training), the narrower ones distilled from the
widest (inplace_distill).

The configurations come from scripts/autoslim_search.py, one per width of
width_mult_list: the entry for width w is the searched network whose MACs
match our uniform width w. width_mult_list is therefore only a label, the
index of a configuration; the channel counts are read from
FLAGS.autoslim_configs, the search's JSON. The backbone is the one the
other rows use: the CIFAR stem (3x3, stride 1, no max pool) and the
bottleneck with the stride on the 3x3.
"""
import json
import math

import torch.nn as nn

from .slimmable_ops import (SwitchableBatchNorm2d, SlimmableConv2d,
                             SlimmableLinear)
from utils.config import FLAGS

STAGES = [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]


def load_configs():
    with open(FLAGS.autoslim_configs) as f:
        result = json.load(f)
    by_width = dict(zip(result['widths'], result['configs']))
    missing = [w for w in FLAGS.width_mult_list if w not in by_width]
    if missing:
        raise ValueError('no searched configuration for widths {}'.format(
            missing))
    return [by_width[w]['channels'] for w in FLAGS.width_mult_list]


class Block(nn.Module):
    def __init__(self, inp, mid1, mid2, outp, stride, shortcut):
        super(Block, self).__init__()
        self.body = nn.Sequential(
            SlimmableConv2d(inp, mid1, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(mid1),
            nn.ReLU(inplace=True),
            SlimmableConv2d(mid1, mid2, 3, stride, 1, bias=False),
            SwitchableBatchNorm2d(mid2),
            nn.ReLU(inplace=True),
            SlimmableConv2d(mid2, outp, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(outp),
        )
        self.residual_connection = not shortcut
        if shortcut:
            self.shortcut = nn.Sequential(
                SlimmableConv2d(inp, outp, 1, stride=stride, bias=False),
                SwitchableBatchNorm2d(outp),
            )
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.body(x)
        res = res + (x if self.residual_connection else self.shortcut(x))
        return self.post_relu(res)


class Model(nn.Module):
    def __init__(self, num_classes=100, input_size=32):
        super(Model, self).__init__()
        if getattr(FLAGS, 'depth', 50) != 50:
            raise ValueError('autoslim_resnet_cifar is ResNet-50 only')
        configs = load_configs()

        def column(name):
            return [c[name] for c in configs]

        channels = column('stem')
        features = [nn.Sequential(
            SlimmableConv2d([3 for _ in configs], channels, 3, 1, 1,
                            bias=False),
            SwitchableBatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )]
        for s, (_, n, stride) in enumerate(STAGES):
            outp = column('out{}'.format(s + 1))
            for b in range(n):
                prefix = 's{}b{}'.format(s + 1, b + 1)
                features.append(Block(
                    channels, column(prefix + 'm1'), column(prefix + 'm2'),
                    outp, stride if b == 0 else 1, shortcut=(b == 0)))
                channels = outp
        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)
        self.outp = channels
        self.classifier = SlimmableLinear(
            channels, [num_classes for _ in channels])
        self.width_mult = max(FLAGS.width_mult_list)
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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
