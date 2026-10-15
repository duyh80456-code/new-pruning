"""SOLAR (WACV 2026) as its authors run it, on our CIFAR and Tiny ResNets

github.com/sakt90/SOLAR builds on Slimmable Networks, not US-Net: a fixed
list of widths, a private BatchNorm per width (SwitchableBatchNorm2d), and
one classifier per width whose input is that width's own feature size
(their resnet_slimmable.py, SlimmableResNet_SOL.output_heads). train.py's
s-net branch already trains such a model their way, every width of the
list on every batch with cross entropy only.

The backbone is the one our US-Net rows use (models/us_resnet.py): the
CIFAR stem, 3x3 with stride 1 and no max pool, or stride 2 for 64x64 Tiny
ImageNet inputs (stem_stride), and depth 50 (bottlenecks, the block of
models/s_resnet.py) or 18 (basic blocks, as in their resnet_slimmable.py).
Channel counts are int(c * width), as in their code and in s_resnet, not
us_resnet's make_divisible, so a few widths carry a handful of channels
fewer than the US-Net rows do.
"""
import math

import torch.nn as nn

from .s_resnet import Block
from .slimmable_ops import (SwitchableBatchNorm2d, SlimmableConv2d,
                             SlimmableLinear)
from utils.config import FLAGS


class BasicBlock(nn.Module):
    """two 3x3 convolutions, slimmable, a private BN per width"""
    def __init__(self, inp, outp, stride):
        super(BasicBlock, self).__init__()
        self.body = nn.Sequential(
            SlimmableConv2d(inp, outp, 3, stride, 1, bias=False),
            SwitchableBatchNorm2d(outp),
            nn.ReLU(inplace=True),
            SlimmableConv2d(outp, outp, 3, 1, 1, bias=False),
            SwitchableBatchNorm2d(outp),
        )
        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
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
        depth = getattr(FLAGS, 'depth', 50)
        counts = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}
        if depth not in counts:
            raise ValueError('s_resnet_solar has depth 18 or 50, not '
                             '{}'.format(depth))
        expansion = 4 if depth == 50 else 1
        stem_stride = getattr(FLAGS, 'stem_stride', 1)
        widths = FLAGS.width_mult_list
        feats = [64, 128, 256, 512]
        channels = [int(64 * w) for w in widths]
        features = [nn.Sequential(
            SlimmableConv2d([3 for _ in widths], channels, 3, stem_stride, 1,
                            bias=False),
            SwitchableBatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )]
        for stage, n in enumerate(counts[depth]):
            outp = [int(feats[stage] * w * expansion) for w in widths]
            for i in range(n):
                stride = 2 if i == 0 and stage != 0 else 1
                if expansion == 1:
                    features.append(BasicBlock(channels, outp, stride))
                else:
                    features.append(Block(channels, outp, stride))
                channels = outp
        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)

        # one head per width, input the width's own feature size, as their
        # output_heads; a list, so the index matches width_mult_list.
        # shared_head is Slimmable Networks itself (models/s_resnet.py):
        # one classifier, its input sliced to the width's feature size.
        self.outp = channels
        self.shared_head = getattr(FLAGS, 'shared_head', False)
        if self.shared_head:
            self.classifier = SlimmableLinear(
                channels, [num_classes for _ in channels])
        else:
            self.heads = nn.ModuleList(
                [nn.Linear(c, num_classes) for c in channels])
        self.width_mult = max(widths)
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.shared_head:
            return self.classifier(x)
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
