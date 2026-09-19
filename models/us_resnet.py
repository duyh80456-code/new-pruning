import math

import torch.nn as nn

from .slimmable_ops import USBatchNorm2d, USConv2d, USLinear, make_divisible
from utils.config import FLAGS


class BasicBlock(nn.Module):
    def __init__(self, inp, outp, stride):
        super(BasicBlock, self).__init__()
        assert stride in [1, 2]

        self.body = nn.Sequential(
            USConv2d(inp, outp, 3, stride, 1, bias=False),
            USBatchNorm2d(outp),
            nn.ReLU(inplace=True),

            USConv2d(outp, outp, 3, 1, 1, bias=False),
            USBatchNorm2d(outp),
        )

        # Both branches slim by the same width_mult, so their channel counts
        # agree at every width and the identity shortcut stays valid.
        if stride != 1 or inp != outp:
            self.shortcut = nn.Sequential(
                USConv2d(inp, outp, 1, stride, 0, bias=False),
                USBatchNorm2d(outp),
            )
        else:
            self.shortcut = nn.Sequential()
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.post_relu(self.body(x) + self.shortcut(x))


class Model(nn.Module):
    """universally slimmable ResNet for 32x32 inputs

    The CIFAR stem: 3x3 stride 1 and no max pool, so the four stages see
    32, 16, 8 and 4 pixels. Depth [2, 2, 2, 2] is ResNet-18.
    """
    def __init__(self, num_classes=100, input_size=32):
        super(Model, self).__init__()

        # c, n, s
        self.block_setting = [
            [64, 2, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 2, 2],
        ]

        width_mult = FLAGS.width_mult_range[-1]
        assert input_size % 8 == 0
        channels = make_divisible(64 * width_mult)
        self.outp = make_divisible(512 * width_mult)

        # The stem takes RGB, which does not slim, hence us=[False, True].
        features = [nn.Sequential(
            USConv2d(3, channels, 3, 1, 1, bias=False, us=[False, True]),
            USBatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )]

        for c, n, s in self.block_setting:
            outp = make_divisible(c * width_mult)
            for i in range(n):
                features.append(
                    BasicBlock(channels, outp, s if i == 0 else 1))
                channels = outp

        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)

        # us=[True, False]: the input side slims with the width, the number
        # of classes does not. Teacher and student therefore share these
        # rows, which is what the class cost matrix in utils/loss_ops.py
        # relies on.
        self.classifier = nn.Sequential(
            USLinear(self.outp, num_classes, us=[True, False])
        )
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        last_dim = x.size()[1]
        x = x.view(-1, last_dim)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
