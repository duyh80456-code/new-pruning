import math

import torch
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

        # One learnable scalar per tested width on the residual branch,
        # NeFL's inconsistent step size. It sits at the one place per-width
        # BN recalibration cannot reach: body and shortcut each end in a
        # BatchNorm, but the tensor they are added into has none, so the
        # branch-to-skip ratio on the residual stream drifts with width and
        # nothing here corrects it. 8 blocks by 16 widths is 128 scalars.
        # Init 1.0; NeFL reports larger initial values degrade.
        if getattr(FLAGS, 'width_scalars', False):
            self.branch_scale = nn.Parameter(
                torch.ones(len(FLAGS.width_mult_list_test)))
        else:
            self.branch_scale = None
        # set by model.apply the same way the slimmable ops get it; the
        # default keeps a forward before the first apply from failing
        self.width_mult = max(FLAGS.width_mult_list)

    def _slot(self):
        """which scalar this width uses

        Training draws widths from a continuum, so a width lands on the
        nearest of the sixteen tested slots. The sixteen evaluated widths
        hit their own slot exactly, which is what the table reads.
        """
        widths = FLAGS.width_mult_list_test
        here = self.width_mult
        return min(range(len(widths)), key=lambda i: abs(widths[i] - here))

    def forward(self, x):
        body = self.body(x)
        if self.branch_scale is not None:
            body = body * self.branch_scale[self._slot()]
        return self.post_relu(body + self.shortcut(x))


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

        # Where each stage ends, as an index into self.features. Taps are
        # read by walking that Sequential rather than by splitting it into
        # submodules, so the state dict is unchanged and checkpoints from
        # before any of this still load.
        self.stage_ends = {}
        index = 0
        for stage, (_, count, _) in enumerate(self.block_setting):
            index += count
            self.stage_ends['stage{}'.format(stage + 1)] = index

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
        if not getattr(FLAGS, 'return_features', False):
            x = self.features(x)
            return self.classifier(x.view(-1, x.size()[1]))

        # A tap is a stage name or 'final'. Everything but 'final' is a
        # spatial map, averaged over space so that one transport cost
        # serves every tier. The channel count differs by tier and by
        # width, which is why feature_cost has to be told how to align.
        taps = tuple(getattr(FLAGS, 'feature_layers', ['final']))
        wanted = {self.stage_ends[name]: name
                  for name in taps if name != 'final'}
        collected = {}
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in wanted:
                collected[wanted[index]] = x.mean(dim=(2, 3))
        x = x.view(-1, x.size()[1])
        collected['final'] = x
        return self.classifier(x), tuple(collected[name] for name in taps)

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
