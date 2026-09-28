import math

import torch
import torch.nn as nn

from .slimmable_ops import USBatchNorm2d, USConv2d, USLinear, make_divisible
from utils.config import FLAGS


class _Residual(nn.Module):
    """body(x) + shortcut(x), then a ReLU; the two block types differ only
    in the body"""

    def _finish(self, inp, outp, stride):
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


class BasicBlock(_Residual):
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
        self._finish(inp, outp, stride)


class Bottleneck(_Residual):
    """1x1 down, 3x3, 1x1 up by four: the ResNet-50 block

    The stride sits on the 3x3, as in torchvision, rather than on the first
    1x1 as in the original paper. mid and outp are both built at the
    widest width and slimmed by the same multiplier at run time, so the
    four-to-one ratio holds at every width up to make_divisible rounding.
    """
    def __init__(self, inp, mid, outp, stride):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2]

        self.body = nn.Sequential(
            USConv2d(inp, mid, 1, 1, 0, bias=False),
            USBatchNorm2d(mid),
            nn.ReLU(inplace=True),

            USConv2d(mid, mid, 3, stride, 1, bias=False),
            USBatchNorm2d(mid),
            nn.ReLU(inplace=True),

            USConv2d(mid, outp, 1, 1, 0, bias=False),
            USBatchNorm2d(outp),
        )
        self._finish(inp, outp, stride)


class Model(nn.Module):
    """universally slimmable ResNet for 32x32 inputs

    The CIFAR stem: 3x3 stride 1 and no max pool, so the four stages see
    32, 16, 8 and 4 pixels. `depth` picks the block: 18 is [2, 2, 2, 2]
    basic blocks and the default, so every config written before this
    flag builds exactly what it built before; 50 is [3, 4, 6, 3]
    bottlenecks, four times as wide at the output of every stage.
    """
    def __init__(self, num_classes=100, input_size=32):
        super(Model, self).__init__()

        depth = getattr(FLAGS, 'depth', 18)
        counts = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}
        if depth not in counts:
            raise ValueError('us_resnet has depth 18 or 50, not {}'.format(
                depth))
        # c, n, s
        self.block_setting = [
            [c, n, s] for (c, s), n in zip(
                [(64, 1), (128, 2), (256, 2), (512, 2)], counts[depth])]
        expansion = 4 if depth == 50 else 1

        width_mult = FLAGS.width_mult_range[-1]
        assert input_size % 8 == 0
        channels = make_divisible(64 * width_mult)
        self.outp = make_divisible(512 * expansion * width_mult)

        # The stem takes RGB, which does not slim, hence us=[False, True].
        features = [nn.Sequential(
            USConv2d(3, channels, 3, 1, 1, bias=False, us=[False, True]),
            USBatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )]

        for c, n, s in self.block_setting:
            outp = make_divisible(c * expansion * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                if expansion == 1:
                    block = BasicBlock(channels, outp, stride)
                else:
                    block = Bottleneck(channels,
                                       make_divisible(c * width_mult),
                                       outp, stride)
                features.append(block)
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

        # SOLAR (WACV 2026) gives every subnet its own classifier over a
        # shared backbone. Its subnets are a fixed handful; here the width
        # is continuous, so the range is cut into head_groups equal bands
        # and each band gets one head. The band holding the widest width
        # keeps the name classifier, so head_groups 1 is the model as it
        # was, key for key. The narrow heads are registered first on
        # purpose: get_classifier_weight takes the last Linear it meets,
        # and that has to stay the widest width's head, the one the
        # teacher reads.
        groups = getattr(FLAGS, 'head_groups', 1)
        if groups > 1:
            self.narrow_heads = nn.ModuleList([
                USLinear(self.outp, num_classes, us=[True, False])
                for _ in range(groups - 1)])

        # us=[True, False]: the input side slims with the width, the number
        # of classes does not. Teacher and student therefore share these
        # rows, which is what the class cost matrix in utils/loss_ops.py
        # relies on.
        self.classifier = nn.Sequential(
            USLinear(self.outp, num_classes, us=[True, False])
        )
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def head(self):
        """the classifier for the width the model is set to"""
        if not hasattr(self, 'narrow_heads'):
            return self.classifier
        groups = len(self.narrow_heads) + 1
        low, high = FLAGS.width_mult_range
        width = self.classifier[0].width_mult
        band = min(int((width - low) / (high - low) * groups), groups - 1)
        if band == groups - 1:
            return self.classifier
        return self.narrow_heads[band]

    def forward(self, x):
        if not getattr(FLAGS, 'return_features', False):
            x = self.features(x)
            return self.head()(x.view(-1, x.size()[1]))

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
        return self.head()(x), tuple(collected[name] for name in taps)

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
