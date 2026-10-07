"""universally slimmable MobileNetV2 for 32x32 inputs

US-Net's MobileNetV2 (models/us_mobilenet_v2.py, for ImageNet) with the
CIFAR changes every CIFAR MobileNetV2 makes: the stem is 3x3 stride 1 and
the second stage keeps stride 1, so the network downsamples 8x instead of
32x and the last stage sees 4 pixels, not 1. Everything US-Net chose is
kept: the expansion layers slim with the width at the expansion ratio, the
depthwise convolution slims with them, and the last 1x1 convolution to 1280
channels does not slim (us=[True, False]), so every width ends in the same
1280-channel feature and the same classifier input.

The interface is us_resnet's, so train.py and the losses need nothing new:
head_groups gives each band of widths its own classifier (WBH), and with
return_features the forward also returns the pooled 1280-channel feature
after the last ReLU6, the 'final' tap, which is where the feature transport
reads. No other tap and no pre-activation read exist here.

feature_read picks which pooled feature 'final' is. 'head' (the default,
what EP ran) is the unslimmed 1280-channel output above. Every width then
has the full 1280 channels, so prefix alignment compares all of them and
width 0.25 is held to the teacher's whole feature. 'last_slimmed' reads
the output of the last inverted residual instead, 320 x width channels,
the last layer that slims: a narrow width is then compared with the
teacher's leading channels only, as on ResNet, where the read is the last
block's output and that block slims. That output is the block's linear
bottleneck, before any activation, where ResNet's is after a ReLU.
"""
import math

import torch.nn as nn

from .slimmable_ops import USBatchNorm2d, USConv2d, USLinear, make_divisible
from .us_mobilenet_v2 import InvertedResidual
from utils.config import FLAGS


class Model(nn.Module):
    def __init__(self, num_classes=100, input_size=32):
        super(Model, self).__init__()
        if getattr(FLAGS, 'feature_pre_relu', False):
            raise ValueError('us_mobilenet_v2_cifar has no pre-ReLU read')
        taps = tuple(getattr(FLAGS, 'feature_layers', ['final']))
        if taps != ('final',):
            raise ValueError('us_mobilenet_v2_cifar has only the final tap, '
                             'not {}'.format(taps))
        self.feature_read = getattr(FLAGS, 'feature_read', 'head')
        if self.feature_read not in ('head', 'last_slimmed'):
            raise ValueError('feature_read is head or last_slimmed, not '
                             '{}'.format(self.feature_read))

        # t, c, n, s; ImageNet's second stride of 2 becomes 1
        self.block_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        width_mult = FLAGS.width_mult_range[-1]
        assert input_size % 8 == 0
        channels = make_divisible(32 * width_mult)
        self.outp = make_divisible(
            1280 * width_mult) if width_mult > 1.0 else 1280

        features = [nn.Sequential(
            USConv2d(3, channels, 3, 1, 1, bias=False, us=[False, True]),
            USBatchNorm2d(channels),
            nn.ReLU6(inplace=True),
        )]
        for t, c, n, s in self.block_setting:
            outp = make_divisible(c * width_mult)
            for i in range(n):
                features.append(InvertedResidual(
                    channels, outp, s if i == 0 else 1, t))
                channels = outp
        features.append(nn.Sequential(
            USConv2d(channels, self.outp, 1, 1, 0, bias=False,
                     us=[True, False]),
            nn.BatchNorm2d(self.outp),
            nn.ReLU6(inplace=True),
        ))
        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)

        # WBH, as in us_resnet: one head per band of widths, the widest
        # band's head keeping the name classifier and registered last, so
        # get_classifier_weight and the teacher read it. The feature does
        # not slim, so no head slims either; they are USLinear only so that
        # model.apply gives them the width the band is chosen by.
        groups = getattr(FLAGS, 'head_groups', 1)
        if groups > 1:
            self.narrow_heads = nn.ModuleList([
                USLinear(self.outp, num_classes, us=[False, False])
                for _ in range(groups - 1)])
        self.classifier = nn.Sequential(
            USLinear(self.outp, num_classes, us=[False, False]))
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
        if (not getattr(FLAGS, 'return_features', False)
                or self.feature_read == 'head'):
            x = self.features(x)
            x = x.view(-1, self.outp)
            if not getattr(FLAGS, 'return_features', False):
                return self.head()(x)
            return self.head()(x), (x,)
        # the last inverted residual is third from the end, before the
        # unslimmed 1x1 and the pooling
        last_block = len(self.features) - 3
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index == last_block:
                feature = x.mean(dim=(2, 3))
        x = x.view(-1, self.outp)
        return self.head()(x), (feature,)

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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
