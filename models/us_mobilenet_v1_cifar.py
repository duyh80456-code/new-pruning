"""universally slimmable MobileNetV1 for 32x32 inputs

US-Net's MobileNetV1 (models/us_mobilenet_v1.py, for ImageNet) with the
CIFAR changes us_mobilenet_v2_cifar makes: the stem is 3x3 stride 1 and the
first stage that downsamples on ImageNet (128 channels) keeps stride 1, so
the network downsamples 8x instead of 32x and the last stage sees 4 pixels.
Everything US-Net chose is kept, and here that includes the last layer:
unlike MobileNetV2, MobileNetV1 slims its last pointwise convolution to
1024 x width and the classifier's input with it (USLinear us=[True,
False]), as ResNet does with its last block.

The interface is us_resnet's and us_mobilenet_v2_cifar's: head_groups
gives each band of widths its own classifier (WBH), and with
return_features the forward also returns the pooled 1024 x width feature
after the last ReLU6, the 'final' tap the feature transport reads. A
narrow width therefore meets the teacher on its leading channels, as on
ResNet. No other tap and no pre-activation read exist here.
"""
import math

import torch.nn as nn

from .slimmable_ops import USBatchNorm2d, USConv2d, USLinear, make_divisible
from .us_mobilenet_v1 import DepthwiseSeparableConv
from utils.config import FLAGS


class Model(nn.Module):
    def __init__(self, num_classes=100, input_size=32):
        super(Model, self).__init__()
        if getattr(FLAGS, 'feature_pre_relu', False):
            raise ValueError('us_mobilenet_v1_cifar has no pre-ReLU read')
        taps = tuple(getattr(FLAGS, 'feature_layers', ['final']))
        if taps != ('final',):
            raise ValueError('us_mobilenet_v1_cifar has only the final tap, '
                             'not {}'.format(taps))

        # c, n, s; ImageNet's first stride of 2 after the stem becomes 1
        self.block_setting = [
            [64, 1, 1],
            [128, 2, 1],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2],
        ]

        width_mult = FLAGS.width_mult_range[-1]
        assert input_size % 8 == 0
        channels = make_divisible(32 * width_mult)
        self.outp = make_divisible(1024 * width_mult)

        features = [nn.Sequential(
            USConv2d(3, channels, 3, 1, 1, bias=False, us=[False, True]),
            USBatchNorm2d(channels),
            nn.ReLU6(inplace=True),
        )]
        for c, n, s in self.block_setting:
            outp = make_divisible(c * width_mult)
            for i in range(n):
                features.append(DepthwiseSeparableConv(
                    channels, outp, s if i == 0 else 1))
                channels = outp
        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)

        # WBH, as in us_resnet: one head per band of widths, the widest
        # band's head keeping the name classifier and registered last, so
        # get_classifier_weight and the teacher read it. Every head takes
        # the slimmed feature, as the published classifier does.
        groups = getattr(FLAGS, 'head_groups', 1)
        if groups > 1:
            self.narrow_heads = nn.ModuleList([
                USLinear(self.outp, num_classes, us=[True, False])
                for _ in range(groups - 1)])
        self.classifier = nn.Sequential(
            USLinear(self.outp, num_classes, us=[True, False]))
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if not getattr(FLAGS, 'return_features', False):
            return self.head()(x)
        return self.head()(x), (x,)

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
