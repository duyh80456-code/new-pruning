import torch
import torch.nn as nn


from utils.config import FLAGS


class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features_list):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(FLAGS.width_mult_list)
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = FLAGS.width_mult_list.index(self.width_mult)
        y = self.bn[idx](input)
        return y


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input):
        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input):
        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def isolated(width_mult):
    """Scala's isolated activation: the narrowest width takes the last
    channels of every layer instead of the first

    Zhang et al., NeurIPS 2024. Their models_scala.py slices weight[-k:]
    at the smallest ratio and weight[:k] everywhere else, so the narrowest
    subnet shares no leading channels with the widths just above it. At
    the widest the two slices are the same tensor.
    """
    return (getattr(FLAGS, 'isolate_smallest', False)
            and width_mult == FLAGS.width_mult_range[0])


class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True,
                 us=[True, True], ratio=[1, 1]):
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = None
        self.us = us
        self.ratio = ratio

    def forward(self, input):
        if self.us[0]:
            self.in_channels = make_divisible(
                self.in_channels_max
                * self.width_mult
                / self.ratio[0]) * self.ratio[0]
        if self.us[1]:
            self.out_channels = make_divisible(
                self.out_channels_max
                * self.width_mult
                / self.ratio[1]) * self.ratio[1]
        self.groups = self.in_channels if self.depthwise else 1
        if isolated(self.width_mult):
            if self.depthwise:
                raise NotImplementedError(
                    'isolate_smallest is not written for depthwise convs')
            rows = slice(self.weight.size(0) - self.out_channels, None)
            cols = slice(self.weight.size(1) - self.in_channels, None)
        else:
            rows = slice(0, self.out_channels)
            cols = slice(0, self.in_channels)
        weight = self.weight[rows, cols, :, :]
        if self.bias is not None:
            bias = self.bias[rows]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        if getattr(FLAGS, 'conv_averaged', False):
            # US-Net Appendix A: divide the output by how many input
            # channels are live, so the pre-BN scale does not grow with
            # width. Inherited from the upstream slimmable repo reading
            # self.in_channels_list, which SlimmableConv2d has and
            # USConv2d does not, so switching the flag on raised
            # AttributeError and nothing here had ever run it.
            #
            # Worth knowing what it can and cannot be: every conv in this
            # model is followed by BatchNorm, and a per-output-channel
            # constant in front of BN is removed exactly by the
            # per-width running statistics this project already
            # recalibrates. The paper says as much - the constants
            # "come for free since these constants can be merged into BN
            # statistics after training". So this is forward-invisible at
            # inference and whatever it does, it does through the
            # gradient scale during training.
            y = y * (self.in_channels_max / self.in_channels)
        return y


class USLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, us=[True, True]):
        super(USLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.width_mult = None
        self.us = us

    def forward(self, input):
        if self.us[0]:
            self.in_features = make_divisible(
                self.in_features_max * self.width_mult)
        if self.us[1]:
            self.out_features = make_divisible(
                self.out_features_max * self.width_mult)
        if isolated(self.width_mult):
            rows = slice(self.weight.size(0) - self.out_features, None)
            cols = slice(self.weight.size(1) - self.in_features, None)
        else:
            rows = slice(0, self.out_features)
            cols = slice(0, self.in_features)
        weight = self.weight[rows, cols]
        if self.bias is not None:
            bias = self.bias[rows]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1):
        super(USBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=False)
        self.num_features_max = num_features
        # for tracking performance during training
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(i, affine=False) for i in [
                make_divisible(
                    self.num_features_max * width_mult / ratio) * ratio
                for width_mult in FLAGS.width_mult_list]])
        self.ratio = ratio
        self.width_mult = None
        self.ignore_model_profiling = True
        # counted per width, because calibration walks every width on each
        # batch and a single counter would advance len(width_mult_list)
        # times per batch
        self.calibration_batches = [0] * len(FLAGS.width_mult_list)
        # The scale and shift as continuous functions of the width.
        # US-Net shares one gamma and one beta across every width and
        # recalibrates only the running statistics, so whatever a width
        # needs from the affine half it cannot have. BP gave each of the
        # sixteen test widths one private scalar per residual branch,
        # snapped to the nearest slot, and came out on top of the table.
        # This is the same freedom made continuous and per channel: an
        # offset to gamma and to beta at each of affine_knots evenly
        # spaced widths, interpolated linearly in between, so every width
        # in the range has its own affine and neighbouring widths have
        # nearly the same one. The offsets start at zero, so epoch zero
        # is the plain model. They are one-dimensional per knot, so
        # get_optimizer leaves them out of weight decay the way it leaves
        # out gamma and beta.
        knots = getattr(FLAGS, 'affine_knots', 0)
        if knots:
            if knots < 2:
                raise ValueError('affine_knots needs at least two knots')
            self.knot_weight = nn.ParameterList([
                nn.Parameter(torch.zeros(num_features))
                for _ in range(knots)])
            self.knot_bias = nn.ParameterList([
                nn.Parameter(torch.zeros(num_features))
                for _ in range(knots)])
        else:
            self.knot_weight = None
            self.knot_bias = None

    def knot_mix(self):
        """how much each knot contributes at this width: a hat function
        of the distance to it, so at most two are non-zero and they sum
        to one"""
        low, high = FLAGS.width_mult_range
        count = len(self.knot_weight)
        spot = (self.width_mult - low) / (high - low) * (count - 1)
        spot = min(max(spot, 0.0), count - 1.0)
        return [max(0.0, 1.0 - abs(spot - j)) for j in range(count)]

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        if self.knot_weight is not None:
            for share, dw, db in zip(self.knot_mix(), self.knot_weight,
                                     self.knot_bias):
                if share:
                    weight = weight + share * dw
                    bias = bias + share * db
        c = make_divisible(
            self.num_features_max * self.width_mult / self.ratio) * self.ratio
        if isolated(self.width_mult):
            # the affine half moves with the channels; the running
            # statistics do not need to, because the narrowest width has
            # its own BN slot and nothing else reads it
            weight = weight[weight.size(0) - c:]
            bias = bias[bias.size(0) - c:]
        momentum = self.momentum
        if self.width_mult in FLAGS.width_mult_list:
            idx = FLAGS.width_mult_list.index(self.width_mult)
            if momentum is None:
                if self.training:
                    # cumulative moving average over the calibration
                    # batches, which is what cumulative_bn_stats asks for
                    self.calibration_batches[idx] += 1
                    momentum = 1.0 / float(self.calibration_batches[idx])
                else:
                    # unused in eval, but batch_norm still wants a float
                    momentum = 0.1
            # these slices are views, so batch_norm writes the updated
            # statistics straight back into self.bn[idx]
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean[:c],
                self.bn[idx].running_var[:c],
                weight[:c],
                bias[:c],
                self.training,
                momentum,
                self.eps)
        else:
            # an arbitrary width during training: no statistics are kept for
            # it, so this normalizes by the batch
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                weight[:c],
                bias[:c],
                self.training,
                momentum if momentum is not None else 0.1,
                self.eps)
        return y


def pop_channels(autoslim_channels):
    return [i.pop(0) for i in autoslim_channels]


def bn_calibration_init(m):
    """ calculating post-statistics of batch normalization """
    if isinstance(m, USBatchNorm2d):
        # This module holds its statistics in m.bn[idx] and hands those
        # buffers to batch_norm itself, using its own training flag. The
        # branch below cannot reach it: track_running_stats is False on the
        # outer module, so eval() leaves training False, batch_norm is told
        # not to update, and the statistics the branch below has just reset
        # are used as though they had been calibrated. Every width then
        # reads back at chance, which is what a 100-epoch run produced:
        # 75.4% at width 1.0 during training, 1.0% at every width after.
        m.training = True
        m.calibration_batches = [0] * len(FLAGS.width_mult_list)
        if getattr(FLAGS, 'cumulative_bn_stats', False):
            m.momentum = None
        return
    if getattr(m, 'track_running_stats', False):
        # reset all values for post-statistics
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = True
        # if use cumulative moving average
        if getattr(FLAGS, 'cumulative_bn_stats', False):
            m.momentum = None
