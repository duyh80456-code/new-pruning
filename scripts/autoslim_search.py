"""AutoSlim (Yu & Huang, arXiv 1903.11728) search on CIFAR-100 ResNet-50

Steps 1 and 2 of AutoSlim; step 3, training the searched configurations as
one slimmable network, is train.py with models/autoslim_resnet_cifar.py.

1. A layer-wise slimmable supernet. Unlike models/us_resnet.py, where one
   width multiplier slims every layer, each channel group here slims on its
   own: the stem, the two inner convolutions of every bottleneck, and one
   group per stage for the residual stream (the block outputs and the
   shortcut of a stage are added together, so they share a count). That is
   37 groups on ResNet-50. It is trained with the sandwich rule as US-Net's
   nonuniform variant: every group at its largest, every group at its
   smallest, and two samples where each group draws its own ratio, the
   last three distilled from the largest. The authors' code
   (JiahuiYu/slimmable_networks) releases the searched ResNet-50 channels
   but not this supernet, so it is written here; the recipe is ours, cut
   to supernet_epochs (AutoSlim trains the supernet for 10-20 % of the
   full schedule).
2. Greedy slimming, as slimming() in the authors' train.py: from the full
   network, try removing `divisor` channels from each group in turn,
   keep the removal that raises the error on held-out training images
   least, repeat. Each time the MACs fall to a target, the configuration
   is saved. The targets are the MACs of our sixteen uniform widths, so
   the searched networks are compared with every other row at equal cost.
   The held-out images are 50 a class from the training set, never the
   test set; the supernet does not train on them.

Batch norm in the supernet always normalises with the statistics of the
batch it sees, in training and in the search, so no running statistics
have to be kept for a continuum of configurations. The search scores a
configuration on the held-out set in chunks of `eval_chunk` images.

MACs are counted as utils/model_profiling.py counts them for our models:
convolutions, the classifier and the final average pool; the script checks
that its count of a uniform width equals models/us_resnet.py profiled.

Output: a JSON with the sixteen configurations (channel count per group,
widest first is not assumed: listed in the order of the target widths),
their MACs, and the search trace.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

MEAN = [0.5071, 0.4865, 0.4409]
STD = [0.2673, 0.2564, 0.2762]
WIDTHS = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
          0.85, 0.9, 0.95, 1.0]
STAGES = [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]  # c, n, s
SCORE_AMP = [True]  # set from --score_fp32 in main


def make_divisible(v, divisor=8, min_value=None):
    """as models/slimmable_ops.make_divisible"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# ---------------------------------------------------------------- structure

def group_names():
    """the 37 channel groups of ResNet-50, with their full sizes"""
    names = [('stem', 64)]
    for s, (c, _, _) in enumerate(STAGES):
        names.append(('out{}'.format(s + 1), c * 4))
    for s, (c, n, _) in enumerate(STAGES):
        for b in range(n):
            names.append(('s{}b{}m1'.format(s + 1, b + 1), c))
            names.append(('s{}b{}m2'.format(s + 1, b + 1), c))
    return names


def uniform_config(width):
    """the channel counts of models/us_resnet.py at one width"""
    return {name: make_divisible(full * width) for name, full in group_names()}


def macs(config, num_classes=100):
    """multiply-accumulates of one configuration on a 32x32 input, counted
    as utils/model_profiling.py counts them"""
    total = 3 * config['stem'] * 9 * 32 * 32
    inp, size = config['stem'], 32
    for s, (_, n, stride) in enumerate(STAGES):
        out = config['out{}'.format(s + 1)]
        for b in range(n):
            st = stride if b == 0 else 1
            m1 = config['s{}b{}m1'.format(s + 1, b + 1)]
            m2 = config['s{}b{}m2'.format(s + 1, b + 1)]
            total += inp * m1 * size * size                  # 1x1
            after = size // st
            total += m1 * m2 * 9 * after * after             # 3x3, stride
            total += m2 * out * after * after                # 1x1 up
            if b == 0:
                total += inp * out * after * after           # shortcut
            inp, size = out, after
    total += inp * size * size                               # average pool
    total += inp * num_classes                               # classifier
    return total


# ------------------------------------------------------------------- model

class GConv(nn.Conv2d):
    """a convolution whose output size is set by its group; the input size
    is whatever arrives"""

    def __init__(self, cin, cout, k, stride, group, net):
        super().__init__(cin, cout, k, stride, k // 2, bias=False)
        self.group = group
        self.net = [net]  # a list, so the model is not registered twice

    def forward(self, x):
        out = self.net[0].channels[self.group]
        w = self.weight[:out, :x.size(1)]
        return F.conv2d(x, w, None, self.stride, self.padding)


class GBN(nn.BatchNorm2d):
    """batch statistics of the batch at hand, the affine sliced"""

    def __init__(self, c):
        super().__init__(c, affine=True, track_running_stats=False)

    def forward(self, x):
        c = x.size(1)
        return F.batch_norm(x, None, None, self.weight[:c], self.bias[:c],
                            True, 0.0, self.eps)


class GBlock(nn.Module):
    def __init__(self, inp, mid, out, stride, prefix, out_group, net,
                 shortcut):
        super().__init__()
        self.body = nn.Sequential(
            GConv(inp, mid, 1, 1, prefix + 'm1', net), GBN(mid),
            nn.ReLU(inplace=True),
            GConv(mid, mid, 3, stride, prefix + 'm2', net), GBN(mid),
            nn.ReLU(inplace=True),
            GConv(mid, out, 1, 1, out_group, net), GBN(out),
        )
        self.shortcut = (nn.Sequential(
            GConv(inp, out, 1, stride, out_group, net), GBN(out))
            if shortcut else None)

    def forward(self, x):
        res = self.body(x)
        res = res + (x if self.shortcut is None else self.shortcut(x))
        return F.relu(res)


class Supernet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.full = dict(group_names())
        self.channels = dict(self.full)
        layers = [GConv(3, 64, 3, 1, 'stem', self), GBN(64),
                  nn.ReLU(inplace=True)]
        inp = 64
        for s, (c, n, stride) in enumerate(STAGES):
            for b in range(n):
                layers.append(GBlock(
                    inp, c, c * 4, stride if b == 0 else 1,
                    's{}b{}'.format(s + 1, b + 1), 'out{}'.format(s + 1),
                    self, shortcut=(b == 0)))
                inp = c * 4
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(inp, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        w = self.classifier.weight[:, :x.size(1)]
        return F.linear(x, w, self.classifier.bias)


# -------------------------------------------------------------------- data

def split(train_set, per_class, seed):
    """indices of the held-out images, per_class a class, and the rest"""
    rng = np.random.RandomState(seed)
    targets = np.array(train_set.targets)
    held = []
    for c in range(100):
        idx = np.where(targets == c)[0]
        held.extend(rng.choice(idx, per_class, replace=False).tolist())
    held_set = set(held)
    rest = [i for i in range(len(targets)) if i not in held_set]
    return sorted(held), rest


# ------------------------------------------------------------------ train

def soft_ce(student, teacher):
    return -(F.softmax(teacher, 1) * F.log_softmax(student, 1)).sum(1).mean()


def train_supernet(net, loader, args, device):
    decay, no_decay = [], []
    for p in net.parameters():
        (decay if p.dim() in (2, 4) else no_decay).append(p)
    opt = torch.optim.SGD([{'params': decay, 'weight_decay': args.wd},
                           {'params': no_decay, 'weight_decay': 0}],
                          lr=args.lr, momentum=0.9, nesterov=True)
    steps = args.supernet_epochs * len(loader)
    warm = args.warmup_epochs * len(loader)
    names = list(net.full)
    low = args.low
    step = 0
    for epoch in range(args.supernet_epochs):
        net.train()
        t0, seen, correct, loss_sum = time.time(), 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device)
            if step < warm:
                lr = args.lr * (step + 1) / warm
            else:
                lr = 0.5 * args.lr * (1 + math.cos(
                    math.pi * (step - warm) / max(1, steps - warm)))
            for g in opt.param_groups:
                g['lr'] = lr
            opt.zero_grad(set_to_none=True)
            # the sandwich rule, US-Net's nonuniform variant
            ratios = [{n: 1.0 for n in names}, {n: low for n in names}]
            for _ in range(2):
                ratios.append({n: random.uniform(low, 1.0) for n in names})
            teacher = None
            for i, r in enumerate(ratios):
                net.channels = {n: make_divisible(net.full[n] * r[n])
                                for n in names}
                out = net(x)
                if i == 0:
                    loss = F.cross_entropy(out, y)
                    teacher = out.detach()
                    correct += (out.argmax(1) == y).sum().item()
                    loss_sum += loss.item() * x.size(0)
                    seen += x.size(0)
                else:
                    loss = soft_ce(out, teacher)
                loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            opt.step()
            step += 1
        print('supernet epoch {}/{}: loss {:.4f}, top1 {:.2f} (largest), '
              'lr {:.4f}, {:.0f}s'.format(
                  epoch + 1, args.supernet_epochs, loss_sum / seen,
                  100.0 * correct / seen, lr, time.time() - t0), flush=True)
    net.channels = dict(net.full)


# ----------------------------------------------------------------- search

@torch.no_grad()
def score(net, chunks):
    """error and loss on the held-out images

    Under autocast in half precision: the search only ranks candidate
    removals, and the logits are compared and fed to the loss in float32.
    """
    wrong, loss, n = 0, 0.0, 0
    for x, y in chunks:
        with torch.autocast('cuda', dtype=torch.float16,
                            enabled=SCORE_AMP[0]):
            out = net(x).float()
        wrong += (out.argmax(1) != y).sum().item()
        loss += F.cross_entropy(out, y, reduction='sum').item()
        n += y.numel()
    return wrong / n, loss / n


@torch.no_grad()
def search(net, chunks, args):
    net.eval()  # GBN ignores the mode; nothing else in the net has one
    names = list(net.full)
    floor = {n: max(args.divisor, make_divisible(net.full[n] * args.low))
             for n in names}
    targets = sorted(((macs(uniform_config(w)), w) for w in WIDTHS),
                     reverse=True)
    config = dict(net.full)
    found, trace = {}, []
    t0 = time.time()
    while targets:
        now = macs(config)
        while targets and now <= targets[0][0]:
            target, w = targets.pop(0)
            net.channels = dict(config)
            err, ls = score(net, chunks)
            found[w] = {'channels': dict(config), 'macs': now,
                        'target_macs': target, 'heldout_error': err,
                        'heldout_loss': ls}
            print('width {:.2f}: {} MACs (target {}), held-out error '
                  '{:.4f}'.format(w, now, target, err), flush=True)
        if not targets:
            break
        best = None
        for n in names:
            if config[n] - args.divisor < floor[n]:
                continue
            config[n] -= args.divisor
            net.channels = config
            err, ls = score(net, chunks)
            config[n] += args.divisor
            key = (err, ls)
            if best is None or key < best[0]:
                best = (key, n)
        if best is None:
            raise RuntimeError('every group is at its floor before the '
                               'last target: lower --low')
        (err, ls), n = best
        config[best[1]] -= args.divisor
        trace.append({'group': n, 'to': config[n], 'macs': macs(config),
                      'error': err, 'loss': ls})
        if len(trace) % 25 == 0:
            print('step {}: {} -> {}, {:.1f}M MACs, held-out error {:.4f}, '
                  '{:.0f}s'.format(len(trace), n, config[n],
                                   macs(config) / 1e6, err,
                                   time.time() - t0), flush=True)
    net.channels = dict(net.full)
    return found, trace


# ------------------------------------------------------------------- check

def check_macs():
    """our count of a uniform width against models/us_resnet.py profiled"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from utils.config import FLAGS
    FLAGS.depth = 50
    FLAGS.width_mult_range = [0.25, 1.0]
    FLAGS.width_mult_list = [0.25, 1.0]
    FLAGS.reset_parameters = True
    FLAGS.dataset = 'cifar100'
    import importlib
    from utils.model_profiling import model_profiling
    model = importlib.import_module('models.us_resnet').Model(100, 32)
    bad = []
    for w in WIDTHS:
        model.apply(lambda m: setattr(m, 'width_mult', w))
        profiled, _ = model_profiling(model, 32, 32, use_cuda=False,
                                      verbose=False)
        ours = macs(uniform_config(w))
        if profiled != ours:
            bad.append((w, profiled, ours))
    return bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--out', default='results/autoslim_search_r50.json')
    p.add_argument('--seed', type=int, default=1995)
    p.add_argument('--heldout_per_class', type=int, default=50)
    p.add_argument('--supernet_epochs', type=int, default=20)
    p.add_argument('--warmup_epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.2)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--clip', type=float, default=5.0)
    p.add_argument('--low', type=float, default=0.125,
                   help='smallest ratio of a group in training and search')
    p.add_argument('--divisor', type=int, default=8)
    p.add_argument('--eval_chunk', type=int, default=1000)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--supernet_ckpt', default='')
    p.add_argument('--limit_batches', type=int, default=0,
                   help='smoke test: stop each epoch after this many')
    p.add_argument('--check_macs', action='store_true')
    p.add_argument('--score_fp32', action='store_true',
                   help='score the search candidates in float32')
    args = p.parse_args()

    if args.check_macs:
        bad = check_macs()
        print('MACs check:', 'OK' if not bad else bad)
        return

    SCORE_AMP[0] = not args.score_fp32
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    # the channel counts change at every step of the supernet and of the
    # search, so benchmark mode would re-tune cuDNN for every new shape
    torch.backends.cudnn.benchmark = False

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    plain_tf = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(MEAN, STD)])
    download = not os.path.isdir(os.path.join(args.data, 'cifar-100-python'))
    train_aug = datasets.CIFAR100(args.data, True, train_tf, download=download)
    train_plain = datasets.CIFAR100(args.data, True, plain_tf)
    held, rest = split(train_aug, args.heldout_per_class, args.seed)

    net = Supernet().to(device)
    if args.supernet_ckpt and os.path.exists(args.supernet_ckpt):
        net.load_state_dict(torch.load(args.supernet_ckpt,
                                       map_location=device))
        print('supernet loaded from', args.supernet_ckpt)
    else:
        sub = torch.utils.data.Subset(train_aug, rest)
        if args.limit_batches:
            sub = torch.utils.data.Subset(
                sub, list(range(args.limit_batches * args.batch_size)))
        loader = torch.utils.data.DataLoader(
            sub, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.workers, pin_memory=True)
        train_supernet(net, loader, args, device)
        if args.supernet_ckpt:
            os.makedirs(os.path.dirname(args.supernet_ckpt) or '.',
                        exist_ok=True)
            torch.save(net.state_dict(), args.supernet_ckpt)

    # the held-out images, fixed order, without augmentation, on the GPU
    xs = torch.stack([train_plain[i][0] for i in held]).to(device)
    ys = torch.tensor([train_plain.targets[i] for i in held]).to(device)
    chunks = [(xs[i:i + args.eval_chunk], ys[i:i + args.eval_chunk])
              for i in range(0, len(held), args.eval_chunk)]
    full_err, _ = score(net, chunks)
    print('held-out error of the full supernet: {:.4f}'.format(full_err))

    found, trace = search(net, chunks, args)
    result = {
        'what': 'AutoSlim search on CIFAR-100 ResNet-50 '
                '(scripts/autoslim_search.py)',
        'args': vars(args),
        'groups': group_names(),
        'widths': WIDTHS,
        'configs': [found[w] for w in WIDTHS],
        'trace': trace,
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
