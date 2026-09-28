"""affine_knots: every BN scale and shift a continuous function of width.

Built to fail if the flag is silently ignored or wired wrong: the knot
shares have to be a partition of one that moves continuously with the
width, a width's backward has to reach the two knots around it and no
other, an offset has to change the widths near its knot and leave the
far ones alone, zero offsets have to reproduce the plain model exactly,
the offsets have to stay out of weight decay the way gamma and beta do,
and a channel permutation has to carry them along. affine_knots 0 has to
build the model it always did, key for key.

Runs on the CPU in a few seconds: python tests/test_affine_knots.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = [sys.argv[0], 'app:apps/smoke_a_kl.yml']

import torch  # noqa: E402

from utils.config import FLAGS  # noqa: E402

WIDTHS = [round(0.25 + 0.05 * k, 2) for k in range(16)]


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print('ok   ', what)


def build(knots, seed=0):
    FLAGS.affine_knots = knots
    FLAGS.reset_parameters = True
    FLAGS.return_features = False
    torch.manual_seed(seed)
    model = importlib.import_module('models.us_resnet').Model(100, 32)
    model.train()
    return model


def set_width(model, width):
    model.apply(lambda m: setattr(m, 'width_mult', width))


def norms(model):
    from models.slimmable_ops import USBatchNorm2d
    return [m for m in model.modules() if isinstance(m, USBatchNorm2d)]


def knots_of(model):
    return [p for bn in norms(model)
            for p in list(bn.knot_weight) + list(bn.knot_bias)]


if __name__ == '__main__':
    x = torch.randn(4, 3, 32, 32)

    plain = build(0)
    check(not any('knot' in k for k in plain.state_dict()),
          'affine_knots 0 adds nothing to the state dict')
    model = build(4)
    check(set(plain.state_dict()) ==
          {k for k in model.state_dict() if 'knot' not in k},
          'affine_knots 4 only adds keys, it renames none')

    bn = norms(model)[0]
    rows = []
    for width in WIDTHS:
        bn.width_mult = width
        rows.append(bn.knot_mix())
    check(all(abs(sum(r) - 1.0) < 1e-9 and sum(v > 0 for v in r) <= 2
              for r in rows),
          'at every test width the knot shares sum to one, two at most')
    check(rows[0] == [1.0, 0.0, 0.0, 0.0] and rows[-1] == [0.0, 0.0, 0.0,
                                                           1.0],
          '0.25 sits on the first knot and 1.00 on the last')
    bn.width_mult = 0.5
    check(abs(bn.knot_mix()[1] - 1.0) < 1e-9, '0.50 sits on the second knot')
    gaps = []
    for k in range(2000):
        a = 0.25 + 0.75 * k / 2000
        bn.width_mult = a
        left = bn.knot_mix()
        bn.width_mult = a + 1e-4
        right = bn.knot_mix()
        gaps.append(max(abs(p - q) for p, q in zip(left, right)))
    check(max(gaps) < 1e-3,
          'the shares move continuously with the width ({:.1e} per 1e-4)'
          .format(max(gaps)))

    for width in (0.3, 0.7, 1.0):
        set_width(plain, width)
        set_width(model, width)
        same = (plain(x) - model(x)).abs().max().item()
        check(same == 0.0,
              'with zero offsets width {} is the plain model exactly'.format(
                  width))

    for width, reached in ((0.3, {0, 1}), (0.7, {1, 2}), (0.8, {2, 3}),
                           (1.0, {3})):
        model.zero_grad(set_to_none=True)
        set_width(model, width)
        model(x).sum().backward()
        got = {j for j, p in enumerate(norms(model)[0].knot_weight)
               if p.grad is not None and p.grad.abs().sum() > 0}
        check(got == reached,
              'a backward at {} reaches knots {} and no other'.format(
                  width, sorted(reached)))

    with torch.no_grad():
        for bn in norms(model):
            bn.knot_weight[0].add_(0.5)
    moved = {}
    for width in (0.25, 0.3, 0.5, 1.0):
        set_width(plain, width)
        set_width(model, width)
        moved[width] = (plain(x) - model(x)).abs().max().item()
    check(moved[0.25] > 1e-3 and moved[0.3] > 1e-3,
          'an offset at the first knot changes 0.25 and 0.30')
    check(moved[0.5] == 0.0 and moved[1.0] == 0.0,
          'and leaves 0.50 and 1.00, past its reach, exactly alone')

    import train as T
    groups = T.get_optimizer(model).param_groups
    decay = {id(g['params'][0]): g['weight_decay'] for g in groups}
    check(all(decay[id(p)] == 0 for p in knots_of(model)),
          'the knot offsets are out of weight decay, like gamma and beta')

    from utils.channel_reorder import collect, reorder  # noqa: F401
    from utils.channel_reorder import apply as permute
    torch.manual_seed(1)
    with torch.no_grad():
        for p in knots_of(model):
            p.normal_(0, 0.2)
    model.train()  # batch statistics, which the permutation cannot reset
    set_width(model, 1.0)
    with torch.no_grad():
        before = model(x)
    for group in collect(model):
        permute(group, torch.randperm(group['size']))
    with torch.no_grad():
        after = model(x)
    check((before - after).abs().max().item() < 1e-4,
          'a channel permutation carries the offsets with it')
    print('all checks passed')
