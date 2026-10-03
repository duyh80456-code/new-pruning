"""lora_knots: a low-rank update to every conv kernel, continuous in width.

Built to fail if the flag is silently ignored or wired wrong: lora_knots 0
has to build the model it always did, key for key; with the up matrices
at zero every width has to be the plain model to float precision; a width's backward
has to reach the two knots around it and no other; an update at one knot
has to change the widths near it and leave the far ones alone; the
update has to merge into one plain kernel per width with the same output,
which is what makes it free at inference; and a channel reorder, which
does not know how to move the updates, has to refuse.

Runs on the CPU in a few seconds: python tests/test_lora.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = [sys.argv[0], 'app:apps/smoke_a_kl.yml']

import torch  # noqa: E402

from utils.config import FLAGS  # noqa: E402


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print('ok   ', what)


def build(knots, rank=4, seed=0):
    FLAGS.lora_knots = knots
    FLAGS.lora_rank = rank
    FLAGS.affine_knots = 0
    FLAGS.reset_parameters = True
    FLAGS.return_features = False
    torch.manual_seed(seed)
    model = importlib.import_module('models.us_resnet').Model(100, 32)
    model.train()
    return model


def set_width(model, width):
    model.apply(lambda m: setattr(m, 'width_mult', width))


def convs(model):
    from models.slimmable_ops import USConv2d
    return [m for m in model.modules() if isinstance(m, USConv2d)]


if __name__ == '__main__':
    x = torch.randn(4, 3, 32, 32)

    plain = build(0)
    check(not any('lora' in k for k in plain.state_dict()),
          'lora_knots 0 adds nothing to the state dict')
    model = build(4)
    check(set(plain.state_dict()) ==
          {k for k in model.state_dict() if 'lora' not in k},
          'lora_knots 4 only adds keys, it renames none')
    for key, value in plain.state_dict().items():
        model.state_dict()[key].copy_(value)
    extra = sum(p.numel() for n, p in model.named_parameters()
                if 'lora' in n)
    base = sum(p.numel() for p in plain.parameters())
    print('      rank 4, 4 knots: {:.2f}M extra on {:.2f}M ({:.1f}%)'.format(
        extra / 1e6, base / 1e6, 100.0 * extra / base))

    for width in (0.3, 0.7, 1.0):
        set_width(plain, width)
        set_width(model, width)
        gap = (plain(x) - model(x)).abs().max().item()
        # adding the zero update makes the kernel contiguous, which can
        # send the convolution down another kernel: float noise, not more
        check(gap < 1e-5,
              'with the up matrices at zero width {} is the plain model '
              '({:.0e})'.format(width, gap))

    for width, reached in ((0.3, {0, 1}), (0.7, {1, 2}), (0.8, {2, 3}),
                           (1.0, {3})):
        model.zero_grad(set_to_none=True)
        set_width(model, width)
        model(x).sum().backward()
        got = {j for j, p in enumerate(convs(model)[1].lora_up)
               if p.grad is not None and p.grad.abs().sum() > 0}
        check(got == reached,
              'a backward at {} reaches knots {} and no other'.format(
                  width, sorted(reached)))

    torch.manual_seed(1)
    with torch.no_grad():
        for conv in convs(model):
            conv.lora_up[0].normal_(0, 0.05)
    moved = {}
    for width in (0.25, 0.3, 0.5, 1.0):
        set_width(plain, width)
        set_width(model, width)
        moved[width] = (plain(x) - model(x)).abs().max().item()
    check(moved[0.25] > 1e-3 and moved[0.3] > 1e-3,
          'an update at the first knot changes 0.25 and 0.30')
    check(moved[0.5] < 1e-5 and moved[1.0] < 1e-5,
          'and leaves 0.50 and 1.00, past its reach, alone ({:.0e}, '
          '{:.0e})'.format(moved[0.5], moved[1.0]))

    # merged: one plain kernel per width, the same output
    merged = build(0)
    merged.load_state_dict(
        {k: v for k, v in model.state_dict().items() if 'lora' not in k})
    with torch.no_grad():
        for src, dst in zip(convs(model), convs(merged)):
            src.width_mult = 0.3
            delta = 0
            for share, up, down in zip(src.lora_mix(), src.lora_up,
                                       src.lora_down):
                delta = delta + share * torch.einsum('or,rikl->oikl', up,
                                                     down)
            dst.weight.add_(delta)
    # batch statistics on both, so the comparison needs no calibration
    set_width(model, 0.3)
    set_width(merged, 0.3)
    gap = (model(x) - merged(x)).abs().max().item()
    check(gap < 1e-4,
          'width 0.30 merged into plain kernels gives the same output '
          '({:.1e})'.format(gap))

    import train as T
    groups = T.get_optimizer(model).param_groups
    held = {id(g['params'][0]) for g in groups}
    check(all(id(p) in held for n, p in model.named_parameters()
              if 'lora' in n),
          'every lora parameter is in the optimizer')

    from utils.channel_reorder import reorder
    try:
        reorder(model)
        refused = False
    except NotImplementedError:
        refused = True
    check(refused, 'a channel reorder refuses rather than misplace updates')

    FLAGS.lora_knots = 0
    print('all checks passed')
