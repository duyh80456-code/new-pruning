"""the permutation has to be a symmetry of the full network

A consistent reordering of a channel space changes nothing the network
computes at full width: the same values arrive at the same places by a
different route. At a narrow width it changes everything that matters,
because weight[:k] now selects a different set of channels.

Those two facts together are the only way to know the groups are right.
A permutation that misses a tensor still produces a network that runs
and trains, and the damage shows up as a number in a table weeks later.
"""
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not any(arg.startswith('app:') for arg in sys.argv):
    sys.argv.append('app:apps/cifar100_k_feature_pair.yml')

import importlib

import torch

from utils.config import FLAGS
from utils.channel_reorder import collect, reorder, score_l1, score_read

FAILURES = []


def check(name, condition, detail=''):
    print('{}  {}{}'.format('pass' if condition else 'FAIL', name,
                            '  ' + detail if detail else ''))
    if not condition:
        FAILURES.append(name)


def build():
    model = importlib.import_module(FLAGS.model).Model(
        FLAGS.num_classes, input_size=FLAGS.image_size)
    model.apply(lambda m: setattr(m, 'width_mult', 1.0))
    model.eval()
    return model


def at_width(model, x, width):
    model.apply(lambda m: setattr(m, 'width_mult', width))
    with torch.no_grad():
        return model(x)


def test_the_groups_cover_every_channel_space():
    model = build()
    groups = collect(model)
    shared = [g for g in groups if g['shared']]
    private = [g for g in groups if not g['shared']]
    check('one shared space per stage',
          [g['size'] for g in shared] == [64, 128, 256, 512],
          str([g['size'] for g in shared]))
    check('one private space per block', len(private) == 8,
          '{} spaces'.format(len(private)))
    check('the last shared space is read by the classifier',
          any(w.dim() == 2 for w in shared[-1]['consume']))
    check('the first shared space is written by the stem',
          len(shared[0]['produce']) > len(shared[1]['produce']) - 1,
          '{} producers against {}'.format(len(shared[0]['produce']),
                                           len(shared[1]['produce'])))


def test_a_permutation_leaves_the_full_width_output_alone():
    """the check that says the groups are complete"""
    torch.manual_seed(1995)
    model = build()
    x = torch.randn(4, 3, 32, 32)
    before = at_width(model, x, 1.0)

    groups, moved, held = reorder(model, 'l1')
    check('something actually moved', moved > 0,
          '{} channels across {} spaces, prefix already held {:.0%}'
          .format(moved, groups, held))

    after = at_width(model, x, 1.0)
    gap = float((before - after).abs().max())
    check('and the full width output did not', gap < 1e-4,
          'largest change {:.2e}'.format(gap))


def test_it_changes_the_narrow_subnet():
    """and the check that says it does anything at all"""
    torch.manual_seed(1995)
    model = build()
    x = torch.randn(4, 3, 32, 32)
    before = at_width(model, x, 0.25)
    reorder(model, 'l1')
    after = at_width(model, x, 0.25)
    gap = float((before - after).abs().max())
    check('the narrow subnet is a different network', gap > 1e-3,
          'largest change {:.2e}'.format(gap))


def test_the_prefix_holds_the_best_channels_afterwards():
    torch.manual_seed(1995)
    model = build()
    for criterion, scorer in (('l1', score_l1), ('read', score_read)):
        fresh = build()
        reorder(fresh, criterion)
        worst = 1.0
        for group in collect(fresh):
            score = scorer(group)
            k = max(1, score.numel() // 4)
            top = set(score.argsort(descending=True)[:k].tolist())
            worst = min(worst, len(top & set(range(k))) / k)
        check('{}: every prefix is the top quarter after'.format(criterion),
              worst > 0.999, 'worst overlap {:.0%}'.format(worst))


def test_the_optimizer_state_moves_with_the_weights():
    """a momentum buffer left behind feeds each channel the wrong velocity

    Nothing would crash. Every channel would simply be pushed in the
    direction that belonged to whoever used to sit at its index, and the
    run would finish and report a number.
    """
    torch.manual_seed(1995)
    model = build()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    x = torch.randn(4, 3, 32, 32)
    model.apply(lambda m: setattr(m, 'width_mult', 1.0))
    model.train()
    model(x).sum().backward()
    opt.step()          # fills the momentum buffers

    conv = model.features[1].body[0]
    before = opt.state[conv.weight]['momentum_buffer'].clone()
    groups = collect(model)
    private = [g for g in groups
               if not g['shared'] and g['produce'][0][0] is conv.weight][0]
    order = score_l1(private).argsort(descending=True)
    from utils.channel_reorder import apply as move
    move(private, order, opt)
    after = opt.state[conv.weight]['momentum_buffer']

    check('the buffer is still there', after is not None)
    check('and it was permuted the same way the weight was',
          torch.allclose(after, before[order]),
          'max gap {:.2e}'.format(float((after - before[order]).abs().max())))
    check('which is not the same as leaving it alone',
          not torch.allclose(after, before))


def test_it_reaches_through_the_wrapper_train_py_hands_it():
    """train.py passes model_wrapper, which is a DataParallel

    Every other check here runs on the bare model. If unwrapping were
    wrong this would raise, or worse, permute nothing and report that it
    had.
    """
    torch.manual_seed(1995)
    model = build()
    wrapper = torch.nn.DataParallel(model)
    x = torch.randn(4, 3, 32, 32)
    before = at_width(wrapper, x, 1.0)
    spaces, moved, held = reorder(wrapper, 'read')
    after = at_width(wrapper, x, 1.0)
    check('the wrapper is unwrapped and something moved', moved > 0,
          '{} channels, prefix already held {:.0%}'.format(moved, held))
    check('and full width is still unchanged through it',
          float((before - after).abs().max()) < 1e-4)


def main():
    print('torch', torch.__version__)
    print()
    test_the_groups_cover_every_channel_space()
    test_a_permutation_leaves_the_full_width_output_alone()
    test_it_changes_the_narrow_subnet()
    test_the_prefix_holds_the_best_channels_afterwards()
    test_the_optimizer_state_moves_with_the_weights()
    test_it_reaches_through_the_wrapper_train_py_hands_it()
    print()
    if FAILURES:
        print('{} failed: {}'.format(len(FAILURES), ', '.join(FAILURES)))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
