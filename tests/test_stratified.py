"""width_sampling: stratified, both pairings.

Each check fails if the flag is ignored: plain uniform draws neither cover
every slice exactly once per block, nor keep a step's two free widths one
slice apart (adjacent) or in opposite halves (spread).

Runs on the CPU in a second: python tests/test_stratified.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = [sys.argv[0], 'app:apps/smoke_a_kl.yml']

from utils.config import FLAGS  # noqa: E402
from utils import loss_ops  # noqa: E402


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print('ok   ', what)


def blocks(pairing, n_blocks=50, block=10, draw='slices'):
    FLAGS.width_sampling = 'stratified'
    FLAGS.stratified_pairing = pairing
    FLAGS.stratified_draw = draw
    FLAGS.stratified_block = block
    FLAGS.num_sample_training = 4
    del loss_ops._STRATA[:]
    random.seed(0)
    out = []
    for _ in range(n_blocks):
        steps = [loss_ops.training_widths(50, 0.25, 1.0)
                 for _ in range(block)]
        if any(s[:2] != [1.0, 0.25] or len(s) != 4 for s in steps):
            raise AssertionError('not a sandwich: {}'.format(steps[0]))
        out.append([s[2:] for s in steps])
    return out


def slice_of(width, total=20):
    return min(int((width - 0.25) / 0.75 * total), total - 1)


def main():
    for pairing in ('adjacent', 'spread'):
        every = blocks(pairing)
        check(all(sorted(slice_of(w) for step in b for w in step)
                  == list(range(20)) for b in every),
              '{}: each block hits each of the 20 slices once'.format(
                  pairing))

    adjacent = blocks('adjacent')
    check(all(slice_of(s[1]) == slice_of(s[0]) + 1 for b in adjacent
              for s in b),
          'adjacent: the two free widths of a step are neighbouring slices')
    check(all([s[0] for s in b] == sorted(s[0] for s in b)
               for b in adjacent),
          'adjacent: a block runs from the narrow end to the wide one')

    spread = blocks('spread')
    check(all(min(s) < 0.625 <= max(s) for b in spread for s in b),
          'spread: one free width in each half, every step')
    check(any([min(s) for s in b] != sorted(min(s) for s in b)
              for b in spread),
          'spread: the steps of a block are shuffled')
    gaps = [abs(s[0] - s[1]) for b in spread for s in b]
    check(sum(gaps) / len(gaps) > 0.3,
          'spread: mean gap {:.3f}, about half the range'.format(
              sum(gaps) / len(gaps)))

    # plain uniform draws, sorted and dealt out the same two ways
    adjacent = blocks('adjacent', draw='uniform')
    check(any(sorted(slice_of(w) for step in b for w in step)
              != list(range(20)) for b in adjacent),
          'uniform draw: some block leaves a slice empty, so no slices')
    check(all([w for s in b for w in s] == sorted(w for s in b for w in s)
              for b in adjacent),
          'uniform draw, adjacent: steps take the sorted draws in order')
    spread = blocks('spread', draw='uniform')
    check(all(max(min(s) for s in b) <= min(max(s) for s in b)
              for b in spread),
          'uniform draw, spread: each step one of the lower ten and one '
          'of the upper ten')
    check(any([min(s) for s in b] != sorted(min(s) for s in b)
              for b in spread),
          'uniform draw, spread: the steps are shuffled')

    # slices dealt out at random: the coverage stays, the gap is left to
    # chance as in US-Net
    shuffled = blocks('random', n_blocks=500)
    check(all(sorted(slice_of(w) for step in b for w in step)
              == list(range(20)) for b in shuffled),
          'random: each block still hits each of the 20 slices once')
    gaps = [abs(s[0] - s[1]) for b in shuffled for s in b]
    close = sum(g < 0.1 for g in gaps) / len(gaps)
    check(0.2 < sum(gaps) / len(gaps) < 0.3,
          'random: mean gap {:.3f}, near US-Net\'s 0.25'.format(
              sum(gaps) / len(gaps)))
    check(close > 0.1,
          'random: {:.0%} of pairs closer than 0.1, so not spread'.format(
              close))
    check(any(slice_of(s[1]) != slice_of(s[0]) + 1 for b in shuffled
              for s in b),
          'random: not adjacent')

    FLAGS.width_sampling = 'uniform'
    FLAGS.stratified_draw = 'slices'
    print('all checks passed')


if __name__ == '__main__':
    main()
