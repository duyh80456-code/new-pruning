"""Is the prefix already the important half?

    python scripts/channel_order.py logs/cifar100_k_feature_pair/best_model.pt

US-Net slices channels as weight[:k], so which channels a narrow width
gets is decided by initialisation and never revisited. The obvious fix
is to sort channels by importance - L1 of the filter, or the batch norm
scale - so the prefix holds the strongest ones.

Before spending a session on that, ask whether there is anything left
to sort. The sandwich rule trains the prefix at every sampled width and
the tail only at the widest, and at width 0.25 the prefix has to
classify on its own. So the objective already pushes importance toward
the front. If it has done that, sorting is a no-op.

This prints, per layer, how much of the total importance the prefix
already holds at each width, and how much it would hold if the channels
were sorted. The gap between those two columns is the headroom. No
headroom, no branch.
"""
import os
import sys

import torch


def importance(state, prefix):
    """per-output-channel weight of one conv, by L1 and by BN scale"""
    w = state.get(prefix + '.weight')
    if w is None or w.dim() != 4:
        return None
    return w.abs().sum(dim=(1, 2, 3))


def report(scores, widths):
    """mass in the prefix as it stands, and as it would be if sorted"""
    total = scores.sum().clamp_min(1e-12)
    out = []
    for width in widths:
        k = max(1, int(round(len(scores) * width)))
        now = scores[:k].sum() / total
        best = scores.sort(descending=True).values[:k].sum() / total
        out.append((width, float(now), float(best)))
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    if not os.path.exists(path):
        print('no checkpoint at {}'.format(path))
        return 1
    blob = torch.load(path, map_location='cpu', weights_only=False)
    state = blob.get('model', blob)
    state = {k.replace('module.', '', 1): v for k, v in state.items()}

    convs = sorted({k[:-len('.weight')] for k, v in state.items()
                    if k.endswith('.weight') and v.dim() == 4})
    if not convs:
        print('no conv weights in {}'.format(path))
        return 1

    widths = [0.25, 0.50, 0.75]
    print('{:44} {:>6} {:>8} {:>8} {:>8}'.format(
        'layer', 'width', 'now', 'sorted', 'gap'))
    totals = {w: [0.0, 0.0] for w in widths}
    for name in convs:
        scores = importance(state, name)
        if scores is None or len(scores) < 8:
            continue
        for width, now, best in report(scores, widths):
            totals[width][0] += now
            totals[width][1] += best
            if width == 0.25:
                print('{:44} {:6.2f} {:8.3f} {:8.3f} {:8.3f}'.format(
                    name[-44:], width, now, best, best - now))
    n = len([c for c in convs
             if importance(state, c) is not None
             and len(importance(state, c)) >= 8])
    print()
    print('averaged over {} conv layers:'.format(n))
    print('{:>8} {:>10} {:>10} {:>10}'.format(
        'width', 'now', 'sorted', 'headroom'))
    for width in widths:
        now, best = totals[width][0] / n, totals[width][1] / n
        print('{:8.2f} {:10.3f} {:10.3f} {:10.3f}'.format(
            width, now, best, best - now))
    print()
    print('"now" is the share of L1 the first k channels hold today.')
    print('"sorted" is what the best k would hold. A width whose two')
    print('columns already agree has nothing for sorting to move.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
