"""Is the prefix already the important half, by every criterion at hand?

    python scripts/channel_order.py k_best_model.zip
    BATCHES=16 python scripts/channel_order.py path/to/best_model.pt

US-Net slices channels as weight[:k], so which channels a narrow width
gets is decided by initialisation and never revisited. Sorting them by
importance is the obvious thing to try. Whether it would move anything
depends on a measurement, and the measurement depends on what you call
importance, so this runs every criterion it can and prints them side by
side rather than picking one.

For each it reports the share of importance the first quarter of the
channels holds, against a shuffled control with identical values and
the best any quarter could hold. The gap between the last two is the
headroom sorting could recover.

The criteria that need a backward pass are skipped, with a line saying
so, when no readable CIFAR-100 turns up.
"""
import os
import pickle
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_ARGS = [a for a in sys.argv[1:] if not a.startswith('app:')]
sys.argv = [sys.argv[0], 'app:apps/cifar100_k_feature_pair.yml']

WIDTH = 0.25
SHUFFLES = 8


# ----------------------------------------------------------------- data

def find_train():
    """the pickle, wherever it is

    torchvision re-checks the tarball before opening files it already
    extracted, so a mounted copy without it reads as missing. A sibling
    checkout usually has a complete one.
    """
    tried = []
    if os.environ.get('CIFAR_TRAIN'):
        tried.append(os.environ['CIFAR_TRAIN'])
    tried.append(os.path.join(ROOT, 'data', 'cifar-100-python', 'train'))
    parent = os.path.dirname(ROOT)
    if os.path.isdir(parent):
        for entry in sorted(os.listdir(parent)):
            tried.append(os.path.join(parent, entry, 'data',
                                      'cifar-100-python', 'train'))
    if os.path.isdir('/kaggle/input'):
        for base, dirs, _ in os.walk('/kaggle/input'):
            if 'cifar-100-python' in dirs:
                tried.append(os.path.join(base, 'cifar-100-python', 'train'))
    for here in tried:
        if not os.path.exists(here):
            continue
        try:
            with open(here, 'rb') as handle:
                return pickle.load(handle, encoding='bytes'), here
        except Exception as problem:
            print('  unusable: {} ({})'.format(here, problem))
    return None, None


def batches_of(raw, batch, count):
    images = torch.from_numpy(
        raw[b'data'].reshape(-1, 3, 32, 32)).float().div_(255.0)
    mean = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
    std = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)
    images = (images - mean) / std
    labels = torch.tensor(raw[b'fine_labels'])
    order = torch.randperm(len(labels))[:batch * count]
    return [(images[order[i:i + batch]], labels[order[i:i + batch]])
            for i in range(0, len(order), batch)]


# ------------------------------------------------------ data-free scores

def magnitude(w, p):
    return w.abs().pow(p).sum(dim=(1, 2, 3)).pow(1.0 / p)


def redundancy(w):
    """FPGM: how far a filter sits from the rest of them

    A different question from the others. Magnitude and Taylor ask
    which channels matter; this asks which are duplicates. The sandwich
    rule presses directly for a prefix that is good and only indirectly
    for one that is varied, so this is the criterion with the most room
    to disagree.
    """
    flat = w.reshape(w.size(0), -1)
    return torch.cdist(flat, flat).sum(dim=1)


def downstream(w_next, out_channels):
    """how much the next layer actually reads each of these channels

    The gain says how far a channel is turned up and nothing about
    whether anything downstream uses it. This says the second half.
    """
    if w_next is None or w_next.size(1) != out_channels:
        return None
    return w_next.abs().sum(dim=(0, 2, 3))


# ---------------------------------------------------------------- report

def share(scores):
    total = scores.sum().clamp_min(1e-12)
    k = max(1, int(round(len(scores) * WIDTH)))
    now = float(scores[:k].sum() / total)
    best = float(scores.sort(descending=True).values[:k].sum() / total)
    rand = sum(float(scores[torch.randperm(len(scores))][:k].sum() / total)
               for _ in range(SHUFFLES)) / SHUFFLES
    return rand, now, best


def summarise(name, per_layer):
    rand = now = best = 0.0
    for v in per_layer:
        a, b, c = share(v)
        rand += a
        now += b
        best += c
    n = len(per_layer)
    rand, now, best = rand / n, now / n, best / n
    got = 100.0 * (now - rand) / max(best - rand, 1e-9)
    print('{:<28} {:>9.3f} {:>9.3f} {:>9.3f} {:>10.0f}%'.format(
        name, rand, now, best, got))


def agreement(a, b):
    cors = []
    for x, y in zip(a, b):
        k = min(len(x), len(y))
        u = x[:k].argsort(descending=True).argsort().float()
        v = y[:k].argsort(descending=True).argsort().float()
        u = u - u.mean()
        v = v - v.mean()
        cors.append(float((u * v).sum() / (u.norm() * v.norm() + 1e-12)))
    return sum(cors) / len(cors), min(cors)


def loss_based(state, raw):
    """Taylor and Fisher on the batch-norm gains, at two widths

    Zeroing a gain removes the channel, so the first-order estimate of
    what that costs the loss is the quantity the sorting idea is about.
    Two widths, because the nesting admits one global permutation and
    sixteen widths have to live with it.
    """
    import importlib
    from utils.config import FLAGS
    model = importlib.import_module(FLAGS.model).Model(
        FLAGS.num_classes, input_size=FLAGS.image_size)
    model.load_state_dict(state, strict=False)
    data = batches_of(raw, int(os.environ.get('BATCH', '64')),
                      int(os.environ.get('BATCHES', '8')))
    gains = [(n, m) for n, m in model.named_modules()
             if hasattr(m, 'bn') and getattr(m, 'weight', None) is not None
             and m.weight.dim() == 1 and len(m.weight) >= 8]
    crit = torch.nn.CrossEntropyLoss()
    out = {}
    for width in (1.0, 0.5):
        model.apply(lambda m: setattr(m, 'width_mult', width))
        taylor = {n: torch.zeros_like(m.weight) for n, m in gains}
        fisher = {n: torch.zeros_like(m.weight) for n, m in gains}
        for images, target in data:
            model.zero_grad(set_to_none=True)
            crit(model(images), target).backward()
            for n, m in gains:
                if m.weight.grad is not None:
                    taylor[n] += (m.weight * m.weight.grad).abs().detach()
                    fisher[n] += m.weight.grad.detach().pow(2)
        names = [n for n, _ in gains]
        out[width] = ([taylor[n] for n in names], [fisher[n] for n in names])
    return out


def main():
    path = _ARGS[0] if _ARGS else 'k_best_model.zip'
    if not os.path.exists(path):
        print(__doc__)
        return 1
    torch.manual_seed(1995)
    blob = torch.load(path, map_location='cpu', weights_only=False)
    state = {k.replace('module.', '', 1): v
             for k, v in blob.get('model', blob).items()}

    convs = sorted({k[:-len('.weight')] for k, v in state.items()
                    if k.endswith('.weight') and v.dim() == 4
                    and v.size(0) >= 8})
    bns = sorted({k[:-len('.weight')] for k, v in state.items()
                  if k.endswith('.weight') and v.dim() == 1
                  and len(v) >= 8
                  and any(j.startswith(k[:-len('.weight')] + '.bn.')
                          for j in state)})
    print('{}: {} conv layers, {} batch-norm gains'.format(
        os.path.basename(path), len(convs), len(bns)))

    scores = {'L1 of the conv filter': [], 'L2 of the conv filter': [],
              'distance from the others': []}
    for name in convs:
        w = state[name + '.weight']
        scores['L1 of the conv filter'].append(magnitude(w, 1))
        scores['L2 of the conv filter'].append(magnitude(w, 2))
        scores['distance from the others'].append(redundancy(w))

    read = []
    for i, name in enumerate(convs):
        w = state[name + '.weight']
        nxt = state[convs[i + 1] + '.weight'] if i + 1 < len(convs) else None
        d = downstream(nxt, w.size(0))
        if d is not None:
            read.append(d)
    if read:
        scores['read by the next layer'] = read

    if len(bns) == len(convs):
        scores['absolute batch-norm gain'] = [
            state[b + '.weight'].abs() for b in bns]

    raw, where = find_train()
    if raw is None:
        print('no readable CIFAR-100: skipping Taylor and Fisher')
    else:
        print('reading {}'.format(where))
        got = loss_based(state, raw)
        scores['Taylor at width 1.00'] = got[1.0][0]
        scores['Fisher at width 1.00'] = got[1.0][1]
        scores['Taylor at width 0.50'] = got[0.5][0]

    print()
    print('share of importance held by the first quarter of the channels')
    print('{:<28} {:>9} {:>9} {:>9} {:>11}'.format(
        'criterion', 'shuffled', 'now', 'best', 'recovered'))
    order = ['L1 of the conv filter', 'L2 of the conv filter',
             'absolute batch-norm gain', 'read by the next layer',
             'distance from the others', 'Fisher at width 1.00',
             'Taylor at width 1.00', 'Taylor at width 0.50']
    present = [k for k in order if k in scores]
    for key in present:
        summarise(key, scores[key])

    base = ('Taylor at width 1.00' if 'Taylor at width 1.00' in scores
            else 'L1 of the conv filter')
    print()
    print('do the criteria rank the channels the same way, against {}?'
          .format(base))
    for key in present:
        if key == base or len(scores[key]) != len(scores[base]):
            continue
        mean, worst = agreement(scores[base], scores[key])
        print('  {:<28} mean {:+.3f}   min {:+.3f}'.format(key, mean, worst))
    return 0


if __name__ == '__main__':
    sys.exit(main())
