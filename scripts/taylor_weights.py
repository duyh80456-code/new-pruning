"""Compute a fixed per-channel weight from a finished run.

    python scripts/taylor_weights.py k_best_model.zip -o results/taylor.pt

Branch BA estimates the Taylor score while it trains, from its own
gradients, which costs a hook and assumes the ordering drifts slowly
enough that an average over the last twenty firings still applies. This
computes the same score once, from a checkpoint, and writes it out to be
used as a constant.

What that buys is every assumption gone: no drift, no warmup, no hook,
and a weight you can look at before the run rather than after. What it
costs is that the number comes from a model that has already finished,
so a branch using it is a two-run method and is not comparable to K on
one run. Both readings are worth having, which is why this exists
beside BA rather than instead of it.

Measured across the middle widths, not at 1.00. The feature term runs
between two sampled middle widths and never at the top, and importance
at 1.00 is the one reading that is certainly about the wrong thing: at
that width 512 channels are live and the term never sees more than a
fraction of them at once. Each width is normalised over its own channels
before the average, so a width does not count for more by having more of
them.
"""
import argparse
import os
import pickle
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

WIDTHS = (0.75, 0.6, 0.5, 0.4)


def find_train():
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
    raise SystemExit('no readable cifar-100-python/train')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('-o', '--out', default='results/taylor.pt')
    parser.add_argument('-b', '--batches', type=int, default=16)
    args = parser.parse_args()

    sys.argv = [sys.argv[0], 'app:apps/cifar100_k_feature_pair.yml']
    import importlib
    from utils.config import FLAGS

    raw, where = find_train()
    print('reading {}'.format(where))
    torch.manual_seed(1995)
    images = torch.from_numpy(
        raw[b'data'].reshape(-1, 3, 32, 32)).float().div_(255.0)
    mean = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
    std = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)
    images = (images - mean) / std
    labels = torch.tensor(raw[b'fine_labels'])
    take = torch.randperm(len(labels))[:64 * args.batches]
    data = [(images[take[i:i + 64]], labels[take[i:i + 64]])
            for i in range(0, len(take), 64)]

    blob = torch.load(args.checkpoint, map_location='cpu',
                      weights_only=False)
    state = {k.replace('module.', '', 1): v
             for k, v in blob.get('model', blob).items()}
    model = importlib.import_module(FLAGS.model).Model(
        FLAGS.num_classes, input_size=FLAGS.image_size)
    model.load_state_dict(state, strict=False)

    pool = [m for m in model.modules()
            if isinstance(m, torch.nn.AdaptiveAvgPool2d)][-1]
    held = {}

    def hook(mod, inp, out):
        out.retain_grad()
        held['f'] = out

    pool.register_forward_hook(hook)
    crit = torch.nn.CrossEntropyLoss()

    total = None
    counts = None
    print('{:>7} {:>7} {:>12}'.format('width', 'dims', 'top half'))
    for width in WIDTHS:
        model.apply(lambda m: setattr(m, 'width_mult', width))
        score = None
        for x, y in data:
            model.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            f = held['f']
            piece = (f.detach().flatten(1) * f.grad.flatten(1)).abs().sum(0)
            score = piece if score is None else score + piece
        # normalised over its own channels, so a wider reading does not
        # count for more by having more of them
        score = score / score.mean().clamp_min(1e-12)
        k = score.size(0)
        if total is None:
            total = torch.zeros(512)
            counts = torch.zeros(512)
        total[:k] += score
        counts[:k] += 1.0
        half = float(score.sort(descending=True).values[:k // 2].sum()
                     / score.sum())
        print('{:7.2f} {:7d} {:11.0f}%'.format(width, k, 100 * half))

    weight = torch.where(counts > 0, total / counts.clamp_min(1.0),
                         torch.ones_like(total))
    weight = weight / weight.mean().clamp_min(1e-12)
    out = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) \
        else args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({'weight': weight, 'widths': list(WIDTHS),
                'batches': args.batches,
                'source': os.path.basename(args.checkpoint)}, out)
    print('\nwrote {} channels to {}'.format(weight.numel(), args.out))
    print('  mean {:.3f}  min {:.3f}  max {:.3f}'.format(
        float(weight.mean()), float(weight.min()), float(weight.max())))
    print('  the first quarter carries {:.0%} of it'.format(
        float(weight[:128].sum() / weight.sum())))
    return 0


if __name__ == '__main__':
    sys.exit(main())
