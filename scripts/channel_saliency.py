"""Does importance sit in the prefix, and do the widths agree on it?

    python scripts/channel_saliency.py k_best_model.zip

channel_order.py ranks channels by what is written in the weights. This
ranks them by what removing one would cost, which is the thing the
sorting idea is really about: the Taylor score |gamma * dL/dgamma| on
each batch-norm gain, summed over a few batches. Zeroing a gain removes
the channel, so a first-order estimate of that is what this is.

It runs the estimate separately at several widths, because the nesting
admits one global permutation and there are sixteen widths to satisfy.
If the widths disagree about which channels matter, no single order can
serve them and the headroom at any one width is beside the point.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], 'app:apps/cifar100_k_feature_pair.yml']
from utils.config import FLAGS                       # noqa: E402


def build():
    import importlib
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes, input_size=FLAGS.image_size)
    return model


def loader(batch, batches):
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.5071, 0.4865, 0.4409),
                                (0.2673, 0.2564, 0.2762))])
    data = torchvision.datasets.CIFAR100(root='data', train=True,
                                         download=False, transform=tf)
    return torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True,
                                       num_workers=0)


def gains(model):
    return [(n, m) for n, m in model.named_modules()
            if hasattr(m, 'weight') and hasattr(m, 'bn')
            and getattr(m, 'weight', None) is not None
            and m.weight.dim() == 1 and len(m.weight) >= 8]


def saliency(model, data, width, batches):
    model.apply(lambda m: setattr(m, 'width_mult', width))
    named = gains(model)
    score = {n: torch.zeros_like(m.weight) for n, m in named}
    crit = torch.nn.CrossEntropyLoss()
    seen = 0
    for images, target in data:
        model.zero_grad(set_to_none=True)
        crit(model(images), target).backward()
        for n, m in named:
            if m.weight.grad is not None:
                score[n] += (m.weight * m.weight.grad).abs().detach()
        seen += 1
        if seen >= batches:
            break
    return score


def prefix_share(v, width=0.25):
    total = v.sum().clamp_min(1e-12)
    k = max(1, int(round(len(v) * width)))
    return (float(v[:k].sum() / total),
            float(v.sort(descending=True).values[:k].sum() / total))


def main():
    path = _ARGS[0] if _ARGS else 'k_best_model.zip'
    blob = torch.load(path, map_location='cpu', weights_only=False)
    state = {k.replace('module.', '', 1): v
             for k, v in blob.get('model', blob).items()}
    model = build()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print('loaded {}  (missing {}, unexpected {})'.format(
        os.path.basename(path), len(missing), len(unexpected)))

    batches = int(os.environ.get('BATCHES', '4'))
    data = list(loader(int(os.environ.get('BATCH', '64')), batches))[:batches]
    widths = [1.0, 0.5, 0.25]
    scores = {}
    for w in widths:
        scores[w] = saliency(model, data, w, batches)
        print('  measured at width {:.2f}'.format(w))

    print('\nTaylor score |gamma * dL/dgamma|, {} batches, width 0.25 prefix'
          .format(batches))
    print('{:>8} {:>10} {:>10} {:>11}'.format(
        'at width', 'now', 'best', 'recovered'))
    for w in widths:
        now = best = 0.0
        names = list(scores[w])
        for n in names:
            a, b = prefix_share(scores[w][n])
            now += a
            best += b
        now, best = now / len(names), best / len(names)
        print('{:8.2f} {:10.3f} {:10.3f} {:>10.0f}%'.format(
            w, now, best, 100 * (now - 0.25) / max(best - 0.25, 1e-9)))

    print('\ndo the widths agree on the order?  rank correlation per layer')
    for a, b in ((1.0, 0.5), (1.0, 0.25), (0.5, 0.25)):
        cors = []
        for n in scores[a]:
            k = min(len(scores[a][n]), len(scores[b][n]))
            x = scores[a][n][:k].argsort(descending=True).argsort().float()
            y = scores[b][n][:k].argsort(descending=True).argsort().float()
            x = x - x.mean()
            y = y - y.mean()
            cors.append(float((x * y).sum() / (x.norm() * y.norm())))
        print('  {:.2f} against {:.2f}:  mean {:+.3f}   min {:+.3f}'.format(
            a, b, sum(cors) / len(cors), min(cors)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
