"""head_groups: one classifier per band of widths.

Built to fail if the flag is silently ignored: the narrow widths have to
read a different head from the widest one, a width's backward has to
reach its own head and no other, and the widest width has to keep the
head the class cost matrix and the teacher read. head_groups 1 has to
build the model it always did, key for key.

Runs on the CPU in a few seconds: python tests/test_head_groups.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = [sys.argv[0], 'app:apps/smoke_a_kl.yml']

import torch  # noqa: E402

from utils.config import FLAGS  # noqa: E402

WIDTHS = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
          0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print('ok   ', what)


def build(groups):
    import importlib
    FLAGS.head_groups = groups
    FLAGS.reset_parameters = True
    FLAGS.return_features = False
    torch.manual_seed(0)
    model = importlib.import_module('models.us_resnet').Model(100, 32)
    model.train()
    return model


def set_width(model, width):
    model.apply(lambda m: setattr(m, 'width_mult', width))


if __name__ == '__main__':
    from utils.loss_ops import get_classifier_weight

    plain = build(1)
    keys = set(plain.state_dict())
    del FLAGS.head_groups
    torch.manual_seed(0)
    import importlib
    before = importlib.import_module('models.us_resnet').Model(100, 32)
    check(keys == set(before.state_dict()) and
          not any('narrow_heads' in k for k in keys),
          'head_groups 1 builds the same state dict as no flag at all')

    model = build(4)
    bands = []
    for width in WIDTHS:
        set_width(model, width)
        head = model.head()
        bands.append(len(model.narrow_heads) if head is model.classifier
                     else list(model.narrow_heads).index(head))
    check([bands.count(b) for b in range(4)] == [4, 4, 4, 4],
          'the sixteen test widths fall four to a band: {}'.format(bands))
    check(bands[-1] == 3 and bands[0] == 0,
          '0.25 reads the first narrow head and 1.00 reads classifier')
    check(get_classifier_weight(model) is model.classifier[0].weight,
          'the class cost matrix still reads the widest head')

    x = torch.randn(4, 3, 32, 32)
    set_width(model, 0.25)
    own = model(x)
    features = model.features(x).view(4, -1)
    shared = model.classifier(features)
    check((own - shared).abs().max().item() > 1e-3,
          'width 0.25 does not read the widest head')

    for width, band in ((0.25, 0), (0.5, 1), (0.7, 2), (1.0, 3)):
        model.zero_grad(set_to_none=True)
        set_width(model, width)
        model(x).sum().backward()
        heads = list(model.narrow_heads) + [model.classifier[0]]
        touched = [h.weight.grad is not None for h in heads]
        check(touched == [b == band for b in range(4)],
              'a backward at {} reaches head {} and no other'.format(
                  width, band))

    FLAGS.return_features = True
    FLAGS.feature_layers = ['final']
    set_width(model, 0.25)
    logits, _ = model(x)
    check((logits - own).abs().max().item() < 1e-5,
          'the feature-returning path reads the same head')
    FLAGS.return_features = False
    print('all checks passed')
