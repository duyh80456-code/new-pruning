"""every branch config, built and stepped, on the CPU

What this does NOT do: run train.py. The loop below is a reimplementation
of its shape, so anything that lives only in train.py - teacher_chain
reordering the widths, kd_weighting reweighting the KD term - is inert
here. Those are exercised by the smoke config in each notebook, which
runs the real train.py for two iterations before a card is committed to
a long session. A branch that passes here and nothing else has only had
its config and its criteria checked.

A config that does not parse, or a combination of axes that does not wire
together, costs a Kaggle session to discover and seconds to discover here.
This builds the model and every criterion each config asks for, then runs
two training steps of the real loop shape: the sandwich widths, inplace
distillation, whatever vertical and horizontal terms are configured, and
one backward.

    python tests/test_all_branches.py            every cifar100_* config
    python tests/test_all_branches.py e_kl_pair  just that one
"""
import glob
import importlib
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_one(config):
    """each config needs its own process: FLAGS is a module-level singleton"""
    script = os.path.join(ROOT, 'tests', 'test_all_branches.py')
    done = subprocess.run(
        [sys.executable, script, '--single', 'app:' + config],
        capture_output=True, text=True, cwd=ROOT)
    return done.returncode, (done.stdout + done.stderr).strip()


def check_single():
    import random

    import torch

    # Seeded, so the same branch gives the same number twice. Unseeded it
    # could not have caught a regression: K read 22.5315 and 22.4889 on
    # consecutive runs of unchanged code, which is noise wide enough to
    # hide any change worth catching.
    random.seed(1995)
    torch.manual_seed(1995)

    from utils.config import FLAGS
    from utils.loss_ops import build_confusion_embedding
    from utils.loss_ops import build_cost_matrix
    from utils.loss_ops import ClasswiseFeatureLoss
    from utils.loss_ops import build_feature_criterion
    from utils.loss_ops import build_feature_pair_criterion
    from utils.loss_ops import build_pair_criterion
    from utils.loss_ops import build_soft_criterion
    from utils.loss_ops import WassersteinLossSoft, WassersteinPairLoss
    from utils.loss_ops import training_widths
    from utils.loss_ops import width_gate

    FLAGS.width_mult_list = FLAGS.width_mult_range
    soft = build_soft_criterion() if getattr(
        FLAGS, 'inplace_distill', False) else None
    pair = build_pair_criterion()
    feature = build_feature_criterion()
    feature_pair = build_feature_pair_criterion()
    confusion = build_confusion_embedding()
    FLAGS.return_features = (feature is not None
                             or feature_pair is not None)

    model = importlib.import_module(FLAGS.model).Model(
        FLAGS.num_classes, input_size=FLAGS.image_size)

    # Profiling runs its own forward and wants a tensor back. Four feature
    # branches died here on Kaggle, after the loss suites had all passed,
    # because nothing local ever called it.
    if getattr(FLAGS, 'profiling', None):
        import train
        model.apply(lambda m: setattr(m, 'width_mult', FLAGS.width_mult))
        train.profiling(model, use_cuda=False)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # profiling leaves the model in eval, and an arbitrary width has no
    # stored statistics to evaluate with. run_one_epoch calls this too.
    model.train()

    low, high = FLAGS.width_mult_range[0], FLAGS.width_mult_range[-1]
    batch = 8
    # train.py counts epochs from 1, and a branch with a warm-up runs the
    # widest width alone until narrow_start_epoch. That leaves no middle
    # widths, so no pairs, which is a different shape and the one that
    # broke: bd and be passed here, passed the smoke run, and then died
    # on their first real epoch. Both shapes get stepped now, and a
    # branch without a warm-up steps the ordinary one twice as before.
    warm = getattr(FLAGS, 'narrow_start_epoch', 0)
    for epoch in ([1, warm] if warm > 1 else [1, 1]):
        data = torch.randn(batch, 3, FLAGS.image_size, FLAGS.image_size)
        target = torch.randint(0, FLAGS.num_classes, (batch,))
        # the classwise feature term needs this batch's labels, set once a
        # step. run_one_epoch does the same; this harness exists to catch
        # the case where it does not, so it has to follow it here
        for term in (feature, feature_pair):
            if isinstance(term, ClasswiseFeatureLoss):
                term.set_target(target)
        optimizer.zero_grad()

        widths = training_widths(epoch, low, high)
        if isinstance(soft, WassersteinLossSoft) or isinstance(
                pair, WassersteinPairLoss):
            cost = build_cost_matrix(model, confusion)
            if isinstance(soft, WassersteinLossSoft):
                soft.set_cost(cost)
            if isinstance(pair, WassersteinPairLoss):
                pair.set_cost(cost)

        losses, mids, mid_features, mid_widths = [], [], [], []
        teacher_prob, teacher_feature = None, None
        temperature = getattr(FLAGS, 'kd_temperature', 1.0)
        for width in widths:
            model.apply(lambda m: setattr(m, 'width_mult', width))
            out = model(data)
            out, features = out if isinstance(out, tuple) else (out, None)
            if width == high:
                losses.append(torch.mean(criterion(out, target)))
                teacher_prob = torch.softmax(out / temperature, dim=1)
                teacher_feature = features
                if confusion is not None:
                    confusion.update(teacher_prob.detach(), target)
                continue
            loss = (temperature ** 2) * torch.mean(
                soft(out / temperature, teacher_prob.detach()))
            if feature is not None:
                loss = loss + getattr(FLAGS, 'feature_weight', 1.0) * (
                    width_gate(width) * feature(
                        features, tuple(t.detach() for t in teacher_feature)))
            losses.append(loss)
            if width != low:
                mids.append(out)
                mid_features.append(features)
                mid_widths.append(width)

        # mid_widths is empty on a warm-up step, and the average below
        # would divide by zero before the pair terms indexed past the end
        if (pair is not None or feature_pair is not None) and mid_widths:
            gate = width_gate(sum(mid_widths) / len(mid_widths))
            extra = 0.0
            if pair is not None:
                extra = extra + (temperature ** 2) * torch.mean(
                    pair(mids[0] / temperature, mids[1] / temperature))
            if feature_pair is not None:
                extra = extra + feature_pair(mid_features[0],
                                             mid_features[1])
            losses.append(
                getattr(FLAGS, 'horizontal_weight', 1.0) * gate * extra)

        total = sum(losses)
        if not torch.isfinite(total):
            raise ValueError('loss is {}'.format(total.item()))
        total.backward()
        grads = [p.grad.abs().sum().item()
                 for p in model.parameters() if p.grad is not None]
        if not grads or sum(grads) == 0:
            raise ValueError('no gradient reached the weights')
        optimizer.step()

    print('loss {:.4f}'.format(total.item()))


def main():
    if '--single' in sys.argv:
        sys.argv.remove('--single')
        check_single()
        return 0

    wanted = [a for a in sys.argv[1:] if not a.startswith('app:')]
    configs = sorted(glob.glob(os.path.join(ROOT, 'apps', 'cifar100_*.yml')))
    if wanted:
        configs = [c for c in configs
                   if any(w in os.path.basename(c) for w in wanted)]

    failures = []
    for config in configs:
        name = os.path.basename(config)[len('cifar100_'):-len('.yml')]
        code, output = run_one(os.path.relpath(config, ROOT))
        if code == 0:
            last = output.strip().splitlines()[-1] if output.strip() else ''
            print('pass  {:24} {}'.format(name, last))
        else:
            tail = [line for line in output.splitlines() if line.strip()]
            print('FAIL  {:24} {}'.format(
                name, tail[-1] if tail else 'no output'))
            failures.append((name, output))

    print()
    if failures:
        for name, output in failures:
            print('=' * 70)
            print(name)
            print('=' * 70)
            print(output)
        print('\n{} of {} failed'.format(len(failures), len(configs)))
        return 1
    print('all {} branches build and step'.format(len(configs)))
    return 0


if __name__ == '__main__':
    sys.path.insert(0, ROOT)
    sys.exit(main())
