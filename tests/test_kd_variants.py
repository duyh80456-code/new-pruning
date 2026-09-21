"""the extra places a transport cost can be put, and the controls for them

Three independent axes, each with a control that says whether the axis is
doing anything:

  cost_source        fc | confusion | identity
  horizontal_loss    wasserstein | jeffreys | kl
  tier               logit | feature, prefix or gram alignment

Runs on the CPU in a few seconds.
"""
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not any(arg.startswith('app:') for arg in sys.argv):
    sys.argv.append('app:apps/cifar100_c_wasserstein.yml')

import torch

from utils.config import FLAGS

from utils.loss_ops import ConfusionEmbedding
from utils.loss_ops import FeatureWassersteinLoss
from utils.loss_ops import JeffreysPairLoss
from utils.loss_ops import KLPairLoss
from utils.loss_ops import MultiTierFeatureLoss
from utils.loss_ops import WassersteinPairLoss
from utils.loss_ops import class_cost_matrix
from utils.loss_ops import feature_cost
from utils.loss_ops import identity_cost_matrix
from utils.loss_ops import sinkhorn_plan
from utils.loss_ops import width_gate


FAILURES = []


def check(name, condition, detail=''):
    status = 'pass' if condition else 'FAIL'
    print('{:5} {}{}'.format(status, name, '  ' + detail if detail else ''))
    if not condition:
        FAILURES.append(name)


def spread(cost):
    off = cost + torch.eye(cost.size(0)) * 1e9
    return (cost.max(dim=1)[0].mean()
            / off.min(dim=1)[0].mean().clamp_min(1e-12)).item()


def test_identity_cost_is_flat():
    cost = identity_cost_matrix(50)
    check('identity cost is zero on the diagonal',
          cost.diagonal().abs().max().item() < 1e-6)
    off = cost[~torch.eye(50, dtype=torch.bool)]
    check('and constant off it',
          (off - off[0]).abs().max().item() < 1e-6,
          'spread {:.2f}'.format(spread(cost)))


def test_confusion_recovers_structure():
    """the point of the alternative: a teacher that mixes classes up

    Ten classes in five pairs, the teacher splitting its mass inside each
    pair. The confusion embedding should put the partners close and
    everything else far, which is what the classifier rows failed to do.
    """
    n = 10
    generator = torch.Generator().manual_seed(4)
    confusion = ConfusionEmbedding(n, momentum=0.3, warmup=2)
    for _ in range(40):
        target = torch.randint(0, n, (64,), generator=generator)
        partner = target ^ 1  # 0 with 1, 2 with 3, and so on
        prob = torch.full((64, n), 0.02)
        prob.scatter_(1, target.view(-1, 1), 0.55)
        prob.scatter_(1, partner.view(-1, 1), 0.27)
        prob = prob / prob.sum(dim=1, keepdim=True)
        confusion.update(prob, target)

    check('confusion warms up', confusion.ready())
    cost = class_cost_matrix(confusion.embedding())
    partners = torch.tensor([cost[i, i ^ 1] for i in range(n)])
    others = torch.tensor([
        cost[i, j] for i in range(n) for j in range(n)
        if j != i and j != (i ^ 1)])
    check(
        'confused classes come out closer than unrelated ones',
        partners.mean().item() < 0.5 * others.mean().item(),
        'partners {:.3f}, others {:.3f}, spread {:.2f}'.format(
            partners.mean().item(), others.mean().item(), spread(cost)))


def test_confusion_falls_back_before_warmup():
    confusion = ConfusionEmbedding(10, warmup=1000)
    check('not ready before warmup', not confusion.ready())


def test_feature_cost_handles_unequal_widths():
    """two widths never share a dimension, which is the whole difficulty"""
    student = torch.randn(8, 24)
    teacher = torch.randn(8, 64)
    for align in ('prefix', 'gram'):
        cost = feature_cost(student, teacher, align)
        check('{} alignment gives a full cost matrix'.format(align),
              cost.shape == (8, 8),
              'shape {}'.format(tuple(cost.shape)))
        check('{} cost is finite and positive'.format(align),
              torch.isfinite(cost).all() and cost.min().item() >= 0)


def test_feature_loss_is_zero_on_a_copy():
    """a width already matching the teacher should be charged nothing

    Entropic transport between a cloud and itself is not zero, so this only
    holds because the loss is debiased, the same correction the logit
    version needs and for the same reason.
    """
    generator = torch.Generator().manual_seed(6)
    teacher = torch.randn(16, 64, generator=generator)
    loss_fn = FeatureWassersteinLoss(align='prefix')
    same = loss_fn(teacher.clone(), teacher).item()
    nudged = loss_fn(
        teacher + 0.2 * torch.randn(16, 64, generator=generator),
        teacher).item()
    moved = loss_fn(teacher + 2.0, teacher).item()
    check('an identical cloud costs nothing', same < 1e-3,
          'value {:.2e}'.format(same))
    check('a nudged cloud costs a little, a displaced one a lot',
          same < nudged < moved,
          'same {:.3f}, nudged {:.3f}, displaced {:.3f}'.format(
              same, nudged, moved))


def test_prefix_alignment_ignores_dropped_channels():
    """worth knowing before reading any result from this term

    prefix truncates the teacher to the student's width, so the channels a
    narrow subnet does not have are not compared at all. It asks whether
    the shared coordinates agree, never what was lost by dropping the rest.
    Defensible for a nested supernet, which cannot represent them anyway,
    but it is an assumption and not a neutral choice.
    """
    generator = torch.Generator().manual_seed(7)
    teacher = torch.randn(16, 64, generator=generator)
    student = teacher[:, :16].clone()
    untouched = FeatureWassersteinLoss(align='prefix')(student, teacher)
    scrambled = teacher.clone()
    scrambled[:, 16:] = torch.randn(16, 48, generator=generator)
    after = FeatureWassersteinLoss(align='prefix')(student, scrambled)
    check(
        'changing only the dropped channels changes nothing',
        abs(untouched.item() - after.item()) < 1e-5,
        '{:.2e} against {:.2e}'.format(untouched.item(), after.item()))


def test_feature_loss_pulls_the_cloud():
    """the gradient is small by construction, so size the step for it

    Dividing by the spread of the clouds makes the loss dimensionless, and
    divides the gradient by that spread too. Useful to know when choosing
    feature_weight: the loss reads on the scale of a cross entropy, the
    gradient does not.
    """
    generator = torch.Generator().manual_seed(8)
    student = torch.randn(16, 32, generator=generator).requires_grad_()
    teacher = 4.0 + torch.randn(16, 32, generator=generator)
    loss_fn = FeatureWassersteinLoss(align='prefix')
    before = loss_fn(student, teacher).item()
    for _ in range(400):
        loss = loss_fn(student, teacher)
        if student.grad is not None:
            student.grad.zero_()
        loss.backward()
        with torch.no_grad():
            student -= 20.0 * student.grad
    after = loss_fn(student, teacher).item()
    check('descent closes the gap between the clouds', after < before * 0.3,
          '{:.3f} -> {:.3f}'.format(before, after))


def test_multi_tier_matches_single_when_there_is_one_tap():
    """adding tiers must not quietly rescale the one-tier case"""
    generator = torch.Generator().manual_seed(12)
    student = torch.randn(16, 24, generator=generator)
    teacher = torch.randn(16, 64, generator=generator)
    single = FeatureWassersteinLoss(align='prefix')(student, teacher)
    multi = MultiTierFeatureLoss(align='prefix')((student,), (teacher,))
    check('one tap through the multi-tier wrapper is unchanged',
          abs(single.item() - multi.item()) < 1e-6,
          '{:.4f} against {:.4f}'.format(single.item(), multi.item()))


def test_multi_tier_averages_and_weights():
    generator = torch.Generator().manual_seed(13)
    shallow = (torch.randn(16, 32, generator=generator),
               torch.randn(16, 128, generator=generator))
    deep = (torch.randn(16, 24, generator=generator),
            torch.randn(16, 64, generator=generator))
    students, teachers = (shallow[0], deep[0]), (shallow[1], deep[1])

    each = [FeatureWassersteinLoss(align='prefix')(s, t).item()
            for s, t in zip(students, teachers)]
    equal = MultiTierFeatureLoss(align='prefix')(students, teachers).item()
    check('equal weights average the tiers',
          abs(equal - sum(each) / 2) < 1e-6,
          'tiers {:.4f} and {:.4f}, combined {:.4f}'.format(
              each[0], each[1], equal))

    only_deep = MultiTierFeatureLoss(
        align='prefix', tier_weights=[0.0, 1.0])(students, teachers).item()
    check('a zero weight removes a tier',
          abs(only_deep - each[1]) < 1e-6,
          '{:.4f} against {:.4f}'.format(only_deep, each[1]))


def test_width_gate_turns_the_term_off_where_it_hurt():
    """the measured shape, made into a schedule

    Against plain KL the horizontal term ran from about +0.5 at width 0.30
    to -0.8 at 1.00. 'narrow' is full strength at the bottom of the range
    and zero at the top, so the part that lost is not applied.
    """
    low, high = FLAGS.width_mult_range[0], FLAGS.width_mult_range[-1]
    middle = 0.5 * (low + high)

    FLAGS.weight_schedule = 'constant'
    check('constant is flat',
          width_gate(low) == 1.0 and width_gate(high) == 1.0)

    FLAGS.weight_schedule = 'narrow'
    check('narrow is full at the smallest width',
          abs(width_gate(low) - 1.0) < 1e-9,
          'value {:.3f}'.format(width_gate(low)))
    check('and off at the largest',
          abs(width_gate(high)) < 1e-9,
          'value {:.3f}'.format(width_gate(high)))
    check('and monotone between',
          width_gate(low) > width_gate(middle) > width_gate(high),
          'middle {:.3f}'.format(width_gate(middle)))

    FLAGS.weight_schedule = 'wide'
    check('wide is the mirror image',
          abs(width_gate(high) - 1.0) < 1e-9
          and abs(width_gate(low)) < 1e-9)

    FLAGS.weight_schedule = 'narrow'
    check('a width outside the range is clamped, not extrapolated',
          0.0 <= width_gate(high + 1.0) <= 1.0
          and 0.0 <= width_gate(low - 1.0) <= 1.0)
    FLAGS.weight_schedule = 'constant'


def test_transport_plan_is_a_coupling():
    """on the cost the feature loss actually hands it, which is scaled

    An earlier version of this fed it raw distances and failed at random,
    because eps is meaningful only relative to the cost. The scaling in
    feature_cost is what makes one eps work for every width.
    """
    generator = torch.Generator().manual_seed(9)
    worst = 0.0
    for size in (12, 64):
        cost = feature_cost(
            torch.randn(size, 24, generator=generator),
            torch.randn(size, 64, generator=generator))
        plan = sinkhorn_plan(cost, eps=0.2, n_iters=200)
        worst = max(
            worst,
            (plan.sum(dim=1) - 1.0 / size).abs().max().item(),
            (plan.sum(dim=0) - 1.0 / size).abs().max().item())
        check('plan is detached', not plan.requires_grad)
    check('rows and columns sum to the uniform marginals',
          worst < 1e-5,
          'worst violation {:.2e}'.format(worst))


def test_horizontal_controls_separate_the_claims():
    """kl asymmetric, jeffreys symmetric, wasserstein symmetric and metric

    If the three agreed there would be nothing to attribute the effect to.
    """
    generator = torch.Generator().manual_seed(5)
    a = 2.0 * torch.randn(32, 20, generator=generator)
    b = 2.0 * torch.randn(32, 20, generator=generator)

    kl = KLPairLoss(reduction='none')
    gap = (kl(a, b) - kl(b, a)).abs().max().item()
    check('kl is asymmetric, as the argument assumes', gap > 1e-2,
          'max gap {:.3f}'.format(gap))

    jeffreys = JeffreysPairLoss(reduction='none')
    gap = (jeffreys(a, b) - jeffreys(b, a)).abs().max().item()
    check('jeffreys is symmetric', gap < 1e-5,
          'max gap {:.2e}'.format(gap))
    check('and non-negative', jeffreys(a, b).min().item() >= -1e-6)

    cost = class_cost_matrix(torch.randn(20, 40))
    pair = WassersteinPairLoss()
    pair.set_cost(cost)
    gap = (pair(a, b) - pair(b, a)).abs().max().item()
    check('wasserstein is symmetric too', gap < 1e-5,
          'max gap {:.2e}'.format(gap))

    flat = identity_cost_matrix(20)
    pair.set_cost(flat)
    metric = pair(a, b)
    pair.set_cost(cost)
    structured = pair(a, b)
    difference = (metric - structured).abs().mean().item()
    check(
        'and the cost matrix changes its value, or it is not using one',
        difference > 1e-3,
        'mean difference between flat and structured cost {:.4f}'.format(
            difference))


def test_all_controls_reach_the_gradient():
    for name, loss_fn in (('kl', KLPairLoss(reduction='none')),
                          ('jeffreys', JeffreysPairLoss(reduction='none'))):
        a = torch.randn(8, 12, requires_grad=True)
        b = torch.randn(8, 12, requires_grad=True)
        loss_fn(a, b).mean().backward()
        check('{} reaches both students'.format(name),
              a.grad is not None and b.grad is not None
              and a.grad.abs().sum().item() > 0
              and b.grad.abs().sum().item() > 0)


def main():
    print('torch', torch.__version__)
    print()
    test_identity_cost_is_flat()
    test_confusion_recovers_structure()
    test_confusion_falls_back_before_warmup()
    test_feature_cost_handles_unequal_widths()
    test_feature_loss_is_zero_on_a_copy()
    test_prefix_alignment_ignores_dropped_channels()
    test_feature_loss_pulls_the_cloud()
    test_multi_tier_matches_single_when_there_is_one_tap()
    test_multi_tier_averages_and_weights()
    test_width_gate_turns_the_term_off_where_it_hurt()
    test_transport_plan_is_a_coupling()
    test_horizontal_controls_separate_the_claims()
    test_all_controls_reach_the_gradient()
    print()
    if FAILURES:
        print('{} failed: {}'.format(len(FAILURES), ', '.join(FAILURES)))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
