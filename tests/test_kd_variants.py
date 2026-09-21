"""the extra places a transport cost can be put, and the controls for them

Three independent axes, each with a control that says whether the axis is
doing anything:

  cost_source        fc | confusion | identity
  horizontal_loss    wasserstein | jeffreys | kl
  feature_loss       wasserstein | sliced | mmd | mse
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
from utils.loss_ops import FeatureMMDLoss
from utils.loss_ops import FeatureMSELoss
from utils.loss_ops import FeatureGromovLoss
from utils.loss_ops import FeatureSlicedWassersteinLoss
from utils.loss_ops import FeatureUnbalancedWassersteinLoss
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


def feature_losses():
    """the feature_loss axis, in the order the decomposition reads"""
    return [
        ('mse', FeatureMSELoss(align='prefix')),
        ('mmd', FeatureMMDLoss(align='prefix')),
        ('sliced', FeatureSlicedWassersteinLoss(align='prefix')),
        ('wasserstein', FeatureWassersteinLoss(align='prefix')),
    ]


def test_every_feature_loss_is_zero_at_agreement():
    """whatever else they disagree about, all four must agree here

    A width that already matches must be charged nothing, or the term
    pushes a run away from the thing it is asking for. Entropic transport
    needs an explicit correction to manage it; the other three get it for
    free, which is worth knowing when reading their numbers.
    """
    generator = torch.Generator().manual_seed(31)
    teacher = torch.randn(24, 64, generator=generator)
    nudge = teacher + 0.15 * torch.randn(24, 64, generator=generator)
    far = teacher + 1.5
    for name, loss_fn in feature_losses():
        torch.manual_seed(5)
        same = loss_fn(teacher.clone(), teacher).item()
        torch.manual_seed(5)
        near = loss_fn(nudge, teacher).item()
        torch.manual_seed(5)
        away = loss_fn(far, teacher).item()
        check('{:11} is zero at agreement'.format(name), abs(same) < 1e-5,
              'value {:.2e}'.format(same))
        check('{:11} grows with the gap'.format(name), same < near < away,
              '{:.4f} < {:.4f} < {:.4f}'.format(same, near, away))


def test_every_feature_loss_is_symmetric():
    """the horizontal slot has no teacher, so the term cannot have one"""
    generator = torch.Generator().manual_seed(32)
    left = torch.randn(24, 64, generator=generator)
    right = 0.4 + torch.randn(24, 64, generator=generator)
    for name, loss_fn in feature_losses():
        torch.manual_seed(6)
        forward = loss_fn(left, right).item()
        torch.manual_seed(6)
        backward = loss_fn(right, left).item()
        check('{:11} is symmetric'.format(name),
              abs(forward - backward) < 1e-5,
              '{:.6f} against {:.6f}'.format(forward, backward))


def test_only_mse_cares_which_sample_is_which():
    """the property the axis exists to separate

    Shuffling one cloud changes nothing about it as a distribution. mse
    reads the shuffle as a large disagreement because it compares sample i
    with sample i; the other three are set functions and cannot see it at
    all. If a branch on this axis moves, this is the difference it moved
    on.
    """
    generator = torch.Generator().manual_seed(33)
    teacher = torch.randn(24, 64, generator=generator)
    student = teacher + 0.3 * torch.randn(24, 64, generator=generator)
    order = torch.randperm(24, generator=generator)
    for name, loss_fn in feature_losses():
        torch.manual_seed(7)
        plain = loss_fn(student, teacher).item()
        torch.manual_seed(7)
        shuffled = loss_fn(student[order], teacher[order]).item()
        drift = abs(plain - shuffled)
        if name == 'mse':
            # both sides permuted the same way, so mse is unchanged too;
            # permuting one side is what separates them
            torch.manual_seed(7)
            one_side = loss_fn(student[order], teacher).item()
            check('mse         sees a one sided shuffle',
                  one_side > 2.0 * plain,
                  'paired {:.4f}, shuffled {:.4f}'.format(plain, one_side))
            continue
        torch.manual_seed(7)
        one_side = loss_fn(student[order], teacher).item()
        check('{:11} ignores a one sided shuffle'.format(name),
              abs(one_side - plain) < 1e-4,
              'paired {:.6f}, shuffled {:.6f}'.format(plain, one_side))
        check('{:11} ignores a paired shuffle'.format(name), drift < 1e-4,
              'drift {:.2e}'.format(drift))


def test_every_feature_loss_descends():
    """each one has to move a cloud it is pointed at

    The step sizes differ by two orders of magnitude across the axis, which
    is the thing to remember when setting feature_weight: these read on
    comparable scales but their gradients do not.
    """
    for name, loss_fn in feature_losses():
        generator = torch.Generator().manual_seed(34)
        teacher = torch.randn(24, 32, generator=generator)
        student = (teacher + 1.0).clone().requires_grad_()
        torch.manual_seed(8)
        before = loss_fn(student, teacher).item()
        for _ in range(200):
            loss = loss_fn(student, teacher)
            if student.grad is not None:
                student.grad.zero_()
            loss.backward()
            with torch.no_grad():
                student -= 5.0 * student.grad
        torch.manual_seed(8)
        after = loss_fn(student, teacher).item()
        check('{:11} closes the gap'.format(name), after < 0.5 * before,
              '{:.4f} -> {:.4f}'.format(before, after))


def test_no_feature_loss_reads_the_dimension():
    """one weight has to mean the same thing at every width

    The feature dimension moves with the width, 128 channels at 0.25
    against 512 at 1.00, and feature_weight is a single number shared by
    all of them. A loss whose value tracks the dimension is therefore
    applying a width schedule nobody wrote down, and the branch would be
    measuring that schedule rather than the loss.

    The sliced version failed this when it was first written, by exactly
    the factor of the dimension that random projection removes.
    """
    for name, loss_fn in feature_losses():
        values = []
        for dim in (64, 256):
            generator = torch.Generator().manual_seed(36)
            teacher = torch.randn(32, dim, generator=generator)
            student = teacher + 0.5 * torch.randn(
                32, dim, generator=generator)
            torch.manual_seed(10)
            values.append(loss_fn(student, teacher).item())
        ratio = values[1] / max(values[0], 1e-12)
        check('{:11} is the same at 64 and 256 channels'.format(name),
              0.5 < ratio < 2.0,
              '{:.4f} and {:.4f}, ratio {:.2f}'.format(
                  values[0], values[1], ratio))


def test_sliced_ranks_clouds_like_the_full_transport():
    """the cost argument for sliced only holds if it measures the same thing

    Sliced transport is cheap because it replaces the plan with a sort
    along random directions. That is only a saving if it orders pairs of
    clouds the way the full version does, so this compares them over a
    range of displacements rather than trusting the name.
    """
    generator = torch.Generator().manual_seed(35)
    teacher = torch.randn(32, 48, generator=generator)
    sliced = FeatureSlicedWassersteinLoss(align='prefix', n_projections=256)
    full = FeatureWassersteinLoss(align='prefix')
    rows = []
    for gap in (0.1, 0.3, 0.6, 1.0, 2.0):
        student = teacher + gap * torch.randn(
            32, 48, generator=generator)
        torch.manual_seed(9)
        rows.append((gap, sliced(student, teacher).item(),
                     full(student, teacher).item()))
    print('        gap      sliced   wasserstein')
    for gap, left, right in rows:
        print('        {:.1f}   {:9.4f}   {:9.4f}'.format(gap, left, right))
    ordered = all(
        rows[i][1] < rows[i + 1][1] and rows[i][2] < rows[i + 1][2]
        for i in range(len(rows) - 1))
    check('sliced and full transport order the gaps alike', ordered)


def test_unbalanced_stays_monotone_at_the_shipped_tau():
    """the setting below which this term stops being a distance

    Unbalanced transport is debiased the same way the balanced one is, and
    that correction is only valid while the plan still carries most of the
    mass. At small tau it does not: the cross term shrinks faster than the
    self terms it is measured against and the value goes negative on the
    clouds that disagree most, which is the opposite of what the term is
    for. tau 1.0 is the shipped value and the lowest that survives this.
    """
    generator = torch.Generator().manual_seed(37)
    teacher = torch.randn(48, 128, generator=generator)
    base = teacher[:, :32].clone()
    gaps = [0.0, 0.4, 1.2, 3.0]
    print('      value by gap, by tau:')
    for tau in (0.1, 0.5, 1.0, 3.0):
        loss_fn = FeatureUnbalancedWassersteinLoss(tau=tau, align='prefix')
        values = []
        for gap in gaps:
            student = base + gap * torch.randn(
                48, 32, generator=torch.Generator().manual_seed(38))
            values.append(loss_fn(student, teacher).item())
        print('        tau {:<5.1f}'.format(tau) + ''.join(
            '{:>10.4f}'.format(v) for v in values))
        if tau < 1.0:
            continue
        rising = all(values[i] < values[i + 1] for i in range(len(gaps) - 1))
        check('tau {} rises with the gap'.format(tau), rising)
        check('tau {} is zero at agreement'.format(tau),
              abs(values[0]) < 1e-5, 'value {:.2e}'.format(values[0]))


def test_cosine_ground_behaves_like_a_cost():
    """the other ground metric has to satisfy the same three things"""
    generator = torch.Generator().manual_seed(39)
    teacher = torch.randn(32, 96, generator=generator)
    base = teacher[:, :48].clone()
    loss_fn = FeatureWassersteinLoss(align='prefix', ground='cosine')
    same = loss_fn(teacher.clone(), teacher).item()
    near = loss_fn(
        base + 0.3 * torch.randn(32, 48, generator=generator),
        teacher).item()
    far = loss_fn(
        base + 2.0 * torch.randn(32, 48, generator=generator),
        teacher).item()
    check('cosine ground is zero at agreement', abs(same) < 1e-5,
          'value {:.2e}'.format(same))
    check('cosine ground rises with the gap', same < near < far,
          '{:.4f} < {:.4f} < {:.4f}'.format(same, near, far))
    left = torch.randn(32, 48, generator=generator)
    check('cosine ground is symmetric',
          abs(loss_fn(left, teacher).item()
              - loss_fn(teacher, left).item()) < 1e-5)


def test_gromov_is_invariant_to_what_it_claims():
    """kept for the record: it does what it says and measures almost nothing

    Gromov transport compares the clouds' internal distances, so it is
    blind to anything that leaves those alone. This holds it to that, and
    to the reason no branch runs it: on these clouds the two internal
    structures are nearly identical after each is scaled by its own
    spread, so the value sits near zero however far apart the clouds are,
    and it does not even rank the gaps in order.
    """
    generator = torch.Generator().manual_seed(43)
    teacher = torch.randn(48, 96, generator=generator)
    base = teacher[:, :32].clone()
    loss_fn = FeatureGromovLoss()
    check('gromov is zero at agreement',
          abs(loss_fn(teacher.clone(), teacher).item()) < 1e-5)
    plain = loss_fn(base, teacher).item()
    check('gromov ignores a translation',
          abs(loss_fn(base + 3.0, teacher).item() - plain) < 1e-5)
    order = torch.randperm(32, generator=generator)
    check('gromov ignores a channel permutation',
          abs(loss_fn(base[:, order], teacher).item() - plain) < 1e-5)
    far = loss_fn(
        base + 3.0 * torch.randn(48, 32, generator=generator), teacher).item()
    check('and it is too small to use, which is why no branch does',
          abs(far) < 0.01,
          'largest gap reads {:.5f}, against about 0.4 for transport'.format(
              far))


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
    test_every_feature_loss_is_zero_at_agreement()
    test_every_feature_loss_is_symmetric()
    test_only_mse_cares_which_sample_is_which()
    test_every_feature_loss_descends()
    test_no_feature_loss_reads_the_dimension()
    test_sliced_ranks_clouds_like_the_full_transport()
    test_unbalanced_stays_monotone_at_the_shipped_tau()
    test_cosine_ground_behaves_like_a_cost()
    test_gromov_is_invariant_to_what_it_claims()
    print()
    if FAILURES:
        print('{} failed: {}'.format(len(FAILURES), ', '.join(FAILURES)))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
