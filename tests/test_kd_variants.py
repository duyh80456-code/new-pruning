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

from utils.loss_ops import channel_weights
from utils.loss_ops import training_widths
from utils.loss_ops import ClasswiseFeatureLoss
from utils.loss_ops import _feature_scale
from utils.loss_ops import ConfusionEmbedding
from utils.loss_ops import CrossEntropyLossSoft
from utils.loss_ops import FeatureBuresLoss
from utils.loss_ops import FeatureChannelLoss
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
from utils.loss_ops import horizontal_pairs
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
        ('channel', FeatureChannelLoss(align='prefix')),
        ('bures', FeatureBuresLoss(align='prefix')),
        ('bures diag', FeatureBuresLoss(align='prefix', diagonal=True)),
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


def test_classwise_is_coarser_rather_than_class_aware():
    """what the classwise form actually buys, which is not what it promised

    It was built to answer an objection: the batch is one unlabelled cloud
    everywhere else, so a narrow width's cat may be explained by the
    teacher's dog and nothing in the objective minds. Two measurements say
    it does not answer that.

    Forbidding cross class matches outright is ruled out by arithmetic.
    Batch 256 over 100 classes is under three images a class, and
    transport between two or three points is a fixed pairing, so the
    masked form would be branch Q by a longer route.

    Transporting between class centroids instead, which is what this does,
    does not forbid it either: ninety odd centroids are still free to
    match a cat's to a dog's. And the labels turn out to matter little.
    Student and teacher are the same batch, so shuffling the labels
    regroups both sides identically and the centroid clouds stay in
    correspondence. The loss moves by the fraction printed below, not by a
    factor.

    What is left is real but smaller than advertised: transport at class
    granularity rather than sample granularity, which drops the within
    class spread from the objective and asks only that the two widths
    place the classes alike.
    """
    generator = torch.Generator().manual_seed(61)
    count, classes = 256, 100
    labels = torch.randint(0, classes, (count,), generator=generator)
    present = labels.unique().numel()
    print('      {} images, {} classes present, {:.2f} per class'.format(
        count, present, float(count) / present))
    check('too few samples a class to transport within one',
          float(count) / present < 4.0)

    centres = 3.0 * torch.randn(classes, 96, generator=generator)
    teacher = centres[labels] + torch.randn(count, 96, generator=generator)
    student = teacher[:, :32] + 0.4 * torch.randn(
        count, 32, generator=generator)

    plain = FeatureWassersteinLoss(align='prefix')
    classwise = ClasswiseFeatureLoss(inner=FeatureWassersteinLoss(
        align='prefix'))

    classwise.set_target(labels)
    before = classwise(student, teacher).item()
    plain_before = plain(student, teacher).item()

    shuffled = labels[torch.randperm(count, generator=generator)]
    classwise.set_target(shuffled)
    after = classwise(student, teacher).item()
    plain_after = plain(student, teacher).item()

    check('the unlabelled loss cannot see a label shuffle',
          abs(plain_before - plain_after) < 1e-6,
          '{:.4f} and {:.4f}'.format(plain_before, plain_after))
    check('the classwise one barely can, which is the finding',
          1.0 < after / before < 1.6,
          'labels right {:.4f}, shuffled {:.4f}, ratio {:.2f}'.format(
              before, after, after / before))
    coarse = classwise(student, teacher).item()
    check('and it is not the sample level loss under another name',
          abs(coarse - plain_before) > 0.2 * plain_before,
          'centroids {:.4f}, samples {:.4f}'.format(coarse, plain_before))

    classwise.set_target(labels)
    check('and it is zero when the two widths agree',
          abs(classwise(teacher.clone(), teacher).item()) < 1e-5)


def test_horizontal_pairs_keeps_the_old_default():
    """the rewrite that let more pairs in must not move the finished runs

    Every result in RESULTS.md was produced with exactly one pair, the two
    middle widths. The loop now collects every co-sampled width instead of
    dropping the narrowest, so the default has to pick out that same pair
    or the table stops being comparable to anything run after it.

    widths arrive as the sandwich rule draws them: narrowest first, then
    the middles.
    """
    widths = [0.25, 0.55, 0.80]  # min, then two middles
    previous = getattr(FLAGS, 'horizontal_pairs', 'middle')
    try:
        FLAGS.horizontal_pairs = 'middle'
        pairs = horizontal_pairs(widths)
        check('the default is the one pair the finished runs used',
              pairs == [(1, 2)], '{}'.format(pairs))

        FLAGS.horizontal_pairs = 'all'
        pairs = horizontal_pairs(widths)
        check('all takes every pair among the students',
              sorted(pairs) == [(0, 1), (0, 2), (1, 2)], '{}'.format(pairs))
        check('and never pairs a width with itself',
              all(left != right for left, right in pairs))

        five = [0.25, 0.4, 0.6, 0.8]
        FLAGS.horizontal_pairs = 'all'
        check('four students give six pairs',
              len(horizontal_pairs(five)) == 6)
        FLAGS.horizontal_pairs = 'middle'
        check('and the default still gives three of them',
              len(horizontal_pairs(five)) == 3)
    finally:
        FLAGS.horizontal_pairs = previous


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


def test_bures_survives_more_channels_than_samples():
    """the shape that killed AK three minutes into a session

    A batch of 256 in 512 channels gives a covariance of rank at most
    255, so over 250 of its eigenvalues are exactly zero. eigh refuses
    that outright on the GPU and returns a nan gradient on the CPU, and
    the branch had no test at this shape because the roster above ran at
    24 by 32 and Bures was not on it. Both are fixed here.

    Above width 0.5 the paired features have more channels than the batch
    has rows, so this is the usual case rather than an unlucky one.
    """
    loss_fn = FeatureBuresLoss(align='prefix')
    torch.manual_seed(1995)
    student = torch.randn(256, 512, requires_grad=True)
    teacher = torch.randn(256, 512) * 1.3 + 0.2
    value = loss_fn(student, teacher)
    value.backward()
    check('bures runs where the covariance is singular',
          torch.isfinite(value).item(), 'value {}'.format(value.item()))
    check('and its gradient is finite there',
          bool(torch.isfinite(student.grad).all()),
          'max {:.4g}'.format(student.grad.abs().max().item()))
    check('and a cloud against itself still reads zero',
          loss_fn(teacher.clone(), teacher).item() == 0.0,
          '{:.3e}'.format(loss_fn(teacher.clone(), teacher).item()))

    # Where eigh can be trusted - more rows than channels - the Newton
    # iteration has an exact answer to be checked against. The ridge
    # costs about a tenth of a per cent; anything larger would be buying
    # stability with the number itself.
    left = torch.randn(512, 128)
    right = torch.randn(512, 128) * 1.3 + 0.2
    got = loss_fn(left, right).item()
    # the loss divides both clouds by one shared spread first, so the
    # reference has to as well or it is measuring a different quantity
    scale = _feature_scale(left, right)
    left, right = left / scale, right / scale
    gap = (left.mean(dim=0) - right.mean(dim=0)).pow(2).sum()
    centred_l = left - left.mean(dim=0, keepdim=True)
    centred_r = right - right.mean(dim=0, keepdim=True)
    cov_l = centred_l.t().mm(centred_l).double() / 511.0
    cov_r = centred_r.t().mm(centred_r).double() / 511.0

    def exact_sqrt(matrix):
        values, vectors = torch.linalg.eigh(matrix)
        return (vectors * values.clamp_min(0.0).sqrt().unsqueeze(0)).mm(
            vectors.t())

    root = exact_sqrt(cov_l)
    cross = exact_sqrt(root.mm(cov_r).mm(root))
    exact = (gap.double() + cov_l.trace() + cov_r.trace()
             - 2.0 * cross.trace()).item()
    check('bures tracks the eigendecomposition where that one works',
          abs(got - exact) < 0.005 * abs(exact),
          '{:.4f} against {:.4f}, off by {:+.4f}'.format(
              got, exact, got - exact))


def test_the_three_new_axes_leave_k_exactly_where_it_was():
    """K has to be bit identical, not merely close

    Three settings were threaded through train.py for AS through AW, and
    all three sit on the path K already runs. The branch harness cannot
    catch a regression here because it draws its widths unseeded, so the
    same branch gives a different number twice in a row. These check the
    arithmetic directly.
    """
    generator = torch.Generator().manual_seed(44)
    logits = torch.randn(32, 100, generator=generator)
    teacher = torch.softmax(
        torch.randn(32, 100, generator=generator) * 3.0, dim=1)
    soft = CrossEntropyLossSoft()

    # kd_weighting 'none' is the published reduction, spelt differently
    was = torch.mean(soft(logits, teacher)).item()
    now = torch.mean(soft(logits, teacher).view(-1)).item()
    check('kd_weighting none is the old reduction exactly', was == now,
          '{:.8f} against {:.8f}'.format(was, now))

    # both weightings must leave the term's scale alone, or a branch
    # that changes attention is confounded with one that changes weight
    entropy = -(teacher * teacher.clamp_min(1e-12).log()).sum(1)
    for name, raw in (('entropy', entropy),
                      ('confidence', entropy.max() - entropy)):
        weight = raw / raw.mean().clamp_min(1e-12)
        check('{:10} weights average to one'.format(name),
              abs(weight.mean().item() - 1.0) < 1e-5,
              '{:.6f}'.format(weight.mean().item()))

    # and they have to point opposite ways, which is the whole reason
    # AV is worth a session
    sharp = entropy.argmin()
    blunt = entropy.argmax()
    ent_w = entropy / entropy.mean()
    con_w = (entropy.max() - entropy) / (entropy.max() - entropy).mean()
    check('entropy backs the sample the teacher is least sure of',
          ent_w[blunt] > ent_w[sharp],
          '{:.3f} against {:.3f}'.format(ent_w[blunt], ent_w[sharp]))
    check('confidence backs the opposite one',
          con_w[sharp] > con_w[blunt],
          '{:.3f} against {:.3f}'.format(con_w[sharp], con_w[blunt]))


def test_the_chain_does_not_move_which_widths_get_coupled():
    """AW runs the loop widest first, and horizontal_pairs reads order

    horizontal_pairs is positional: it documents 'narrowest first' and
    takes everything after index 0 as the middles. Running the sandwich
    widest first to pass the teaching baton reverses that, and would
    quietly pair the narrowest with a middle instead of the two middles
    with each other. train.py sorts back before pairing; this is what
    says it still does.
    """
    drawn = [0.25, 0.61, 0.43]          # as the loop collects them now
    reversed_loop = [0.61, 0.43, 0.25]  # as teacher_chain collects them

    restored = sorted(reversed_loop)
    chosen = horizontal_pairs(drawn)
    after = horizontal_pairs(restored)
    check('the pairing is one pair either way',
          len(chosen) == len(after) == 1,
          '{} and {}'.format(chosen, after))
    picked = {drawn[i] for i in chosen[0]}
    picked_after = {restored[i] for i in after[0]}
    check('and it is the two middle widths both times',
          picked == picked_after == {0.61, 0.43},
          '{} against {}'.format(sorted(picked), sorted(picked_after)))
    check('the narrowest is in neither',
          0.25 not in picked and 0.25 not in picked_after)


def test_a_weighted_ground_cost_keeps_what_it_is_supposed_to():
    """AX charges the channels unequally, and three things must survive

    The weight is a diagonal metric folded into the distance, so it can
    break the debiasing: if the cross term and the two self terms do not
    see the same metric, transport from a cloud to itself stops reading
    zero and a width already matching its partner is charged anyway.

    The reversal is the other risk. AV reversed a weighting by
    subtracting it from a batch maximum that drifts toward zero, and the
    run collapsed with no mechanism anyone could name. This one permutes
    the weights instead, so the reversed set is the original set and
    cannot leave the range it started in.
    """
    torch.manual_seed(1995)
    left = torch.randn(24, 16)
    right = torch.randn(24, 16)
    # one channel carries nothing: constant across the batch
    left[:, 3] = 2.0
    right[:, 3] = 2.0

    plain = channel_weights(left, right, 'none')
    check('none means no weighting at all', plain is None)

    weight = channel_weights(left, right, 'variance')
    check('the weights average to one',
          abs(float(weight.mean()) - 1.0) < 1e-5,
          'mean {:.6f}'.format(float(weight.mean())))
    check('a channel that never varies is charged almost nothing',
          float(weight[3]) < 1e-3,
          'weight {:.2e}'.format(float(weight[3])))

    flipped = channel_weights(left, right, 'inverse')
    check('the reversal is a permutation of the same weights',
          torch.allclose(flipped.sort().values, weight.sort().values,
                         atol=1e-5))
    check('and it puts the largest weight where the smallest was',
          int(flipped.argmax()) == int(weight.argmin()),
          '{} against {}'.format(int(flipped.argmax()),
                                 int(weight.argmin())))

    for mode in ('none', 'variance', 'inverse'):
        loss = FeatureWassersteinLoss(eps=0.2, n_iters=100, align='prefix',
                                      debiased=True, weighting=mode)
        same = float(loss(left, left))
        check('{}: a cloud against itself still reads zero'.format(mode),
              abs(same) < 1e-4, '{:.2e}'.format(same))

    unweighted = FeatureWassersteinLoss(
        eps=0.2, n_iters=100, align='prefix', debiased=True)
    was = float(unweighted(left, right))
    now = float(FeatureWassersteinLoss(
        eps=0.2, n_iters=100, align='prefix', debiased=True,
        weighting='none')(left, right))
    check('and leaving the flag off reproduces K exactly',
          abs(was - now) < 1e-9, '{:.6f} against {:.6f}'.format(was, now))


def test_the_taylor_estimate_warms_up_before_it_is_trusted():
    """AZ weights by a score it cannot compute from the batch in hand

    The Taylor score needs the gradient of a loss that contains this
    term, so the value for this step does not exist while this step is
    being built. The loss keeps a running estimate from the steps
    already taken and uses that, detached, which makes it a constant to
    the current graph rather than a circular reference.

    Two things have to hold. Before the estimate exists the branch has
    to be K exactly, not K with an arbitrary weight, or the first fifty
    steps train something nobody chose. And the estimate has to end up
    tracking the gradient rather than the activation, or it is the
    activation branch under a different name.
    """
    torch.manual_seed(1995)
    plain = FeatureWassersteinLoss(eps=0.2, n_iters=100, align='prefix',
                                   debiased=True)
    late = FeatureWassersteinLoss(eps=0.2, n_iters=100, align='prefix',
                                  debiased=True, weighting='taylor')
    late.warmup = 4

    student = torch.randn(16, 8, requires_grad=True)
    teacher = torch.randn(16, 8)
    check('before warmup it is K to the last digit',
          abs(float(plain(student, teacher))
              - float(late(student, teacher))) < 1e-9)
    check('and the estimate is withheld rather than guessed',
          late.estimate() is None)

    # drive the hook: one channel is handed no gradient at all
    for _ in range(8):
        f = torch.randn(16, 8, requires_grad=True)
        value = late(f, teacher)
        mask = torch.ones(8)
        mask[5] = 0.0
        (value + (f * mask).sum()).backward()

    got = late.estimate()
    check('after warmup there is an estimate', got is not None)
    if got is not None:
        weight = channel_weights(student.detach(), teacher, 'taylor', got)
        check('the weights average to one',
              abs(float(weight.mean()) - 1.0) < 1e-5,
              'mean {:.6f}'.format(float(weight.mean())))
        check('the channel nothing pushed on is charged least',
              int(got.argmin()) == 5,
              'argmin {} of {}'.format(int(got.argmin()),
                                       [round(float(v), 3) for v in got]))
        # the check the smoke run cannot make for itself. Warmup is 50
        # hook firings by default and a smoke run is six steps, so
        # without this the weighted path could reach Kaggle having never
        # once been evaluated, reading exactly like K the whole way.
        was = float(plain(student, teacher))
        now = float(late(student, teacher))
        check('and past warmup the loss is no longer K',
              abs(was - now) > 1e-6,
              '{:.6f} against {:.6f}'.format(was, now))


def test_activation_and_variance_are_not_the_same_criterion():
    """a channel can be large and constant, and that is the difference"""
    torch.manual_seed(1995)
    left = torch.randn(32, 6)
    right = torch.randn(32, 6)
    left[:, 2] = 5.0          # big, says nothing about which sample
    right[:, 2] = 5.0
    by_var = channel_weights(left, right, 'variance')
    by_act = channel_weights(left, right, 'activation')
    check('variance charges the constant channel nothing',
          float(by_var[2]) < 1e-3, '{:.2e}'.format(float(by_var[2])))
    check('activation charges it the most',
          int(by_act.argmax()) == 2,
          'argmax {}'.format(int(by_act.argmax())))


def test_the_warm_up_holds_the_narrow_end_back_and_then_returns_it():
    """BD to BG train only the widest width for the first epochs

    Neither the branch suite nor a two step smoke run reaches the epoch
    loop, so nothing else here sees this. What it has to get right:
    during the warm-up the narrow end is absent entirely, and the step
    after it the sandwich is whole again. Coming back one width short
    would quietly train a different method for ninety epochs.
    """
    was_warm = getattr(FLAGS, 'narrow_start_epoch', None)
    was_n = getattr(FLAGS, 'num_sample_training', None)
    FLAGS.narrow_start_epoch = 10
    FLAGS.num_sample_training = 4
    try:
        for epoch in (0, 5, 9):
            widths = training_widths(epoch, 0.25, 1.0)
            check('epoch {}: the widest alone'.format(epoch),
                  widths == [1.0], str(widths))
        for epoch in (10, 11, 99):
            widths = training_widths(epoch, 0.25, 1.0)
            check('epoch {}: the sandwich is whole'.format(epoch),
                  len(widths) == 4 and widths[0] == 1.0
                  and widths[1] == 0.25, str(widths))
            check('epoch {}: and the free draws are inside'.format(epoch),
                  all(0.25 <= w <= 1.0 for w in widths[2:]))
        FLAGS.narrow_start_epoch = 0
        check('with no warm-up epoch zero is already whole',
              len(training_widths(0, 0.25, 1.0)) == 4)
    finally:
        if was_warm is None:
            del FLAGS.narrow_start_epoch
        else:
            FLAGS.narrow_start_epoch = was_warm
        if was_n is None:
            del FLAGS.num_sample_training
        else:
            FLAGS.num_sample_training = was_n


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
    test_classwise_is_coarser_rather_than_class_aware()
    test_horizontal_pairs_keeps_the_old_default()
    test_unbalanced_stays_monotone_at_the_shipped_tau()
    test_cosine_ground_behaves_like_a_cost()
    test_gromov_is_invariant_to_what_it_claims()
    test_bures_survives_more_channels_than_samples()
    test_the_three_new_axes_leave_k_exactly_where_it_was()
    test_the_chain_does_not_move_which_widths_get_coupled()
    test_a_weighted_ground_cost_keeps_what_it_is_supposed_to()
    test_the_taylor_estimate_warms_up_before_it_is_trusted()
    test_activation_and_variance_are_not_the_same_criterion()
    test_the_warm_up_holds_the_narrow_end_back_and_then_returns_it()
    print()
    if FAILURES:
        print('{} failed: {}'.format(len(FAILURES), ', '.join(FAILURES)))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
