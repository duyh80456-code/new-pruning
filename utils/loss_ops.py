import math
import random

import torch

from utils.config import FLAGS


class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss


class CrossEntropyLossSmooth(torch.nn.modules.loss._Loss):
    """ label smooth """
    def forward(self, output, target):
        eps = FLAGS.label_smoothing
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss


def get_classifier_weight(model):
    """last linear layer of a (possibly wrapped) model

    In a universally slimmable net the classifier is USLinear(us=[True,
    False]): the number of output rows is fixed at num_classes and only the
    input side is sliced. Teacher and student therefore index the same rows
    of this matrix, which is what makes the class cost matrix below intrinsic
    to the model rather than imposed from outside.
    """
    weight = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            weight = m.weight
    if weight is None:
        raise ValueError('no linear layer found to build a cost matrix from')
    return weight


def class_cost_matrix(weight, normalize=True):
    """d(i, j) = ||w_i - w_j||_2 between classifier rows

    Detached on purpose: the cost matrix is the metric the transport plan is
    measured against, not something the loss should reshape. Normalizing by
    the mean keeps the loss scale stable as the weights grow during training.
    """
    w = weight.detach()
    if w.dim() != 2:
        w = w.view(w.size(0), -1)
    cost = torch.cdist(w, w, p=2)
    if normalize:
        cost = cost / cost.mean().clamp_min(1e-12)
    return cost


def identity_cost_matrix(num_classes, device=None, dtype=None):
    """every pair of classes equally far apart

    The control the measured spread asks for. With this cost, transport has
    no geometry left to use and the divergence collapses to a scaled total
    variation. If a run with it matches a run with the classifier cost,
    then the classifier cost was contributing nothing and the case for
    Wasserstein at the logit level is closed.
    """
    cost = 1.0 - torch.eye(num_classes, device=device, dtype=dtype)
    return cost / cost.mean().clamp_min(1e-12)


class ConfusionEmbedding(object):
    """what the teacher confuses each class with, accumulated online

    The classifier rows turned out to be nearly equidistant: on a trained
    supernet the farthest class is only 1.32 times as far as the nearest,
    against 3.79 for an embedding with real structure. Rows of a linear
    layer in 512 dimensions are not where the semantics live.

    Confusion is. Two classes are close here when the teacher spreads the
    same mass over the same other classes, which is measured rather than
    assumed, and is sparse enough not to concentrate.
    """
    def __init__(self, num_classes, momentum=0.01, warmup=50):
        self.num_classes = num_classes
        self.momentum = momentum
        self.warmup = warmup
        self.updates = 0
        self.profile = None

    def update(self, teacher_prob, target):
        with torch.no_grad():
            one_hot = torch.zeros_like(teacher_prob)
            one_hot.scatter_(1, target.view(-1, 1), 1.0)
            counts = one_hot.sum(dim=0)
            batch = one_hot.t().mm(teacher_prob)
            seen = counts > 0
            batch[seen] = batch[seen] / counts[seen].unsqueeze(1)
            if self.profile is None:
                self.profile = torch.zeros_like(batch)
                self.profile[seen] = batch[seen]
            else:
                self.profile[seen] = (
                    (1.0 - self.momentum) * self.profile[seen]
                    + self.momentum * batch[seen])
            self.updates += 1

    def ready(self):
        return self.profile is not None and self.updates >= self.warmup

    def embedding(self):
        rows = self.profile.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return self.profile / rows


def _uniform_log_marginal(n, reference):
    return torch.full(
        (1, n), -math.log(n),
        device=reference.device, dtype=reference.dtype)


def sinkhorn_plan(cost, eps=0.2, n_iters=100):
    """optimal coupling for uniform marginals, detached

    Used where the gradient has to come through the cost rather than the
    marginals: at the feature level the two clouds carry equal mass by
    construction, so nothing about the marginals is learnable and all the
    signal is in the pairwise distances. With the plan held fixed, the
    envelope theorem gives the same gradient as differentiating the solve.
    """
    with torch.no_grad():
        log_p = _uniform_log_marginal(cost.size(0), cost)
        log_q = _uniform_log_marginal(cost.size(1), cost)
        f, g = _sinkhorn_potentials(log_p, log_q, cost, eps, n_iters)
        plan = torch.exp(
            (f.unsqueeze(2) + g.unsqueeze(1) - cost.unsqueeze(0)) / eps)
    return plan.squeeze(0)


def unbalanced_sinkhorn_plan(cost, eps=0.2, n_iters=100, tau=1.0):
    """coupling that may fall short of its marginals, detached

    The balanced solve enforces the marginals exactly. Here they are
    penalized instead, by a KL term of strength tau, which turns the hard
    constraint into a price: mass that is expensive to move can be left
    where it is. The scaling steps are the balanced ones shrunk by
    tau / (tau + eps), so tau large recovers the balanced plan and tau
    small lets most of the mass stay put.

    Damped and simultaneous for the same reason the balanced version is:
    swapping the two sides has to swap the two potentials exactly, since
    this is used between two widths that stand in no order.
    """
    with torch.no_grad():
        log_p = _uniform_log_marginal(cost.size(0), cost)
        log_q = _uniform_log_marginal(cost.size(1), cost)
        cost_eps = cost.unsqueeze(0) / eps
        damping = tau / (tau + eps)
        f = torch.zeros_like(log_p)
        g = torch.zeros_like(log_q)
        for _ in range(n_iters):
            f_next = damping * eps * (log_p - torch.logsumexp(
                g.unsqueeze(1) / eps - cost_eps, dim=2))
            g_next = damping * eps * (log_q - torch.logsumexp(
                f.unsqueeze(2) / eps - cost_eps, dim=1))
            f = 0.5 * (f + f_next)
            g = 0.5 * (g + g_next)
        plan = torch.exp(
            (f.unsqueeze(2) + g.unsqueeze(1) - cost.unsqueeze(0)) / eps)
    return plan.squeeze(0)


def align_features(student, teacher, align='prefix'):
    """put two widths into a space where they can be compared

    They never share a dimension, which is the whole difficulty. Two ways,
    and they are not the same claim:

    prefix - a narrow subnet uses the leading channels of the wide one, so
    the wide features truncated to the narrow width are the same
    coordinates. This is the feature-level version of the argument that
    made the classifier cost intrinsic, and it exists only because the
    widths are nested.

    gram - compare which samples each width considers similar, not where it
    places them. Dimension free, and the standard way relational
    distillation sidesteps this, so it is what the literature compares
    against.
    """
    if align == 'prefix':
        width = min(student.size(1), teacher.size(1))
        return student[:, :width], teacher[:, :width]
    if align == 'gram':
        left = torch.nn.functional.normalize(student, dim=1)
        right = torch.nn.functional.normalize(teacher, dim=1)
        return left.mm(left.t()), right.mm(right.t())
    raise ValueError('unknown feature_align {}'.format(align))


def _feature_scale(left, right):
    """a yardstick that does not move when the clouds do

    Dividing the cost by its own mean, the obvious thing, is wrong here.
    At the logit level the cost is a detached metric and normalizing only
    fixes the loss scale. Here the cost *is* the objective: a student cloud
    sliding toward the teacher shrinks every pairwise distance by the same
    factor, so a self-normalized cost is invariant to exactly the movement
    it is supposed to reward, and the gradient vanishes.

    The spread within each cloud is the scale instead. Detached, so it
    cannot be gamed, and symmetric in the two arguments, which the
    horizontal use needs.
    """
    with torch.no_grad():
        spread = 0.5 * (torch.cdist(left, left).mean()
                        + torch.cdist(right, right).mean())
    return spread.clamp_min(1e-12)


def channel_weights(left, right, mode, estimate=None):
    """how much each channel counts toward the distance

    The cost charges every channel the same, and the channels are not
    the same. Measured on K at the pooled tap, at the middle widths the
    feature term actually runs between, the top half of the channels
    carries 81 to 87 per cent of the Taylor score and only 44 to 66 per
    cent of them are effective at all.

    variance is the free stand-in for that score. A channel that does
    not vary across the batch says nothing about which sample is which,
    so distance along it is distance spent on nothing. It needs no
    gradient, which matters because the Taylor score needs the gradient
    of a loss that contains this term, and weighting a term by its own
    gradient is circular. Measured against Taylor it ranks the channels
    the same way to 0.78 at width 0.6, 0.84 at 0.5 and 0.83 at 0.4.

    inverse is the control, and it is built from ranks rather than from
    1/v. Branch AV reversed a weighting by subtracting it from a batch
    maximum that drifts to zero, and what came back was a collapse
    nobody could diagnose. A reversal that permutes the same weights
    cannot leave the range they already occupy.

    activation is the other free one: the mean absolute value of the
    channel. It differs from variance in counting a channel that is
    large and constant, which carries no information about which sample
    is which, so the two are worth telling apart rather than assuming.

    taylor is the score the other two stand in for, and it cannot be
    computed from this batch. It needs the gradient of a loss that
    contains this term, so using this step's value would weight a term
    by its own gradient. The estimate passed in is an average over the
    steps already taken, detached, and is therefore a constant to the
    graph being built now. Not a one step lag: at momentum 0.05 it is an
    average over roughly the last twenty firings, so what it assumes is
    that the ordering drifts slowly, not that it is fixed.

    Normalised to mean one every way, so this moves what the cost
    attends to and not how large it is.
    """
    if mode in (None, 'none'):
        return None
    if mode == 'taylor':
        if estimate is None:
            return None            # still warming up: uniform, which is K
        weight = estimate[:left.size(1)].clone()
    elif mode == 'activation':
        weight = 0.5 * (left.abs().mean(dim=0) + right.abs().mean(dim=0))
    else:
        spread = 0.5 * (left.var(dim=0) + right.var(dim=0))
        if mode == 'variance':
            weight = spread
        elif mode == 'inverse':
            order = spread.argsort()
            weight = torch.empty_like(spread)
            weight[order] = spread.sort(descending=True).values
        else:
            raise ValueError('unknown feature_weighting {}'.format(mode))
    return weight / weight.mean().clamp_min(1e-12)


def ground_cost(left, right, ground='euclidean', scale=None, weight=None):
    """the distance transport is charged per unit of mass moved

    euclidean is the distance the clouds live in, divided by their spread
    so one eps serves every width. cosine asks only about direction, which
    drops the magnitude a wider subnet can put into a channel and is
    already bounded, so it needs no scaling. Which of the two is right is
    not obvious and is why there is a branch for it.

    weight, when given, scales each channel before the distance is taken,
    which makes it a diagonal metric rather than a plain euclidean one.
    """
    if weight is not None:
        root = weight.clamp_min(0.0).sqrt()
        left, right = left * root, right * root
    if ground == 'euclidean':
        if scale is None:
            scale = _feature_scale(left, right)
        return torch.cdist(left, right) / scale
    if ground == 'cosine':
        left = torch.nn.functional.normalize(left, dim=1)
        right = torch.nn.functional.normalize(right, dim=1)
        return 1.0 - left.mm(right.t())
    raise ValueError('unknown feature_ground {}'.format(ground))


def feature_cost(student, teacher, align='prefix', ground='euclidean'):
    """scaled pairwise distance between two batches of features"""
    left, right = align_features(student, teacher, align)
    return ground_cost(left, right, ground)


class FeatureWassersteinLoss(torch.nn.modules.loss._Loss):
    """transport between the feature clouds of two widths

    One number for the batch, not one per sample: the clouds are matched as
    distributions, so a sample may be explained by a different sample of
    the other width. This is the tier RQ1 measured, where the geometry
    demonstrably carries information, rather than the logit tier where the
    cost matrix turned out flat.

    Debiased for the same reason the logit version is. Entropic transport
    between a cloud and itself is not zero, so without the correction a
    width already matching the teacher is still charged, and the horizontal
    use is not minimized at agreement.
    """
    def __init__(self, eps=0.2, n_iters=100, align='prefix', debiased=True,
                 ground='euclidean', weighting='none', reduction='mean'):
        super(FeatureWassersteinLoss, self).__init__(reduction=reduction)
        self.eps = eps
        self.n_iters = n_iters
        self.align = align
        self.debiased = debiased
        self.ground = ground
        self.weighting = weighting
        # the Taylor estimate, accumulated across steps. The loss object
        # is built once outside the loop, the same way set_target relies
        # on, so this survives between batches.
        self.momentum = getattr(FLAGS, 'taylor_momentum', 0.05)
        # counted in hook firings, not optimizer steps: the term runs
        # once per student width, so 50 is roughly 17 steps of a
        # four-width sandwich. The smoke config lowers it, or neither
        # the suite nor the smoke run would ever leave warmup and the
        # weighted path would reach Kaggle untried.
        self.warmup = getattr(FLAGS, 'taylor_warmup', 50)
        self._taylor = None
        self._seen = 0

    def observe(self, feature):
        """accumulate |f * dL/df| per channel when the backward runs

        A hook on the feature rather than a second backward: the
        gradient is already computed for the update, and reading it
        costs one reduction.
        """
        if not (self.weighting == 'taylor' and feature.requires_grad):
            return

        def absorb(grad):
            with torch.no_grad():
                score = (feature.detach() * grad).abs().sum(dim=0)
                width = score.size(0)
                if self._taylor is None:
                    self._taylor = torch.zeros(
                        width, dtype=score.dtype, device=score.device)
                if self._taylor.size(0) < width:
                    grown = torch.zeros(
                        width, dtype=self._taylor.dtype,
                        device=self._taylor.device)
                    grown[:self._taylor.size(0)] = self._taylor
                    self._taylor = grown
                head = self._taylor[:width]
                self._taylor[:width] = (
                    (1.0 - self.momentum) * head + self.momentum * score)
                self._seen += 1

        feature.register_hook(absorb)

    def estimate(self):
        if self._taylor is None or self._seen < self.warmup:
            return None
        return self._taylor

    def _transport(self, left, right, scale, weight=None):
        cost = ground_cost(left, right, self.ground, scale, weight)
        plan = sinkhorn_plan(cost.detach(), self.eps, self.n_iters)
        return (plan * cost).sum()

    def forward(self, student, teacher):
        left, right = align_features(student, teacher, self.align)
        scale = _feature_scale(left, right)
        # one metric for all three terms. A weight recomputed inside each
        # would make the self-transports use a different geometry from the
        # cross term, and the debiasing would stop landing on zero.
        self.observe(left)
        weight = channel_weights(left.detach(), right.detach(),
                                 self.weighting, self.estimate())
        value = self._transport(left, right, scale, weight)
        if self.debiased:
            value = value - 0.5 * self._transport(left, left, scale, weight)
            value = value - 0.5 * self._transport(right, right, scale, weight)
        return value


class FeatureUnbalancedWassersteinLoss(FeatureWassersteinLoss):
    """transport that may leave mass behind, for a price

    Balanced transport insists every sample of one width is explained by
    the samples of the other. Two widths of one supernet do not have to
    agree that completely: a narrow subnet can simply fail on part of the
    batch, and forcing its failures to be matched somewhere spends the
    term on the samples it can do least about.

    tau is what leaving mass unmatched costs. Large tau is the balanced
    problem back again, small tau matches only the easy part of the batch,
    so this is a knob on how much of the disagreement the term is asked to
    carry.
    """
    def __init__(self, tau=1.0, **kwargs):
        super(FeatureUnbalancedWassersteinLoss, self).__init__(**kwargs)
        self.tau = tau

    def _transport(self, left, right, scale, weight=None):
        cost = ground_cost(left, right, self.ground, scale, weight)
        plan = unbalanced_sinkhorn_plan(
            cost.detach(), self.eps, self.n_iters, self.tau)
        return (plan * cost).sum()


class FeatureGromovLoss(torch.nn.modules.loss._Loss):
    """transport between the two clouds' own distance structures

    Every other feature loss here has to answer a question it should not
    have to: the two widths live in spaces of different dimension, so
    something must say how a coordinate of one corresponds to a coordinate
    of the other. prefix answers by truncation and gram by dropping
    coordinates entirely, and both are assumptions the result then rests
    on.

    Gromov transport does not ask. It compares how far sample i is from
    sample k inside one cloud against how far their partners are inside
    the other, so the two spaces are never put in correspondence at all,
    only their internal geometry. That is the one formulation whose claim
    survives whatever the channel ordering turns out to mean.

    Entropic, following the squared loss factorization of Peyre, Cuturi
    and Solomon 2016: the objective is quadratic in the plan, so each
    outer step freezes the plan to build a linear cost and solves that
    with Sinkhorn. outer_iters is small because the plan stops moving
    quickly and every step costs a full solve.

    Invariant to anything that leaves a cloud's internal distances alone,
    a translation among them. That is the formulation working as defined,
    not a hole in it: what it asks is whether the two widths organize the
    batch the same way, never whether they put it in the same place. A
    term that also wants them in the same place is what prefix alignment
    is for, and the two branches are how one finds out which matters.

    Debiased like the others. Entropic blur makes the objective positive
    even between a cloud and itself, and a horizontal term has to be
    minimized at agreement.
    """
    def __init__(self, eps=0.2, n_iters=50, outer_iters=5, align=None,
                 ground='euclidean', debiased=True, reduction='mean'):
        super(FeatureGromovLoss, self).__init__(reduction=reduction)
        self.eps = eps
        self.n_iters = n_iters
        self.outer_iters = outer_iters
        self.ground = ground
        self.debiased = debiased

    def _within(self, cloud):
        """distances inside one cloud, in that cloud's own space"""
        return ground_cost(cloud, cloud, self.ground,
                           _feature_scale(cloud, cloud))

    def _gromov(self, left, right):
        n, m = left.size(0), right.size(0)
        weight_left = torch.full((n, 1), 1.0 / n, device=left.device,
                                 dtype=left.dtype)
        weight_right = torch.full((m, 1), 1.0 / m, device=right.device,
                                  dtype=right.dtype)
        # the part of the quadratic objective that does not move with the
        # plan, so it is built once
        constant = (left.pow(2).mm(weight_left).expand(n, m)
                    + weight_right.t().mm(right.pow(2).t()).expand(n, m))
        with torch.no_grad():
            plan = weight_left.mm(weight_right.t())
            for _ in range(self.outer_iters):
                linear = constant - 2.0 * left.mm(plan).mm(right.t())
                plan = sinkhorn_plan(linear, self.eps, self.n_iters)
        linear = constant - 2.0 * left.mm(plan).mm(right.t())
        return (plan * linear).sum()

    def forward(self, student, teacher):
        # no align_features: not needing it is the point
        left = self._within(student)
        right = self._within(teacher)
        value = self._gromov(left, right)
        if self.debiased:
            value = value - 0.5 * self._gromov(left, left)
            value = value - 0.5 * self._gromov(right, right)
        return value


class FeatureMSELoss(torch.nn.modules.loss._Loss):
    """squared distance, each sample against its own counterpart

    The control the transport losses have to clear, and the cheapest thing
    anyone would try first. It pairs sample i of one width with sample i of
    the other and nothing else, so it asks whether the freedom to rematch
    samples is what transport buys, or whether pulling the two clouds
    together by any means would have done.

    Scaled by the same spread as the others, so feature_weight means the
    same thing across the axis and the branches stay comparable.
    """
    def __init__(self, align='prefix', reduction='mean'):
        super(FeatureMSELoss, self).__init__(reduction=reduction)
        self.align = align

    def forward(self, student, teacher):
        left, right = align_features(student, teacher, self.align)
        scale = _feature_scale(left, right)
        return ((left - right) / scale).pow(2).sum(dim=1).mean()


class FeatureMMDLoss(torch.nn.modules.loss._Loss):
    """maximum mean discrepancy between the two clouds

    The control that matters most to the claim. Transport was chosen here
    because two feature clouds have no common support and an f-divergence
    needs one. But MMD does not need one either, and it needs no transport
    plan and no fixed point iteration. If MMD matches Sinkhorn then what
    works is matching the clouds as distributions, and optimal transport in
    particular is not load bearing.

    Gaussian kernel on clouds already divided by their spread, so the
    bandwidth is dimensionless and one setting serves every width. The
    biased estimator is exactly zero when the two clouds coincide, so
    unlike entropic transport this needs no debiasing correction.
    """
    def __init__(self, align='prefix', bandwidth=1.0, reduction='mean'):
        super(FeatureMMDLoss, self).__init__(reduction=reduction)
        self.align = align
        self.bandwidth = bandwidth

    def forward(self, student, teacher):
        left, right = align_features(student, teacher, self.align)
        scale = _feature_scale(left, right)
        left, right = left / scale, right / scale
        gamma = 1.0 / (2.0 * self.bandwidth * self.bandwidth)
        within_left = torch.exp(-gamma * torch.cdist(left, left).pow(2))
        within_right = torch.exp(-gamma * torch.cdist(right, right).pow(2))
        across = torch.exp(-gamma * torch.cdist(left, right).pow(2))
        return within_left.mean() + within_right.mean() - 2.0 * across.mean()


class FeatureSlicedWassersteinLoss(torch.nn.modules.loss._Loss):
    """transport along random one dimensional projections

    The same geometry as the entropic version at a fraction of the price: a
    projection and a sort, no fixed point iteration. That is worth knowing
    because the Sinkhorn branch costs 299 minutes against 165 for the
    baseline, close enough to the session limit to constrain what else can
    be run.

    It is also the quantity this project's earlier work was built on, so a
    result here connects to that history rather than starting beside it.

    Exact at equality and symmetric in its two arguments by construction,
    so no debiasing correction either. The projections are redrawn every
    call, which adds gradient noise and is how sliced transport is normally
    used.

    reduce 'max' keeps the worst direction instead of the average, which
    is the max-sliced distance. Averaging dilutes a disagreement confined
    to a few directions among all the ones where the widths already agree,
    and in a 512 channel space most directions are the second kind.

    p is the exponent on the one dimensional gaps. Squared is the usual
    choice and weights the outlying samples heavily; p = 1 does not, and
    on clouds where a narrow subnet fails on part of the batch that is a
    different quantity rather than a softer one.
    """
    def __init__(self, align='prefix', n_projections=128, reduce='mean',
                 p=2.0, reduction='mean'):
        super(FeatureSlicedWassersteinLoss, self).__init__(
            reduction=reduction)
        self.align = align
        self.n_projections = n_projections
        self.reduce = reduce
        self.p = p

    def forward(self, student, teacher):
        left, right = align_features(student, teacher, self.align)
        if left.size(0) != right.size(0):
            raise ValueError(
                'sliced transport sorts the two sides against each other, '
                'so they must hold the same number of samples')
        scale = _feature_scale(left, right)
        directions = torch.randn(
            left.size(1), self.n_projections,
            device=left.device, dtype=left.dtype)
        directions = directions / directions.norm(dim=0, keepdim=True)
        projected_left = (left / scale).mm(directions).sort(dim=0)[0]
        projected_right = (right / scale).mm(directions).sort(dim=0)[0]
        # Times the dimension, which is not cosmetic. Projecting onto a
        # random unit direction shrinks a squared distance by exactly the
        # dimension in expectation, and the feature dimension here moves
        # with the width: 128 channels at 0.25 against 512 at 1.00. Without
        # the factor the same feature_weight would apply this term four
        # times more weakly at the wide end than the narrow one, which is a
        # width schedule nobody chose. With it the value sits on the scale
        # of the squared distance the full transport reports.
        gaps = (projected_left - projected_right).abs().pow(self.p)
        if self.reduce == 'max':
            # the worst direction, averaged over the samples along it
            return left.size(1) * gaps.mean(dim=0).max()
        if self.reduce == 'mean':
            return left.size(1) * gaps.mean()
        raise ValueError('unknown sliced_reduce {}'.format(self.reduce))


class FeatureChannelLoss(torch.nn.modules.loss._Loss):
    """one dimensional transport along each shared channel

    Sliced transport projects onto random directions because in general no
    direction means anything. Here they do. A narrow subnet uses the
    leading channels of the wide one, the same channels with the same
    weights, so channel j is one coordinate both widths possess and the
    projection onto it is the question of whether they use it the same
    way.

    Exact, not approximated: transport in one dimension is a sort and a
    difference, with no plan to solve and no blur to choose. Cheaper than
    Sinkhorn and cheaper than sliced, because the directions do not have
    to be drawn or multiplied through.

    What this cannot see is any disagreement that lives between channels,
    which is what the full transport is for. The pair says whether the
    coordinates carry it.
    """
    def __init__(self, align='prefix', p=2.0, reduction='mean'):
        super(FeatureChannelLoss, self).__init__(reduction=reduction)
        self.align = align
        self.p = p

    def forward(self, student, teacher):
        left, right = align_features(student, teacher, self.align)
        scale = _feature_scale(left, right)
        left = (left / scale).sort(dim=0)[0]
        right = (right / scale).sort(dim=0)[0]
        return (left - right).abs().pow(self.p).mean() * left.size(1)


class FeatureBuresLoss(torch.nn.modules.loss._Loss):
    """the closed form transport between two Gaussians fitted to the clouds

        W2^2 = ||m1 - m2||^2 + tr(C1 + C2 - 2 (C1^.5 C2 C1^.5)^.5)

    No plan, no iteration, no blur. Where the entropic version estimates
    transport from 256 points and pays a fixed point solve for it, this
    assumes the clouds are Gaussian and then is exact, which trades one
    approximation for a different one rather than for none.

    It is the only loss here that sees the shape of a cloud. Transport
    over a finite sample matches where the points are; the covariance term
    matches how they are spread and in which directions, which is the part
    a batch of 256 in 512 dimensions estimates worst.

    full needs two symmetric eigendecompositions per call, so a diagonal
    mode keeps the means and the per channel variances and drops the cross
    terms. That is nearly free and still matches more than a plain
    distance does.
    """
    def __init__(self, align='prefix', diagonal=False, reduction='mean'):
        super(FeatureBuresLoss, self).__init__(reduction=reduction)
        self.align = align
        self.diagonal = diagonal

    # A batch of 256 in 512 channels gives a covariance of rank at most
    # 255, so over 250 of its eigenvalues are exactly zero. eigh refuses
    # that outright on the GPU - 'too many repeated eigenvalues', which
    # killed the first AK run three minutes in - and where it survives,
    # on the CPU, it returns a nan gradient instead, because tr(sqrt(A))
    # has gradient A^-0.5 / 2 and that is unbounded in the null space.
    # The crash was the visible half.
    #
    # Coupled Newton-Schulz replaces it: no eigendecomposition, so no
    # repeated roots to trip over. It needs every eigenvalue of the
    # normalised matrix inside the unit ball around one, and a zero sits
    # exactly on the boundary, converging only marginally until float
    # error carries it off. Two knobs were measured against each other:
    #
    #   ridge     lifts the zeros off the boundary, and biases the value
    #   precision decides how fast the error in those directions grows
    #
    # In float32 they could not both be satisfied. Stability wanted a
    # ridge of at least 1e-5, below which the iteration is nan by thirty
    # steps; signal wanted at most 1e-7, above which clamp_min flattens
    # the whole term at a gap of 0.03. float64 dissolves the conflict,
    # because it is the iteration and not the trace that needs the
    # precision: at 1e-9 the value settles by twenty steps and is
    # unchanged at eighty, and near agreement it reproduces the unridged
    # number to three digits.
    #
    # It costs about twice the float32 time for a 512 by 512 pair, once
    # per step.
    RIDGE = 1e-9
    ITERATIONS = 20

    @classmethod
    def _sqrt_psd(cls, matrix):
        eye = torch.eye(matrix.size(0), dtype=matrix.dtype,
                        device=matrix.device)
        matrix = matrix + (cls.RIDGE * matrix.diagonal().mean().detach()) * eye
        norm = matrix.norm(p='fro')
        root, inverse = matrix / norm, eye
        for _ in range(cls.ITERATIONS):
            step = 0.5 * (3.0 * eye - inverse.mm(root))
            root = root.mm(step)
            inverse = step.mm(inverse)
        return root * norm.sqrt()

    def forward(self, student, teacher):
        left, right = align_features(student, teacher, self.align)
        scale = _feature_scale(left, right)
        left, right = left / scale, right / scale
        mean_gap = (left.mean(dim=0) - right.mean(dim=0)).pow(2).sum()
        if self.diagonal:
            # the Bures distance between diagonal Gaussians is the squared
            # difference of the standard deviations, channel by channel
            spread_gap = (left.std(dim=0) - right.std(dim=0)).pow(2).sum()
            return mean_gap + spread_gap
        # float64 from here down: see the note on RIDGE. The traces
        # below are of order the total variance and cancel to nothing at
        # agreement, so the arithmetic that produces them is the part
        # that has to be exact.
        left = (left - left.mean(dim=0, keepdim=True)).double()
        right = (right - right.mean(dim=0, keepdim=True)).double()
        mean_gap = mean_gap.double()
        count = max(left.size(0) - 1, 1)
        cov_left = left.t().mm(left) / count
        cov_right = right.t().mm(right) / count
        root = self._sqrt_psd(cov_left)
        cross = self._sqrt_psd(root.mm(cov_right).mm(root))
        # the trace identity is exact and the eigendecompositions are not,
        # so two identical clouds come out a shade below zero. Left alone
        # that is a term rewarding a width for matching its partner and
        # then pushing past it.
        return (mean_gap + cov_left.trace() + cov_right.trace()
                - 2.0 * cross.trace()).clamp_min(0.0).float()


class ClasswiseFeatureLoss(torch.nn.modules.loss._Loss):
    """transport between the two widths' class positions, not their samples

    Everywhere else the batch is one unlabelled cloud, so a narrow width's
    picture of a cat is free to be explained by the teacher's picture of a
    dog if the two happen to land near each other. Nothing in the
    objective says that is wrong.

    Matching inside each class is the obvious repair and the arithmetic
    rules it out: 256 images over 100 classes is about two and a half per
    class, and transport between two points is a fixed pairing, which is
    branch Q by a longer route.

    So the clouds are the class positions themselves. Each class present
    in the batch contributes its mean, and transport runs between those,
    asking whether the two widths arrange the classes the same way rather
    than whether they place each image the same way. That is the geometry
    the logit tier went looking for in the classifier rows and did not
    find, measured instead where the project has evidence geometry exists.

    The labels are set once a step, the way the class cost matrix is, and
    the loss is unusable without them.
    """
    def __init__(self, inner, min_count=1, reduction='mean'):
        super(ClasswiseFeatureLoss, self).__init__(reduction=reduction)
        self.inner = inner
        self.min_count = min_count
        self.target = None

    def set_target(self, target):
        self.target = target

    def _centroids(self, cloud, labels, present):
        return torch.stack([
            cloud[labels == label].mean(dim=0) for label in present])

    def forward(self, student, teacher):
        if self.target is None:
            raise RuntimeError('set_target must be called before forward')
        labels = self.target
        counts = torch.bincount(labels)
        present = (counts >= self.min_count).nonzero().flatten()
        if present.numel() < 2:
            return student[0].sum() * 0.0 if isinstance(
                student, (tuple, list)) else student.sum() * 0.0
        if not isinstance(student, (tuple, list)):
            student, teacher = (student,), (teacher,)
        total = 0.0
        for left, right in zip(student, teacher):
            total = total + self.inner(
                self._centroids(left, labels, present),
                self._centroids(right, labels, present))
        return total / len(student)


class ClassifierSpreadLoss(torch.nn.modules.loss._Loss):
    """push the classifier rows apart, rather than pull anything together

    Every other term here is a pull: two distributions are compared and
    the gap is charged. This is the opposite, and it exists because of a
    measurement. The class cost matrix on a trained supernet has a spread
    of 1.32 between its nearest and farthest pairs, against 3.79 for an
    embedding with real structure, so transport over it had almost no
    geometry to use and six branches at that tier failed.

    Rather than move to a tier where geometry exists, this tries to put
    geometry where it was missing: a repulsion on the rows of the shared
    classifier, so that classes the network has no reason to separate stop
    sitting on top of each other.

    Negative by construction, since it rewards spread. The pairs already
    far apart are the ones with nothing left to gain, so the penalty is on
    the closest neighbours only, which is what keeps it from simply
    inflating every row.
    """
    def __init__(self, neighbours=5, reduction='mean'):
        super(ClassifierSpreadLoss, self).__init__(reduction=reduction)
        self.neighbours = neighbours

    def forward(self, weight):
        rows = torch.nn.functional.normalize(weight, dim=1)
        distance = torch.cdist(rows, rows)
        distance = distance + torch.eye(
            rows.size(0), device=rows.device, dtype=rows.dtype) * 1e9
        nearest = distance.topk(
            min(self.neighbours, rows.size(0) - 1),
            dim=1, largest=False)[0]
        return -nearest.mean()


class MultiTierFeatureLoss(torch.nn.modules.loss._Loss):
    """the same transport applied at several depths at once

    A supernet can disagree with itself anywhere, not only at the end. The
    final pooled feature is one tap among several; the stage outputs are
    averaged over space so that a single cost works at every depth.

    Deeper tiers are more specific to the task and shallower ones more
    generic, so tier_weights exists to say that they are not
    interchangeable. Equal weights is the honest default, not a claim.
    """
    def __init__(self, eps=0.2, n_iters=100, align='prefix', debiased=True,
                 tier_weights=None, inner=None, reduction='mean'):
        super(MultiTierFeatureLoss, self).__init__(reduction=reduction)
        # inner is what compares one pair of taps. It defaults to entropic
        # transport, which is what every finished feature run used, so a
        # caller that does not name one gets the published behaviour.
        self.inner = inner or FeatureWassersteinLoss(
            eps=eps, n_iters=n_iters, align=align, debiased=debiased)
        self.tier_weights = tier_weights

    def forward(self, student, teacher):
        if not isinstance(student, (tuple, list)):
            student, teacher = (student,), (teacher,)
        if len(student) != len(teacher):
            raise ValueError('a tap is missing on one side')
        weights = self.tier_weights or [1.0] * len(student)
        total = 0.0
        for weight, left, right in zip(weights, student, teacher):
            total = total + weight * self.inner(left, right)
        return total / max(sum(weights), 1e-12)


def sample_width(low, high):
    """where in the width range the sandwich rule spends its free samples

    US-Net draws the middle widths uniformly and says nothing about why.
    On this supernet that spends the budget in the wrong place: A gains
    3.10 points across 0.25 to 0.50 and 1.90 across 0.55 to 1.00, so the
    curve is steepest at the narrow end, while a uniform draw puts two
    thirds of its samples in the flat half.

    Three settings, and the first is what every finished run used:

      uniform  flat in width, as published
      log      flat in log width, which doubles the mass below 0.5
      macs     flat in compute. MACs measured at width^1.965 on this
               model, so this is flat in width squared and leans the
               other way, toward the wide end

    log and macs are a bracket rather than a proposal and a control: one
    moves the samples narrow, the other wide, and if neither beats
    uniform then where they land does not matter.
    """
    mode = getattr(FLAGS, 'width_sampling', 'uniform')
    if mode == 'uniform':
        return random.uniform(low, high)
    if mode == 'log':
        return math.exp(random.uniform(math.log(low), math.log(high)))
    if mode == 'macs':
        return math.sqrt(random.uniform(low * low, high * high))
    raise ValueError('unknown width_sampling {}'.format(mode))


def width_gate(width_mult):
    """how strongly an extra term applies at this width

    Measured five times in this project and three more in the literature:
    every intervention in supernet training helps the narrow widths and
    costs something at the wide ones. Against plain KL the horizontal term
    ran from +0.5 at width 0.30 to -0.8 at 1.00.

    A term that is known to change sign across the range should not be
    applied at one strength across the range. 'narrow' turns it off where
    it was measured to hurt.
    """
    schedule = getattr(FLAGS, 'weight_schedule', 'constant')
    if schedule == 'constant':
        return 1.0
    low, high = FLAGS.width_mult_range[0], FLAGS.width_mult_range[-1]
    position = (width_mult - low) / max(high - low, 1e-12)
    position = min(max(position, 0.0), 1.0)
    if schedule == 'narrow':
        return 1.0 - position
    if schedule == 'wide':
        return position
    raise ValueError('unknown weight_schedule {}'.format(schedule))


class KLPairLoss(torch.nn.modules.loss._Loss):
    """asymmetric KL between two students, the control for the claim

    The argument for Wasserstein between two co-sampled widths is that they
    stand in no teacher-student relation, so an asymmetric divergence is
    wrong in principle. That argument predicts this loss does worse. If it
    does not, the symmetry was not what was doing the work.
    """
    def forward(self, output_a, output_b):
        log_a = torch.nn.functional.log_softmax(output_a, dim=1)
        log_b = torch.nn.functional.log_softmax(output_b, dim=1)
        return (log_b.exp() * (log_b - log_a)).sum(dim=1)


class JeffreysPairLoss(torch.nn.modules.loss._Loss):
    """symmetrized KL, the other half of the control

    Symmetric like the transport cost, but with no metric between classes.
    Sitting between KLPairLoss and WassersteinPairLoss, it separates what
    symmetry buys from what the geometry buys.
    """
    def forward(self, output_a, output_b):
        log_a = torch.nn.functional.log_softmax(output_a, dim=1)
        log_b = torch.nn.functional.log_softmax(output_b, dim=1)
        difference = log_a - log_b
        return 0.5 * ((log_a.exp() - log_b.exp()) * difference).sum(dim=1)


def _sinkhorn_potentials(log_p, log_q, cost, eps, n_iters):
    """dual potentials of entropic OT, log domain, no grad

    Returned detached: by the envelope theorem the gradient of the transport
    cost with respect to a marginal is its own potential, so <f, p> + <g, q>
    with f, g constant already carries the right gradient. This keeps memory
    flat in n_iters instead of unrolling every iteration.

    Both potentials are updated from the previous iterate rather than g from
    the freshly written f. Swapping the two marginals then swaps f and g
    exactly, at every iteration, so the divergence is symmetric by
    construction instead of only at convergence. That matters here: symmetry
    is the argument for using this on a pair of students, not a detail.
    """
    with torch.no_grad():
        cost_eps = cost.unsqueeze(0) / eps
        f = torch.zeros_like(log_p)
        g = torch.zeros_like(log_q)
        for _ in range(n_iters):
            f_next = eps * (log_p - torch.logsumexp(
                g.unsqueeze(1) / eps - cost_eps, dim=2))
            g_next = eps * (log_q - torch.logsumexp(
                f.unsqueeze(2) / eps - cost_eps, dim=1))
            # damped, because a simultaneous update oscillates undamped
            f = 0.5 * (f + f_next)
            g = 0.5 * (g + g_next)
    return f, g


def _sinkhorn_symmetric_potential(log_p, cost, eps, n_iters):
    """potential of OT(p, p), used to debias the divergence"""
    with torch.no_grad():
        cost_eps = cost.unsqueeze(0) / eps
        a = torch.zeros_like(log_p)
        for _ in range(n_iters):
            a_new = eps * (log_p - torch.logsumexp(
                a.unsqueeze(1) / eps - cost_eps, dim=2))
            a = 0.5 * (a + a_new)
    return a


def sinkhorn_divergence(p, q, cost, eps=0.2, n_iters=100, debiased=True):
    """entropic Wasserstein between two batches of distributions

    p, q: (batch, n_class) probability vectors. cost: (n_class, n_class).
    Returns one value per sample. Gradient flows into whichever of p, q
    requires it, so this serves the vertical term (student against a detached
    teacher) and the horizontal term (two live students) alike.

    Debiasing subtracts the self-transport terms. Entropic OT(p, p) is not
    zero, so without it the horizontal term is not minimized at p == q and
    drags both students toward a blurrier distribution.
    """
    log_p = torch.log(p.clamp_min(1e-30))
    log_q = torch.log(q.clamp_min(1e-30))
    f, g = _sinkhorn_potentials(log_p, log_q, cost, eps, n_iters)
    if debiased:
        f = f - _sinkhorn_symmetric_potential(log_p, cost, eps, n_iters)
        g = g - _sinkhorn_symmetric_potential(log_q, cost, eps, n_iters)
    return (f * p).sum(dim=1) + (g * q).sum(dim=1)


class WassersteinLossSoft(torch.nn.modules.loss._Loss):
    """Wasserstein replacement for CrossEntropyLossSoft

    Same call signature as the inplace distillation loss it replaces:
    forward(student_logits, teacher_probs) returns one value per sample. The
    cost matrix has to be refreshed from the classifier as it trains, so
    train.py calls set_cost() during the loop.
    """
    def __init__(self, eps=0.2, n_iters=100, debiased=True, reduction='none'):
        super(WassersteinLossSoft, self).__init__(reduction=reduction)
        self.eps = eps
        self.n_iters = n_iters
        self.debiased = debiased
        self.cost = None

    def set_cost(self, cost):
        self.cost = cost

    def forward(self, output, target):
        if self.cost is None:
            raise RuntimeError('set_cost must be called before forward')
        p = torch.nn.functional.softmax(output, dim=1)
        return sinkhorn_divergence(
            p, target, self.cost, self.eps, self.n_iters, self.debiased)


class WassersteinPairLoss(WassersteinLossSoft):
    """horizontal term W(s_a, s_b) between two co-sampled students

    Both arguments are live logits. Unlike the vertical term neither side is
    detached, and unlike KL the quantity is symmetric, which is the whole
    point: two students at different widths stand in no teacher-student
    relation to each other.
    """
    def forward(self, output_a, output_b):
        if self.cost is None:
            raise RuntimeError('set_cost must be called before forward')
        p = torch.nn.functional.softmax(output_a, dim=1)
        q = torch.nn.functional.softmax(output_b, dim=1)
        return sinkhorn_divergence(
            p, q, self.cost, self.eps, self.n_iters, self.debiased)


class AlphaDivergenceLossSoft(torch.nn.modules.loss._Loss):
    """adaptive alpha-divergence of AlphaNet, arXiv:2102.07954

    The real baseline for any claim about replacing KL in supernet KD, so it
    follows the reference implementation rather than the textbook formula.
    Two departures, neither cosmetic:

    The importance ratio is raised to alpha and then clipped. The plain
    divergence is unbounded as alpha moves away from one; written out
    directly it reaches 1e6 on a randomly initialized supernet and blows the
    weights up within a handful of steps.

    What is minimized is not the divergence but a surrogate whose weights
    are detached, with the gradient carried only through log q. The two
    alphas are compared by their true divergence per sample and the
    surrogate belonging to the larger one is returned, which is what makes
    the loss adaptive: neither over- nor under-estimating the teacher's
    uncertainty goes unpenalized.

    Checked line by line against AdaptiveLossSoft and f_divergence in
    facebookresearch/AlphaNet. One adaptation and one safeguard:

    The reference takes teacher logits and softmaxes them. US-Net hands its
    sub-networks teacher probabilities, since forward_loss already took the
    softmax at the widest width, so those are used directly.

    q_prob is floored at 1e-12 before the division. The reference divides
    straight through, which yields inf when the student has collapsed a
    class to zero, and relies on the clip and on gradient clipping to
    contain it.

    That gradient clip is not optional. The reference says so in a comment
    and alpha = 1, its own default upper alpha, is unclipped. The CIFAR
    configs set grad_clip for every branch so that the optimizer is
    identical across them and only the loss differs.
    """
    def __init__(self, alpha_min=-1.0, alpha_max=1.0, iw_clip=5.0,
                 reduction='none'):
        super(AlphaDivergenceLossSoft, self).__init__(reduction=reduction)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.iw_clip = iw_clip

    def _divergence(self, output, target, alpha):
        q_prob = torch.nn.functional.softmax(output, dim=1).detach()
        p_prob = target.detach()
        log_q = torch.nn.functional.log_softmax(output, dim=1)

        # Where the clip goes is the whole thing. It is applied to the ratio
        # raised to alpha, not to the ratio, so a class the student is
        # confident about and the teacher is not stays informative instead of
        # being flattened. alpha = 1 is left unclipped, as upstream; the
        # gradient clip in train.py is what bounds it, which is exactly what
        # the reference implementation tells you to add.
        # q cancels: q_prob * rho_f is p_prob again, which is what keeps the
        # ratio usable at 1e17. Flooring q at anything a student can
        # actually reach breaks that cancellation and silently zeroes the
        # very samples this is meant to punish. Only the smallest
        # representable number is held back, to turn a division by an
        # underflowed zero into a large finite number instead of a nan.
        importance_ratio = p_prob / q_prob.clamp_min(
            torch.finfo(q_prob.dtype).tiny)
        if abs(alpha) < 1e-3:
            importance_ratio = importance_ratio.clamp(0, self.iw_clip)
            log_ratio = importance_ratio.clamp_min(1e-12).log()
            f = -log_ratio
            f_base = 0.0
            rho_f = log_ratio - 1.0
        elif abs(alpha - 1.0) < 1e-3:
            f = importance_ratio * importance_ratio.clamp_min(1e-12).log()
            f_base = 0.0
            rho_f = importance_ratio
        else:
            iw_alpha = torch.pow(importance_ratio, alpha)
            iw_alpha = iw_alpha.clamp(0, self.iw_clip)
            f = iw_alpha / alpha / (alpha - 1.0)
            f_base = 1.0 / alpha / (alpha - 1.0)
            rho_f = iw_alpha / alpha + f_base

        divergence = torch.sum(q_prob * (f - f_base), dim=1)
        surrogate = -torch.sum(q_prob * rho_f * log_q, dim=1)
        return divergence, surrogate

    def forward(self, output, target):
        low, surrogate_low = self._divergence(output, target, self.alpha_min)
        high, surrogate_high = self._divergence(output, target, self.alpha_max)
        return torch.where(low > high, surrogate_low, surrogate_high)


def build_soft_criterion():
    """branch selector for the A/B/C/D experiment

    soft_ce      A  inplace KD of US-Net, which is KL
    alpha        B  AlphaNet
    wasserstein  C  WKD, vertical only
                 D  is C plus horizontal_kd, see build_pair_criterion
    """
    kd_loss = getattr(FLAGS, 'kd_loss', 'soft_ce')
    if kd_loss == 'soft_ce':
        return CrossEntropyLossSoft(reduction='none')
    if kd_loss == 'alpha':
        return AlphaDivergenceLossSoft(
            alpha_min=getattr(FLAGS, 'alpha_min', -1.0),
            alpha_max=getattr(FLAGS, 'alpha_max', 1.0),
            iw_clip=getattr(FLAGS, 'alpha_iw_clip', 5.0),
            reduction='none')
    if kd_loss == 'wasserstein':
        return WassersteinLossSoft(
            eps=getattr(FLAGS, 'sinkhorn_eps', 0.2),
            n_iters=getattr(FLAGS, 'sinkhorn_iters', 100),
            debiased=getattr(FLAGS, 'sinkhorn_debiased', True),
            reduction='none')
    raise ValueError('unknown kd_loss {}'.format(kd_loss))


def build_pair_criterion():
    """horizontal criterion at the logit level, or None

    horizontal_loss picks what the two co-sampled widths are pulled
    together with. The three settings are a decomposition, not a menu:
    wasserstein is symmetric and metric aware, jeffreys is symmetric only,
    kl is neither. Which pair of them differ says which property mattered.
    """
    if not getattr(FLAGS, 'horizontal_kd', False):
        return None
    if getattr(FLAGS, 'horizontal_where', 'logit') == 'feature':
        return None
    # 'both' keeps this one and the feature one at the same time. The two
    # tiers are the only places anything has reached the published method,
    # by different mechanisms - the logit pair moves calibration, the
    # feature pair moves accuracy - and nothing has asked whether they add.
    horizontal_loss = getattr(FLAGS, 'horizontal_loss', 'wasserstein')
    if horizontal_loss == 'kl':
        return KLPairLoss(reduction='none')
    if horizontal_loss == 'jeffreys':
        return JeffreysPairLoss(reduction='none')
    if horizontal_loss == 'wasserstein':
        return WassersteinPairLoss(
            eps=getattr(FLAGS, 'sinkhorn_eps', 0.2),
            n_iters=getattr(FLAGS, 'sinkhorn_iters', 100),
            debiased=getattr(FLAGS, 'sinkhorn_debiased', True),
            reduction='none')
    raise ValueError('unknown horizontal_loss {}'.format(horizontal_loss))


def _inner_feature_loss(align):
    """what compares one pair of taps, chosen by feature_loss

    Four settings, and like horizontal_loss at the logit level they are a
    decomposition rather than a menu:

      mse          pairs sample i with sample i, no distribution matching
      mmd          matches the clouds, no transport plan
      sliced       transport, along random one dimensional projections
      wasserstein  entropic transport with a full plan

    mse against mmd says whether rematching samples is needed at all. mmd
    against either transport setting says whether optimal transport in
    particular is doing the work or any distribution distance would.
    sliced against wasserstein is a question about cost at fixed geometry.

    The feature tier is the only place anything has beaten the published
    method, and it was run with one of these four and no controls. That
    asymmetry against the logit tier, which got a three way decomposition
    and lost, is the first thing a reader will pick at.
    """
    name = getattr(FLAGS, 'feature_loss', 'wasserstein')
    ground = getattr(FLAGS, 'feature_ground', 'euclidean')
    weighting = getattr(FLAGS, 'feature_weighting', 'none')
    if name == 'mse':
        return FeatureMSELoss(align=align)
    if name == 'mmd':
        return FeatureMMDLoss(
            align=align, bandwidth=getattr(FLAGS, 'mmd_bandwidth', 1.0))
    if name == 'sliced':
        return FeatureSlicedWassersteinLoss(
            align=align,
            n_projections=getattr(FLAGS, 'sliced_projections', 128),
            reduce=getattr(FLAGS, 'sliced_reduce', 'mean'),
            p=getattr(FLAGS, 'wasserstein_p', 2.0))
    if name == 'channel':
        return FeatureChannelLoss(
            align=align, p=getattr(FLAGS, 'wasserstein_p', 2.0))
    if name == 'bures':
        return FeatureBuresLoss(
            align=align, diagonal=getattr(FLAGS, 'bures_diagonal', False))
    if name == 'gromov':
        return FeatureGromovLoss(
            eps=getattr(FLAGS, 'sinkhorn_eps', 0.2),
            n_iters=getattr(FLAGS, 'gromov_inner_iters', 50),
            outer_iters=getattr(FLAGS, 'gromov_outer_iters', 5),
            ground=ground)
    if name == 'unbalanced':
        return FeatureUnbalancedWassersteinLoss(
            tau=getattr(FLAGS, 'unbalanced_tau', 1.0),
            eps=getattr(FLAGS, 'sinkhorn_eps', 0.2),
            n_iters=getattr(FLAGS, 'sinkhorn_iters', 100),
            align=align,
            debiased=getattr(FLAGS, 'sinkhorn_debiased', True),
            ground=ground, weighting=weighting)
    if name == 'wasserstein':
        return FeatureWassersteinLoss(
            eps=getattr(FLAGS, 'sinkhorn_eps', 0.2),
            n_iters=getattr(FLAGS, 'sinkhorn_iters', 100),
            align=align,
            debiased=getattr(FLAGS, 'sinkhorn_debiased', True),
            ground=ground, weighting=weighting)
    raise ValueError('unknown feature_loss {}'.format(name))


def _feature_loss():
    align = getattr(FLAGS, 'feature_align', 'prefix')
    if getattr(FLAGS, 'feature_classwise', False):
        # the tiers are handled inside, since the centroids have to be
        # taken per tap before anything is transported
        return ClasswiseFeatureLoss(
            inner=_inner_feature_loss(align),
            min_count=getattr(FLAGS, 'classwise_min_count', 1))
    return MultiTierFeatureLoss(
        inner=_inner_feature_loss(align),
        tier_weights=getattr(FLAGS, 'tier_weights', None))


def build_feature_criterion():
    """vertical feature term, or None when the branch does not use one"""
    if not getattr(FLAGS, 'feature_kd', False):
        return None
    return _feature_loss()


def build_feature_pair_criterion():
    """horizontal feature term, or None

    The combination the two roles were meant to meet in: geometry where
    RQ1 measured it, between the pair of widths that nothing else relates.
    """
    if not getattr(FLAGS, 'horizontal_kd', False):
        return None
    if getattr(FLAGS, 'horizontal_where', 'logit') == 'logit':
        return None
    return _feature_loss()


def horizontal_pairs(widths):
    """which co-sampled widths get pulled together

    'middle' is what every finished run used: the sandwich rule spends two
    of its samples on the widest and narrowest, and the pair left over is
    the two in between. That leaves most of the available couplings on the
    floor. With four samples the narrowest is also a student with no
    teacher-student relation to either middle, so there are three pairs
    available and one was being used.

    'all' takes every pair among them. The horizontal term is where K's
    gain came from, +0.61 of its +0.75, so how much of it there is may
    matter more than which distance it is measured with.

    widths is the list of co-sampled widths excluding the teacher, in the
    order they were run, with the narrowest first.
    """
    count = len(widths)
    if getattr(FLAGS, 'horizontal_pairs', 'middle') == 'all':
        return [(i, j)
                for i in range(count) for j in range(i + 1, count)]
    # the middles are everything but the narrowest, which came first
    middles = list(range(1, count))
    return [(middles[i], middles[j])
            for i in range(len(middles))
            for j in range(i + 1, len(middles))]


def build_spread_criterion():
    """repulsion on the classifier rows, or None when unused

    Off unless spread_weight is set, because it changes the objective for
    every width at once rather than adding a term between two of them.
    """
    if not getattr(FLAGS, 'spread_weight', 0):
        return None
    return ClassifierSpreadLoss(
        neighbours=getattr(FLAGS, 'spread_neighbours', 5))


def build_confusion_embedding():
    """accumulator for cost_source 'confusion', or None"""
    if getattr(FLAGS, 'cost_source', 'fc') != 'confusion':
        return None
    return ConfusionEmbedding(
        FLAGS.num_classes,
        momentum=getattr(FLAGS, 'confusion_momentum', 0.01),
        warmup=getattr(FLAGS, 'confusion_warmup', 50))


def build_cost_matrix(model, confusion=None):
    """the class metric the logit-level transport is measured against

    fc         rows of the shared classifier. Intrinsic to the model, and
               measured at a spread of 1.32 on a trained supernet, which is
               nearly flat.
    confusion  what the teacher mixes each class up with. Has to warm up,
               so it falls back to fc until it has.
    identity   no geometry at all, the control.
    """
    source = getattr(FLAGS, 'cost_source', 'fc')
    normalize = getattr(FLAGS, 'cost_normalize', True)
    if source == 'identity':
        weight = get_classifier_weight(model)
        return identity_cost_matrix(
            weight.size(0), weight.device, weight.dtype)
    if source == 'confusion' and confusion is not None and confusion.ready():
        return class_cost_matrix(confusion.embedding(), normalize=normalize)
    if source not in ('fc', 'confusion'):
        raise ValueError('unknown cost_source {}'.format(source))
    return class_cost_matrix(
        get_classifier_weight(model), normalize=normalize)
