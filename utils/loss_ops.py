import math

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


def feature_cost(student, teacher, align='prefix'):
    """scaled pairwise distance between two batches of features"""
    left, right = align_features(student, teacher, align)
    return torch.cdist(left, right) / _feature_scale(left, right)


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
                 reduction='mean'):
        super(FeatureWassersteinLoss, self).__init__(reduction=reduction)
        self.eps = eps
        self.n_iters = n_iters
        self.align = align
        self.debiased = debiased

    def _transport(self, left, right, scale):
        cost = torch.cdist(left, right) / scale
        plan = sinkhorn_plan(cost.detach(), self.eps, self.n_iters)
        return (plan * cost).sum()

    def forward(self, student, teacher):
        left, right = align_features(student, teacher, self.align)
        scale = _feature_scale(left, right)
        value = self._transport(left, right, scale)
        if self.debiased:
            value = value - 0.5 * self._transport(left, left, scale)
            value = value - 0.5 * self._transport(right, right, scale)
        return value


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


def build_feature_criterion():
    """vertical feature term, or None when the branch does not use one"""
    if not getattr(FLAGS, 'feature_kd', False):
        return None
    return FeatureWassersteinLoss(
        eps=getattr(FLAGS, 'sinkhorn_eps', 0.2),
        n_iters=getattr(FLAGS, 'sinkhorn_iters', 100),
        align=getattr(FLAGS, 'feature_align', 'prefix'))


def build_feature_pair_criterion():
    """horizontal feature term, or None

    The combination the two roles were meant to meet in: geometry where
    RQ1 measured it, between the pair of widths that nothing else relates.
    """
    if not getattr(FLAGS, 'horizontal_kd', False):
        return None
    if getattr(FLAGS, 'horizontal_where', 'logit') == 'logit':
        return None
    return FeatureWassersteinLoss(
        eps=getattr(FLAGS, 'sinkhorn_eps', 0.2),
        n_iters=getattr(FLAGS, 'sinkhorn_iters', 100),
        align=getattr(FLAGS, 'feature_align', 'prefix'))


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
