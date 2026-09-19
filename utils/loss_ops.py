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
    """horizontal criterion, or None when the branch does not use one"""
    if not getattr(FLAGS, 'horizontal_kd', False):
        return None
    return WassersteinPairLoss(
        eps=getattr(FLAGS, 'sinkhorn_eps', 0.2),
        n_iters=getattr(FLAGS, 'sinkhorn_iters', 100),
        debiased=getattr(FLAGS, 'sinkhorn_debiased', True),
        reduction='none')
