"""numerical checks for the Wasserstein and alpha-divergence losses

Run before spending any GPU time:

    python tests/test_loss_ops.py

utils/config.py calls app() at import, which reads sys.argv for app: or else
blocks on stdin, so a config is appended below before anything is imported.
"""
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not any(arg.startswith('app:') for arg in sys.argv):
    sys.argv.append('app:apps/kd_c_wasserstein.yml')

import torch

from utils.loss_ops import AlphaDivergenceLossSoft
from utils.loss_ops import CrossEntropyLossSoft
from utils.loss_ops import WassersteinLossSoft
from utils.loss_ops import WassersteinPairLoss
from utils.config import FLAGS
from utils.loss_ops import _sinkhorn_potentials
from utils.loss_ops import class_cost_matrix
from utils.loss_ops import sinkhorn_divergence


FAILURES = []


def check(name, condition, detail=''):
    status = 'pass' if condition else 'FAIL'
    print('{:5} {}{}'.format(status, name, '  ' + detail if detail else ''))
    if not condition:
        FAILURES.append(name)


def random_probs(batch, n_class, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.softmax(
        torch.randn(batch, n_class, generator=generator), dim=1)


def toy_cost():
    """three classes where 0 and 1 are neighbours and 2 is far off

    Stands in for the dog / wolf / truck example: KL cannot tell the two
    mistakes apart, a transport cost can.
    """
    return torch.tensor([
        [0.0, 0.1, 3.0],
        [0.1, 0.0, 3.0],
        [3.0, 3.0, 0.0],
    ])


def test_self_distance_is_zero():
    p = random_probs(8, 20, seed=0)
    cost = torch.cdist(torch.randn(20, 6), torch.randn(20, 6))
    cost = 0.5 * (cost + cost.t())
    cost.fill_diagonal_(0.0)
    value = sinkhorn_divergence(p, p, cost, eps=0.05, n_iters=200)
    check(
        'debiased S(p, p) == 0',
        value.abs().max().item() < 1e-3,
        'max |S| = {:.2e}'.format(value.abs().max().item()))
    biased = sinkhorn_divergence(p, p, cost, eps=0.05, n_iters=200,
                                 debiased=False)
    check(
        'biased OT(p, p) != 0, which is why debiasing is on',
        biased.abs().max().item() > 1e-3,
        'max |OT| = {:.2e}'.format(biased.abs().max().item()))


def test_symmetry():
    p = random_probs(8, 20, seed=1)
    q = random_probs(8, 20, seed=2)
    cost = torch.cdist(torch.randn(20, 6), torch.randn(20, 6))
    cost = 0.5 * (cost + cost.t())
    cost.fill_diagonal_(0.0)
    forward = sinkhorn_divergence(p, q, cost, eps=0.05, n_iters=200)
    reverse = sinkhorn_divergence(q, p, cost, eps=0.05, n_iters=200)
    gap = (forward - reverse).abs().max().item()
    check('S(p, q) == S(q, p)', gap < 1e-5, 'max gap = {:.2e}'.format(gap))


def test_nonnegative():
    p = random_probs(16, 20, seed=3)
    q = random_probs(16, 20, seed=4)
    cost = torch.cdist(torch.randn(20, 6), torch.randn(20, 6))
    cost = 0.5 * (cost + cost.t())
    cost.fill_diagonal_(0.0)
    value = sinkhorn_divergence(p, q, cost, eps=0.05, n_iters=200)
    check(
        'S(p, q) >= 0',
        value.min().item() > -1e-4,
        'min = {:.2e}'.format(value.min().item()))


def test_metric_awareness():
    """the whole point: a near mistake must cost less than a far one"""
    cost = toy_cost()
    teacher = torch.tensor([[0.8, 0.1, 0.1]])
    near = torch.tensor([[0.1, 0.8, 0.1]])   # mass moved to the neighbour
    far = torch.tensor([[0.1, 0.1, 0.8]])    # mass moved far away
    w_near = sinkhorn_divergence(near, teacher, cost, eps=0.01, n_iters=500)
    w_far = sinkhorn_divergence(far, teacher, cost, eps=0.01, n_iters=500)
    check(
        'W penalizes the far mistake more',
        w_far.item() > w_near.item() * 1.5,
        'near = {:.4f}, far = {:.4f}'.format(w_near.item(), w_far.item()))

    kl = CrossEntropyLossSoft(reduction='none')
    kl_near = kl(torch.log(near), teacher).view(-1)
    kl_far = kl(torch.log(far), teacher).view(-1)
    check(
        'KL cannot tell them apart, which is the gap being exploited',
        abs(kl_near.item() - kl_far.item()) < 1e-4,
        'near = {:.4f}, far = {:.4f}'.format(kl_near.item(), kl_far.item()))


def test_gradients_flow_both_ways():
    """the horizontal term has no detached side, unlike the vertical one"""
    cost = toy_cost()
    logits_a = torch.randn(4, 3, requires_grad=True)
    logits_b = torch.randn(4, 3, requires_grad=True)
    pair = WassersteinPairLoss(eps=0.05, n_iters=200)
    pair.set_cost(cost)
    loss = pair(logits_a, logits_b).mean()
    loss.backward()
    check(
        'gradient reaches student a',
        logits_a.grad is not None and torch.isfinite(logits_a.grad).all()
        and logits_a.grad.abs().sum().item() > 0)
    check(
        'gradient reaches student b',
        logits_b.grad is not None and torch.isfinite(logits_b.grad).all()
        and logits_b.grad.abs().sum().item() > 0)


def test_vertical_loss_interface():
    """must be a drop-in for CrossEntropyLossSoft in forward_loss"""
    cost = toy_cost()
    logits = torch.randn(5, 3, requires_grad=True)
    teacher = torch.softmax(torch.randn(5, 3), dim=1)
    loss_fn = WassersteinLossSoft(eps=0.05, n_iters=200)
    loss_fn.set_cost(cost)
    value = loss_fn(logits, teacher)
    check('returns one value per sample', value.shape == (5,),
          'shape = {}'.format(tuple(value.shape)))
    torch.mean(value).backward()
    check(
        'gradient is finite',
        torch.isfinite(logits.grad).all()
        and logits.grad.abs().sum().item() > 0)


def test_gradient_points_the_right_way():
    """moving the student toward the teacher must lower the loss"""
    cost = toy_cost()
    teacher = torch.tensor([[0.8, 0.1, 0.1]])
    logits = torch.zeros(1, 3, requires_grad=True)
    loss_fn = WassersteinLossSoft(eps=0.02, n_iters=500)
    loss_fn.set_cost(cost)
    before = loss_fn(logits, teacher).item()
    for _ in range(200):
        loss = loss_fn(logits, teacher).mean()
        if logits.grad is not None:
            logits.grad.zero_()
        loss.backward()
        with torch.no_grad():
            logits -= 5.0 * logits.grad
    after = loss_fn(logits, teacher).item()
    student = torch.softmax(logits, dim=1).detach()
    check(
        'descent lowers the loss',
        after < before,
        'before = {:.4f}, after = {:.4f}'.format(before, after))
    check(
        'student moved onto the teacher argmax',
        student.argmax().item() == teacher.argmax().item(),
        'student = {}'.format([round(v, 3) for v in student[0].tolist()]))


def marginal_violation(p, q, cost, eps, n_iters):
    """how far the recovered transport plan is from its prescribed marginals

    The potentials are solved under no_grad and the gradient is taken from
    them by the envelope theorem, which is only valid at convergence. An
    under-solved Sinkhorn therefore produces a wrong gradient and no error,
    so this is the one number worth watching.
    """
    log_p = torch.log(p.clamp_min(1e-30))
    log_q = torch.log(q.clamp_min(1e-30))
    f, g = _sinkhorn_potentials(log_p, log_q, cost, eps, n_iters)
    plan = torch.exp(
        (f.unsqueeze(2) + g.unsqueeze(1) - cost.unsqueeze(0)) / eps)
    row = (plan.sum(dim=2) - p).abs().max().item()
    col = (plan.sum(dim=1) - q).abs().max().item()
    return max(row, col)


def test_sinkhorn_converges_at_configured_settings():
    """checks the eps and iteration count the yml files actually ship

    If this fails, raise sinkhorn_iters or sinkhorn_eps in apps/kd_c and
    apps/kd_d before training anything.
    """
    n_class = 100
    cost = class_cost_matrix(torch.randn(n_class, 128))
    p = random_probs(32, n_class, seed=7)
    q = random_probs(32, n_class, seed=8)
    eps = getattr(FLAGS, 'sinkhorn_eps', 0.05)
    n_iters = getattr(FLAGS, 'sinkhorn_iters', 50)
    violation = marginal_violation(p, q, cost, eps, n_iters)
    check(
        'sinkhorn converged at eps={}, iters={}'.format(eps, n_iters),
        violation < 1e-3,
        'max marginal violation = {:.2e}'.format(violation))
    print('      convergence sweep at eps = {}:'.format(eps))
    for iters in (10, 25, 50, 100, 200, 400):
        print('        {:4d} iters -> {:.2e}'.format(
            iters, marginal_violation(p, q, cost, eps, iters)))
    print('      sweep over eps at {} iters:'.format(n_iters))
    for trial_eps in (0.01, 0.05, 0.1, 0.25, 0.5):
        print('        eps {:.2f} -> {:.2e}'.format(
            trial_eps, marginal_violation(p, q, cost, trial_eps, n_iters)))


def test_cost_matrix_from_classifier():
    weight = torch.randn(10, 7)
    cost = class_cost_matrix(weight)
    check('cost is square over classes', cost.shape == (10, 10),
          'shape = {}'.format(tuple(cost.shape)))
    check('zero on the diagonal', cost.diagonal().abs().max().item() < 1e-5)
    check('symmetric', (cost - cost.t()).abs().max().item() < 1e-5)
    check('mean normalized to one', abs(cost.mean().item() - 1.0) < 1e-5)
    check('detached from the classifier', not cost.requires_grad)


def test_alpha_reduces_to_kl():
    logits = torch.randn(6, 9)
    teacher = torch.softmax(torch.randn(6, 9), dim=1)
    alpha = AlphaDivergenceLossSoft(alphas=(1.0,), clamp=1e9)
    kl = CrossEntropyLossSoft(reduction='none')
    # the soft cross entropy carries the teacher entropy term, KL does not
    entropy = -(teacher * torch.log(teacher.clamp_min(1e-30))).sum(dim=1)
    gap = (alpha(logits, teacher)
           - (kl(logits, teacher).view(-1) - entropy)).abs().max().item()
    check(
        'alpha = 1 is forward KL',
        gap < 1e-4,
        'max gap = {:.2e}'.format(gap))


def test_alpha_matches_direct_formula():
    """guards the log-domain rewrite of sum_i p^a q^(1-a)

    Mild distributions and a wide clamp so the closed form is safe to
    evaluate directly for comparison.
    """
    teacher = torch.softmax(0.5 * torch.randn(4, 8), dim=1).double()
    logits = 0.5 * torch.randn(4, 8).double()
    student = torch.softmax(logits, dim=1)
    for a in (-1.0, 0.5, 2.0, 3.0):
        direct = ((teacher ** a * student ** (1.0 - a)).sum(dim=1) - 1.0)
        direct = direct / (a * (a - 1.0))
        got = AlphaDivergenceLossSoft(alphas=(a,), clamp=1e9)(logits, teacher)
        gap = (got - direct).abs().max().item()
        check(
            'alpha = {} matches the closed form'.format(a),
            gap < 1e-8,
            'max gap = {:.2e}'.format(gap))


def test_alpha_is_finite_under_mismatch():
    """a confident student against a confident, disagreeing teacher"""
    logits = torch.tensor([[20.0, -20.0, -20.0]])
    teacher = torch.tensor([[1e-8, 1.0 - 2e-8, 1e-8]])
    alpha = AlphaDivergenceLossSoft(alphas=(-1.0, 1.0), clamp=5.0)
    value = alpha(logits, teacher)
    check(
        'clamping keeps alpha-divergence finite',
        torch.isfinite(value).all(),
        'value = {:.4f}'.format(value.item()))


def main():
    print('torch', torch.__version__)
    print()
    test_self_distance_is_zero()
    test_symmetry()
    test_nonnegative()
    test_metric_awareness()
    test_gradients_flow_both_ways()
    test_vertical_loss_interface()
    test_gradient_points_the_right_way()
    test_sinkhorn_converges_at_configured_settings()
    test_cost_matrix_from_classifier()
    test_alpha_reduces_to_kl()
    test_alpha_matches_direct_formula()
    test_alpha_is_finite_under_mismatch()
    print()
    if FAILURES:
        print('{} failed: {}'.format(len(FAILURES), ', '.join(FAILURES)))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
