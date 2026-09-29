"""student_kd_weight: the logit KD on every student, off at 0.

Built to fail if the flag is ignored: at 0, with student_ce_weight 1, a
student's loss has to be the label cross entropy exactly and its gradient
has to be blind to the teacher's soft target; at the default it has to be
the KD loss it always was.

Runs on the CPU in a few seconds: python tests/test_no_kd.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = [sys.argv[0], 'app:apps/smoke_k_feature_pair.yml']

import torch  # noqa: E402

import train as T  # noqa: E402
from utils.config import FLAGS  # noqa: E402


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print('ok   ', what)


def student_loss(model, x, y, soft, kd, ce):
    FLAGS.student_kd_weight = kd
    FLAGS.student_ce_weight = ce
    model.zero_grad(set_to_none=True)
    model.apply(lambda m: setattr(m, 'width_mult', 0.5))
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    loss = T.forward_loss(model, criterion, x, y, None, soft_target=soft,
                          soft_criterion=T.build_soft_criterion())
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in model.parameters()
                      if p.grad is not None])
    return loss.item(), grad


if __name__ == '__main__':
    FLAGS.reset_parameters = True
    FLAGS.return_features = False
    torch.manual_seed(0)
    model = importlib.import_module('models.us_resnet').Model(100, 32)
    model.train()
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 100, (8,))
    soft_a = torch.softmax(torch.randn(8, 100), dim=1)
    soft_b = torch.softmax(3 * torch.randn(8, 100), dim=1)

    model.apply(lambda m: setattr(m, 'width_mult', 0.5))
    with torch.no_grad():
        ce = torch.nn.functional.cross_entropy(model(x), y).item()

    loss_a, grad_a = student_loss(model, x, y, soft_a, 0.0, 1.0)
    loss_b, grad_b = student_loss(model, x, y, soft_b, 0.0, 1.0)
    check(abs(loss_a - ce) < 1e-5,
          'at 0 the student loss is the label cross entropy ({:.5f})'.format(
              loss_a))
    check(torch.equal(grad_a, grad_b),
          'at 0 the gradient does not depend on the teacher at all')

    kd_a, kg_a = student_loss(model, x, y, soft_a, 1.0, 0.0)
    kd_b, kg_b = student_loss(model, x, y, soft_b, 1.0, 0.0)
    check(abs(kd_a - kd_b) > 1e-3 and not torch.equal(kg_a, kg_b),
          'at the default the student hears the teacher, so this can fail')
    del FLAGS.student_kd_weight
    FLAGS.student_ce_weight = 0.0
    model.zero_grad(set_to_none=True)
    default = T.forward_loss(
        model, torch.nn.CrossEntropyLoss(reduction='none'), x, y, None,
        soft_target=soft_a, soft_criterion=T.build_soft_criterion()).item()
    check(default == kd_a,
          'without the flag the loss is the KD loss it always was')
    print('all checks passed')
