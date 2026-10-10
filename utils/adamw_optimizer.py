"""AdamW for the ViT runs (optimizer: utils.adamw_optimizer)

train.py imports a module named by `optimizer` when it is not 'sgd' and
calls its get_optimizer. Weight decay goes on the matrices only: biases,
LayerNorm parameters, the class token and the position embedding take
none, as in DeiT and timm. lr_schedule_per_iteration sets the rate of
every group each step, so warm-up and cosine apply here unchanged.
"""
import torch

from utils.config import FLAGS


NO_DECAY = ('cls_token', 'pos_embed')


def get_optimizer(model):
    decay, plain = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.split('.')[-1] in NO_DECAY:
            plain.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{'params': decay, 'weight_decay': FLAGS.weight_decay},
         {'params': plain, 'weight_decay': 0.0}],
        lr=FLAGS.lr, betas=tuple(getattr(FLAGS, 'adam_betas', (0.9, 0.999))))
