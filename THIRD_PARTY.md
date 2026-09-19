# Third party code

Both of the sources below are licensed **Creative Commons
Attribution-NonCommercial 4.0 International** (CC BY-NC 4.0). This
repository inherits those terms: attribution is required and commercial use
is not permitted.

## US-Net / slimmable_networks

- Source: https://github.com/JiahuiYu/slimmable_networks
- Snapshot: `5dc14d0`, 2020-09-03
- Vendored as the first commit of this branch, unmodified, so that every
  later change reads as a diff against it.
- Paper: *Universally Slimmable Networks and Improved Training Techniques*,
  Yu and Huang, ICCV 2019, arXiv:1903.05134
- License file kept at `LICENSE`.

Everything outside `models/us_resnet.py`, `data/`, `tests/`, the
`apps/cifar100_*.yml` and `apps/smoke_*.yml` configs, the Wasserstein and
alpha-divergence code in `utils/loss_ops.py`, and the changes to `train.py`
comes from this snapshot.

## AlphaNet

- Source: https://github.com/facebookresearch/AlphaNet
- Snapshot: `1df57bf`
- Vendored at `tests/reference/alphanet_loss_ops.py`, unmodified.
- Paper: *AlphaNet: Improved Training of Supernets with Alpha-Divergence*,
  Wang et al., ICML 2021, arXiv:2102.07954

Used only by `tests/test_loss_ops.py`, which checks that the branch B loss
in `utils/loss_ops.py` agrees with it to the last bit in value and
gradient. No training code imports it. It is vendored rather than cloned on
demand so the check runs wherever the repository does, including on Kaggle
with the network off.
