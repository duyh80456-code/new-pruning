# Third-party sources and attribution

This project follows a reuse-first policy. No upstream repository is vendored.
The local adaptations are small modern-PyTorch ports and retain source links in
module docstrings.

## Slimmable Networks / Universally Slimmable Networks

- Source repository: https://github.com/JiahuiYu/slimmable_networks
- Source file studied: `models/slimmable_ops.py`
- Concepts adapted: runtime width switching, prefix slicing of convolution and
  linear weights, shared-affine universally slimmable BatchNorm, per-width
  statistic banks, and post-training BN calibration.
- Local files: `models/slimmable_ops.py`, `evaluation.py`, and the explicit width
  loop in `training.py`.
- Changes: removed global `FLAGS`, used exact `floor(alpha * channels)` as required
  by this experiment, made supported budgets model-local, and ported to current
  PyTorch APIs.
- Upstream license: CC BY-NC 4.0; research/non-commercial restrictions apply to
  adapted upstream material. See the upstream repository for its complete terms.

## torchvision ResNet and CIFAR-100

- ResNet source: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
- Dataset source/API:
  https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py
- Reused/adapted: ResNet `BasicBlock`, stage construction, residual downsampling,
  `CIFAR100`, `FakeData`, and standard transforms.
- Changes: 3x3 stride-1 CIFAR stem, no max-pooling, dynamic operators, and a shared
  fixed-dimensional projection.
- Upstream license: BSD-3-Clause.

## Once-for-All

- Source repository: https://github.com/mit-han-lab/once-for-all
- Inspected for dynamic-layer, active-subnet, channel-ordering, and BN-calibration
  design. No OFA source is copied. Its broader elastic search stack is deliberately
  excluded from this width-only baseline.

## Libraries used unchanged

- PyTorch / torchvision: model execution, datasets, transforms, serialization.
- THOP: operation counting, with local hooks only for the dynamic wrappers.
- SciPy: exact one-dimensional Wasserstein distances and correlation statistics.
- scikit-learn: simple linear regression controls.
- pandas / NumPy / Matplotlib / PyYAML: tables, arrays, figures, configuration.

Dependency licenses remain with their respective copyright holders.
