# Repository audit

This audit was completed before implementation of the research-specific geometry code.

## Sources inspected

### JiahuiYu/slimmable_networks (official)

- Repository: https://github.com/JiahuiYu/slimmable_networks
- Relevant files: `models/slimmable_ops.py`, `models/slimmable_ops.py`, and the
  width loop in `train.py`.
- Width switching is implemented by setting a runtime `width_mult` on slimmable
  modules. Convolutional/linear weights are prefix-sliced, so channel sets are
  naturally nested.
- The original slimmable models use switchable BatchNorm for a finite set of
  widths. Universally Slimmable Networks extend training to arbitrary widths and
  rely on post-training BatchNorm-statistics calibration for reliable subnet
  evaluation.
- The implementation is compact and directly matches this project's single-axis
  (channel width) compression coordinate. Its published dependency stack is old,
  so copying the environment or global mutable configuration would add needless
  compatibility risk.

### mit-han-lab/once-for-all (official)

- Repository: https://github.com/mit-han-lab/once-for-all
- Relevant area: `ofa/imagenet_classification/elastic_nn/modules/dynamic_layers.py`
  and dynamic operators under `elastic_nn/modules`.
- OFA provides dynamic layers, active-subnet extraction, BatchNorm recalibration,
  channel reordering, and progressive shrinking.
- OFA is designed for a larger search space (kernel size, expansion ratio, depth,
  resolution, and sometimes width). That machinery is unnecessary for a CIFAR-100
  ResNet-18 width-only smoke test and would obscure the experimental variable.

### torchvision / PyTorch

- ResNet source: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
- CIFAR-100 dataset API:
  https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html
- `torchvision.models.resnet.BasicBlock` supplies the cleanest modern ResNet-18
  topology. CIFAR adaptation requires only a 3x3 stride-1 stem and removal of the
  max-pool. The data pipeline can use `torchvision.datasets.CIFAR100` unchanged.

## Decision

Use a modern, minimal port of the official slimmable-operator design inside a
torchvision-style ResNet-18 topology. Prefix slicing gives deterministic nested
channels. Maintain independent running statistics for every configured evaluation
width and recalibrate them before dense evaluation. Use the exact configured
anchor list in training; never sample an intermediate width.

The fixed-dimensional geometry representation is a shared linear projection of
the active pooled feature prefix. This is deliberately simpler than a learned
alignment network and is trained jointly with classification.

## REUSED

- `torchvision.datasets.CIFAR100` and torchvision transforms.
- The ResNet-18 stage/block topology and residual-downsample pattern from
  `torchvision.models.resnet`.
- Prefix weight slicing, runtime width switching, and width-specific BatchNorm
  concepts from `JiahuiYu/slimmable_networks`.
- BatchNorm calibration concept used by US-Net/OFA evaluation.
- `thop` for FLOPs/MAC and parameter accounting, with custom counting hooks for
  dynamic operators.
- SciPy, pandas, scikit-learn, and Matplotlib for statistics, tabulation,
  regression, and plots.

## ADAPTED

- ResNet uses a CIFAR stem (3x3, stride 1, no max-pool).
- Slimmable operators are a small modern-PyTorch port with explicit model-local
  width state rather than the upstream global flags.
- BatchNorm banks cover the dense evaluation grid. Only anchor banks receive
  gradient-training batches; unseen-width running statistics are populated solely
  by explicit post-training calibration.
- A shared, prefix-sliced projection head produces a fixed-dimensional feature
  vector for every width.
- THOP custom hooks count active rather than allocated supernet operations.

## NEW RESEARCH CODE

- Feature extraction with stable sample IDs/order across budgets.
- Pluggable distribution-distance interface and Sliced Wasserstein computation.
- Pairwise geometry/resource matrices, local width/FLOPs sensitivity, coverage,
  distortion, fit proxies, cliff detection, bootstrap correlations, and simple
  predictor comparisons.
- Selected-budget oracle fine-tuning/evaluation and oracle-gap analysis.
- The central analysis table and go/no-go smoke-test report.

## Deliberately excluded

- New pruning/ranking criteria, learned compression metrics, geometry
  regularization, CFM, and a full OFA-style NAS/search stack.
- Random intermediate-width sampling during anchor-only training.

