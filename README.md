# Compression geometry smoke test

Research code for the question: **can Wasserstein geometry of a shared family of
width-compressed networks explain performance degradation at unseen budgets?**

The baseline is intentionally narrow: CIFAR-style ResNet-18, nested channel
prefixes, four training anchors, in-place full-width knowledge distillation, and
no geometry regularizer. Dense intermediate widths are genuinely unseen during
optimization.

## Integrity guarantees

- `training.py` iterates only over `compression.train_widths`, which the config
  loader validates as exactly `[0.25, 0.50, 0.75, 1.00]`.
- Every evaluation width owns a BN-statistics bank. Unseen banks are populated only
  by explicit post-training calibration (no gradient update).
- Feature subsets are selected once per seed. Every saved feature file contains
  features, labels, sample IDs, budget, and normalization mode; extraction aborts
  if IDs/order differ across budgets.
- Channels are exact nested prefixes of size `floor(alpha * C)`.
- `ID_fit_proxy` is labeled as a proxy against the shared full-width model, never
  as a true independent reference.

The implementation/reuse decision is documented in [docs/repo_audit.md](docs/repo_audit.md),
with source and license attribution in [THIRD_PARTY.md](THIRD_PARTY.md).

## Quick smoke test (no download)

```bash
python -m scripts.run_experiment --config configs/smoke.yaml
```

This uses deterministic `torchvision.datasets.FakeData` to exercise training,
dense evaluation, feature extraction, geometry analysis, plots, and reporting.
It validates plumbing only; its scientific numbers are meaningless.

## CIFAR-100 experiment

```bash
python -m scripts.run_experiment --config configs/cifar100.yaml
```

The default scientific config runs three seeds and selected unseen-budget oracles
at `0.30, 0.40, 0.60, 0.80`. Adjust epochs or batch sizes in a copied config for
available hardware. CIFAR data downloads only when `dataset.download: true`.

## Modal T4 smoke test

Use [modal_smoke_test.ipynb](modal_smoke_test.ipynb) for the guided notebook or:

```bash
python -m pip install "modal>=1.0"
modal setup
modal run modal_smoke.py --run-name modal-smoke-seed0-001
```

This real-CIFAR smoke configuration runs one seed for 20 epochs, disables oracles,
and evaluates only Sliced Wasserstein plus the Euclidean-mean control. See
[docs/MODAL_SMOKE_TEST.md](docs/MODAL_SMOKE_TEST.md) for persistence, download,
failure-check, and GitHub-clone instructions.

To recompute one seed's analysis without retraining:

```bash
python -m scripts.analyze --config configs/cifar100.yaml --seed 0
```

## Main outputs

Each `OUTPUT/seed_N/` contains:

- `training_metrics.csv`: loss, CE, KD, accuracy, LR, FLOPs, and active parameters
  at every anchor/epoch.
- `results/budget_metrics.csv`: required dense budget metrics.
- `features/features_budget_025.pt` through `features_budget_100.pt`.
- pairwise Wasserstein, width-distance, and FLOPs-distance matrices (`.npy`/`.csv`)
  plus `wasserstein_heatmap.png`.
- local sensitivity, cliff candidates, the central analysis table, and regression
  comparisons.
- selected-width oracle checkpoints/features/metrics when enabled.

The experiment root contains `central_analysis_all_seeds.csv`, pooled bootstrap
correlations, the resolved config, and `reports/smoke_test_report.md` with the five
required go/no-go answers.

FLOPs use the explicit convention `2 FLOPs = 1 multiply-accumulate`; THOP supplies
the operation tracing and local hooks account for active dynamic slices.

## Tests

```bash
pytest
```

Sliced Wasserstein uses SciPy's established exact 1-D Wasserstein primitive. The
same interface also supports `euclidean_mean`, `cosine`, `gaussian_w2`, and `mmd`;
`sinkhorn` is available when the optional POT package is installed.
