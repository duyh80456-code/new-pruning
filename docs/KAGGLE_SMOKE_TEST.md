# Kaggle CIFAR-100 smoke test

Use `kaggle_smoke_test.ipynb` for a one-seed, 20-epoch scientific smoke test on
real CIFAR-100. The notebook reuses the repository's existing training,
evaluation, feature-extraction, and geometry entrypoint.

## Kaggle setup

Before running the notebook:

1. Create a Kaggle Notebook and upload/import `kaggle_smoke_test.ipynb`.
2. In Notebook settings, enable a GPU accelerator.
3. Enable Internet so CIFAR-100 and the GitHub repository can be downloaded.
4. Open **Add-ons → Secrets**, create a secret named exactly `github_token`, and
   attach it to the notebook. Use a short-lived read-only GitHub token.

The notebook retrieves it with:

```python
from kaggle_secrets import UserSecretsClient

github_token = UserSecretsClient().get_secret("github_token")
```

The token is passed through a temporary `GIT_ASKPASS` helper. It is not inserted
into the repository URL or printed, and the helper file is removed after clone.

## Experiment scope

- Dataset: real CIFAR-100; FakeData is asserted off.
- Model: slimmable CIFAR ResNet-18.
- Seed: 0.
- Epochs: 20.
- Width-loss reduction: mean, preventing the effective optimizer step from
  scaling with the number of anchors.
- Training widths: 0.25, 0.50, 0.75, 1.00 only.
- Evaluation widths: 0.25 to 1.00 in increments of 0.05.
- Geometry: 128 deterministic Sliced Wasserstein projections.
- Control: Euclidean distance between feature means only.
- Fixed feature subset: 2,000 validation samples.
- Disabled: oracle, CFM, geometry regularization, arbitrary-width sampling,
  hyperparameter search, and multi-seed execution.

The existing baseline does not expose AMP, so the notebook does not modify its
numerical behavior merely to add mixed precision.

## Outputs

Each execution uses a UTC timestamp and writes to:

```text
/kaggle/working/new-pruning-outputs/kaggle-smoke-<timestamp>/
```

It preserves the checkpoint, training/budget metrics, 16 feature files,
Wasserstein and resource matrices, plots, central analysis, control comparison,
cliff candidates, base smoke report, and `reports/kaggle_smoke_summary.md`.

The final notebook cell creates:

```text
/kaggle/working/kaggle-smoke-<timestamp>.zip
```

Download it from the notebook's Files panel or use **Save Version** to persist it
as notebook output.

## Failure checks

The notebook stops if CUDA is unavailable, the experiment config violates the
anchor-only protocol, CIFAR-100 cannot load, the checkpoint is missing, feature
IDs/order differ, features are non-finite/degenerate, the Wasserstein matrix is
invalid, or FLOPs do not increase with width.

The final label—`PROMISING SIGNAL`, `WEAK / INCONCLUSIVE SIGNAL`, or
`NO OBVIOUS SIGNAL`—is descriptive triage for one short run, not a significance,
novelty, or paper-level claim.
