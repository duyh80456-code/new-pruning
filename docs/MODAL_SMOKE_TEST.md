# Modal CIFAR-100 smoke test

This workflow runs the existing CE + KD baseline on one Modal T4. It uses one
seed, 20 epochs, four training anchors, 16 evaluation widths, 2,000 fixed feature
samples, Sliced Wasserstein, and only the Euclidean-mean control. It does not run
oracles, CFM, geometry regularization, intermediate-width training, or tuning.
The four per-width objectives are averaged before backpropagation.

## Notebook workflow

Open `modal_smoke_test.ipynb` in Jupyter or Colab and execute cells in order. The
notebook contains:

```python
GITHUB_TOKEN = ""  # optional: paste a token here only if the repository requires it
```

The token is used only as an in-memory Git HTTP authorization header and is never
inserted into the clone URL, printed, written to Modal, or saved by the project.
For a public repository, leave it empty. Do not save/share the notebook after
pasting a token; using a short-lived, least-privilege token is recommended.

The notebook installs the local Modal client, clones:

```text
https://github.com/duyh80456-code/new-pruning.git
```

authenticates with Modal, launches `run_smoke.remote(...)`, and renders only the
compact returned tables/matrix. Feature tensors remain on the persistent Volume.

## CLI workflow

From any clone of the repository:

```bash
python -m pip install "modal>=1.0"
modal setup
modal run modal_smoke.py --run-name modal-smoke-seed0-001
```

Use a new `run-name` for each execution. The runner refuses to mix output into a
non-empty prior run directory.

## Persistent data and outputs

Both CIFAR-100 and experiment artifacts are stored in the Modal Volume:

```text
new-pruning-smoke-data
├── data/cifar100
└── outputs/<run-name>
```

Download one run recursively with:

```bash
modal volume get new-pruning-smoke-data outputs/<run-name> modal-results/<run-name>
```

Inspect without downloading:

```bash
modal volume ls new-pruning-smoke-data outputs/<run-name>
```

The run retains the checkpoint, budget metrics, feature files, Wasserstein matrix,
geometry CSVs, plots, the base smoke report, and
`reports/modal_smoke_summary.md`. Modal Volume writes are explicitly committed
after CIFAR ingestion and after successful analysis.

## Scientific and failure checks

Before training, the remote runner prints the GPU, CUDA/PyTorch versions, seed,
epochs, dataset mode, and width lists. It exits early or fails the run if:

- CUDA/T4 execution is unavailable;
- the config is not real CIFAR-100, one seed, 20 epochs, and exact anchor-only
  training;
- CIFAR-100 cannot be downloaded/read;
- the checkpoint is absent;
- feature IDs/order differ across widths or features are degenerate;
- the Wasserstein matrix contains invalid values or is not symmetric;
- active FLOPs do not increase monotonically with width.

The baseline currently has no AMP option, so this wrapper reports AMP as disabled
instead of changing training behavior. The final interpretation is descriptive:
`PROMISING SIGNAL`, `WEAK / INCONCLUSIVE SIGNAL`, or `NO OBVIOUS SIGNAL`. It is
not a statistical or novelty claim.
