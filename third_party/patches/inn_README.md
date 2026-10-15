# INN (official code) on CIFAR-100 / CIFAR ResNet-50, converted and fine-tuned, under our protocol

Paper: Solodskikh et al., "Integral Neural Networks", CVPR 2023 (supplementary: Appendix D,
experiment details). Upstream: https://github.com/TheStageAI/TorchIntegral
Pinned commit: `84c5e89047f1d47bbf6c62065aa223d997f9b5e7` (HEAD, 2024-12-02, "Merge pull request
#39"). Our changes: `inn.patch` (= `git diff` against that commit, new files `cifar_resnet.py`,
`protocol.py`, `train_base.py`, `train_inn.py` included), applied with `git apply`. Checked: applies
cleanly to a fresh clone from GitHub and reproduces the tested tree byte for byte (with
`core.autocrlf=false`; on Windows with `autocrlf=true`, `git apply` writes CRLF files, harmless).

## Method
INN does not train a slimmable network from scratch. It converts a **trained ordinary network**
into an integral one and fine-tunes that (the paper's "INN-init", pipeline A, Algorithm 1). So
there are two phases, each its own command:

1. `train_base.py`: a plain CIFAR ResNet-50 (ordinary `torch.nn` modules) trained with our recipe.
   It is also our *individual network at width 1.0*: it prints one protocol line at 1.00.
2. `train_inn.py`: loads that checkpoint and converts it exactly as the released examples do,
   `IntegralWrapper(init_from_discrete=True, fuse_bn=True)(model, [1,3,32,32], continuous_dims,
   discrete_dims)`:
   - **BatchNorm fused** into the preceding conv (their default `fuse_bn=True`, and Appendix D:
     "batch normalization layers were fused into parent convolution"). All 53 BNs are fused (each
     follows a conv with a single user), so the INN has **no BatchNorm at any width**. Each conv
     gets a bias instead: 23,678,692 parameters at 1.00 (= 23,705,252 - 53,120 BN + 26,560 bias).
   - **TSP channel permutation**: `NOptPermutation` (2-opt on the L1 total variation, 100 random
     restarts, threshold 0.001, their defaults) per group, over every tensor in the group, so the
     network function is unchanged. On BX's weights it cut the summed variation of the 37 groups
     from 20,429 to 6,971 (34 %).
   - **Continuous parametrization**: `GridSampleWeights1D/2D` (bicubic `grid_sample`,
     `align_corners=True`, `scale=1`: the stored tensor has the discrete weight's size, no extra
     parameters; `optimize_iters=0`, their default). 107 tensors are parametrized.
   - **Trapezoidal quadrature** along each weight's input-channel dimension (when it has > 3
     channels), scaled by their `train_volume` so that the full-size weights are the discrete ones
     exactly; with n of C points the sampled weights are multiplied by about (C-1)/(n-1).
   - Continuous dims: `standard_continuous_dims(model)` (every Conv2d/Linear weight along out and
     in channels, biases along out), minus `discrete_dims = {conv1.weight: [1], fc.weight: [0],
     fc.bias: [0]}` (RGB input of the stem, the 100 classes; `imagenet.py` keeps `fc` discrete
     the same way). Their examples make only part of the network continuous (`stage3` of NIN,
     `layer4` of ResNet-18); here **every layer** is, as the protocol slims every layer.

   Checked: the converted INN at width 1.00 is the original network. On BX's width-1.0 network
   (loaded into `cifar_resnet.py`, its calibrated running stats fused): loss 1.4216 -> 1.4215,
   top-1 error 0.2204 -> 0.2205, max |logit difference| 0.025 on a logit scale of 70 (fp32
   rounding of the fused BN and of `grid_sample`).

**Integral groups and widths.** The tracer finds 37 groups (sets of tensor dimensions that must
share one integration grid), all without subgroups:
- 32 *inner* groups, two per bottleneck: conv1 out = conv2 in (64/128/256/512 by stage), and
  conv2 out = conv3 in (same sizes);
- 5 *residual-stream* groups (they contain the residual `add`): the stem output 64 (stem conv out,
  layer1.0 conv1 in and downsample in), then each stage's stream 256 / 512 / 1024 / 2048 (every
  conv3 out and the downsample out of the stage, every conv1 in of its later blocks, the next
  stage's first conv1 and downsample in; for 2048 also `fc` in).

So each group is exactly one channel count of our `models/us_resnet.py`, and the protocol width
maps to the group size by the same rounding: a group of C channels gets **n = make_divisible(C*w)**
points (divisor 8). At the evaluation `train_inn.py` resamples every group to n, takes
`get_unparametrized_model()` (an ordinary network) and asserts that every one of its 54 conv /
linear weights has `make_divisible(C*w)` channels on both sides (except stem input and classes),
as `us_resnet` at that width (e.g. stem 16, pooled 512 at w = 0.25; 1,523,716 parameters).

**Width sampling.** Their default draws each group's size independently (`RandomLinspace` with
one `UniformDistribution` per group; `imagenet.py` sets 338-512 on two groups). Here one width
w ~ U(0.25, 1.0) is drawn per step (Python `random`) and every group draws exactly
make_divisible(C*w) points (a one-point `UniformDistribution(n, n)` through their
`reset_distribution`), so each step trains one uniform-width network. One forward/backward per
step, plain cross-entropy (Algorithm 1; no sandwich, no distillation).

**Fine-tuning recipe (the authors').** Adam, lr 1e-4, batch 128, 50 epochs, lr x 0.33 in steps:
- paper, Appendix D: "All training experiments were done using Adam optimizer ... piecewise
  constant monotone learning rate scheduling starting from 1e-4 with the adjustment on a plateau
  (10 epochs without loss function decrement) by 0.33 multiplier ... INNs initialized from
  pre-trained discrete networks were trained for 50 epochs (batch size 128) in the case of Cifar10
  dataset and 100 epochs (256 batch size) in the case of Imagenet";
- code, `examples/classification/imagenet.py`: `Adam(lr=1e-4, weight_decay=1e-8)`,
  `MultiStepLR` x 0.33 at epochs 10, 20, 30, 40 (stepped per batch);
- `nin_cifar.py` is the other pipeline (grid tuning at a fixed size: Adam 1e-3, 10 epochs), not
  this one; the README has no recipe.

Default here: Adam 1e-4, weight decay 1e-8, batch 128, 50 epochs, x 0.33 at epochs 10/20/30/40
(the code's schedule, deterministic). `--lr_schedule plateau` gives the paper's wording instead
(`ReduceLROnPlateau`, factor 0.33, patience 10, on the epoch's mean **training** loss; the paper's
loss is presumably the validation loss, i.e. the test set on CIFAR, which we do not use). No
gradient clipping (none upstream; `--grad_clip` exists), fp32.

**Evaluation.** 16 widths 0.25, 0.30, ..., 1.00, each resampled from the same INN, on the 10k
test set. BatchNorm does not survive the conversion, so there is nothing to recalibrate: the run
prints `(no BatchNorm layers: evaluated without recalibration)` and tests directly
(`--recal_batches` only acts if BN layers exist).

## Setup (Kaggle or local)
```bash
git clone https://github.com/TheStageAI/TorchIntegral && cd TorchIntegral
git checkout 84c5e89047f1d47bbf6c62065aa223d997f9b5e7
git apply /path/to/inn.patch
pip install cython                        # with a torch constraint on Kaggle; scipy is preinstalled
python setup.py build_ext --inplace       # builds the Cython TSP solver (Linux / macOS)
```
Do not install `requirements.txt` (pins torch 2.0.1) and do not `pip install .` (its build
isolation lacks numpy). catalyst and pytorchcv are not needed: they are used only by the examples,
and the patch's own loop replaces catalyst. If the solver is not built, `train_inn.py` runs
`setup.py build_ext --inplace` itself on first use (Linux/macOS), so a notebook needs no extra
cell as long as `cython` is installed. Run from the repo root (`torch_integral` is imported from
there). `--data_dir` is the folder that *contains* `cifar-100-python/` (read-only Kaggle input
works). Loader workers: env `INN_WORKERS` (default 2). One GPU per run (`CUDA_VISIBLE_DEVICES`).

Windows: upstream `setup.py` refuses Windows, and setuptools' MinGW path is broken
(`customize_compiler` with no CC); the local test built the solver by hand with MinGW-w64 g++ 13.2
(ucrt): `cython -3 --cplus torch_integral/tsp_solver/solver.pyx`, then
`g++ -std=c++11 -O3 -shared -DMS_WIN64 -I<python include> -I<numpy include> solver.cpp
-L<python>/libs -lpython310 -static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive
-lwinpthread -Wl,--no-whole-archive -o torch_integral/tsp_solver/solver.cp310-win_amd64.pyd`.
Not needed on Kaggle.

Tested with Python 3.10.20, torch 2.11.0+cu128, torchvision 0.26.0, numpy 2.2.6, scipy 1.15.3,
Cython 3.3.0 (the only package installed for this, into `fimaq_apb`) on an RTX 5060 Ti 16 GB.
`torch.fx` tracing of the model by `IntegralTracer` works unchanged on torch 2.11. The Linux build
(`setup.py`, `-Ofast`) was not run here (no Linux machine); the patched `.pyx` compiles with
Cython 3.3.

For `scripts/build_external_notebooks.py` the `UPSTREAMS` entry would be
`('https://github.com/TheStageAI/TorchIntegral.git', '84c5e89047f1d47bbf6c62065aa223d997f9b5e7',
'TorchIntegral', 'inn.patch', 'cython')` and `WORKER_ENV` should carry `INN_WORKERS`. The two
phases are sequential on one card (chain them, e.g. `bash -c "python -u train_base.py ... &&
python -u train_inn.py ..."`), and give them **different labels or separate `run_pinned` calls**:
`VAL_LINE` ignores the tag, so under one label the INN's 1.00 line overwrites the base's.

## Commands (one T4 per branch)
```bash
D=/kaggle/input/cifar100          # contains cifar-100-python/
CK=/kaggle/working/ckpt
# phase 1: base network, our recipe (100 epochs, prints the width-1.00 line)
CUDA_VISIBLE_DEVICES=0 INN_WORKERS=2 python -u train_base.py --data_dir $D --seed 1995 \
  --ckpt_dir $CK/inn_base --log_prefix inn_base
# phase 2: conversion + the authors' fine-tuning (50 epochs), prints the 16 protocol lines
CUDA_VISIBLE_DEVICES=0 INN_WORKERS=2 python -u train_inn.py --data_dir $D --seed 1995 \
  --base_ckpt $CK/inn_base/last.pt --ckpt_dir $CK/inn_r50 --log_prefix inn_r50
```
Defaults of `train_base.py` = our recipe: 100 epochs, batch 256, Nesterov SGD momentum 0.9, lr 0.2,
per-step linear warm-up over 5 epochs then per-step cosine to 0 (our `lr_schedule_per_iteration`
formula), wd 5e-4 on conv/linear weights only, grad clip 5, 20-batch BN recalibration before the
width-1.00 test (`--recal_batches 0`: trained running stats). Defaults of `train_inn.py` = the
recipe above. Extra flags: `--eval_converted` (also test the 16 widths right after conversion,
before any fine-tuning, tagged `inn_r50_conv`, `-1/0`), `--tsp_iters` (2-opt restarts, default
100), `--min_width/--max_width`, `--lr_schedule plateau`.

Output lines (tab separated, elapsed = seconds since the process started):
`[inn_r50] 15912.3s	val	0.25	-1/50: loss: 1.2345, top1_error: 0.4321, top5_error: 0.1234`
(`-1/50` = the fine-tuning epochs; the base prints `-1/100`). Before each line `train_inn.py`
prints the resampled network's parameter count and the channel check.

Evaluation only, from the resume checkpoint: the same command + `--eval_only` (raises if there is
no checkpoint). ~40 s for 16 widths here.

Resume. After every epoch `{ckpt_dir}/last.pt` is written atomically (`.tmp` + `os.replace`):
model, optimizer state (SGD momentum / Adam moments), the plateau scheduler's state if used, next
epoch, and Python/NumPy/torch/CUDA RNG states. The LR schedules of both phases are pure functions
of the epoch (and step), so nothing else is needed. If `last.pt` exists the identical command
resumes (`Resumed from ... at epoch N`). `train_inn.py` also writes `last.pt` at epoch 0 right
after the conversion; on resume it rebuilds the INN structure from the base checkpoint **without**
the permutation search (`init_from_discrete=False`, as `imagenet.py --checkpoint` does; 0.5 s) and
loads the trained, permuted parametrization. `--stop_at_epoch N` exits cleanly after N epochs (no
eval); copy `last.pt` (and for phase 2 the base `last.pt`) into the new session and rerun.
Granularity is one epoch.

## Timing (RTX 5060 Ti, fp32, 2 workers, steady state over steps 4-10 of each epoch)
| phase | batch | s/step | steps/epoch | per run | peak mem |
|---|---|---|---|---|---|
| 1 base | 256 | 0.290 (mean, both epochs) | 196 | ~57 s/epoch, ~1.6 h | 9.37-9.46 GiB |
| conversion (CPU) | | | | 266 s (smoke base), 328 s (BX weights) | |
| 2 INN | 128 | 0.182 mean, 0.167-0.174 median | 391 | ~71 s/epoch, ~1.0 h | 2.03-2.04 GiB |

Phase 1 uses `cudnn.benchmark = True` (fixed shapes; its peak includes cuDNN trial workspaces);
phase 2 sets it False (a new width almost every step; LCS measured ~20 s of autotuning per new
shape). Phase 2 is cheap per step: one pass at one width (mean width 0.625), no BN, but each step
resamples all 107 weight tensors through `grid_sample`. Our US-Net R50 is ~0.59 s/step here and
10.7 h on a T4 (factor ~3.3); by that factor expect on a T4 ~5 h for phase 1 and ~3.5 h for phase 2,
plus ~5-10 min of conversion on Kaggle's CPU: ~9 h in all, tight for one 12 h session, so plan a
`--stop_at_epoch` (e.g. phase 1 in one session, phase 2 in the next). Not measured on a T4.

The 2-opt solver is the release's algorithm with typed memory access (item 6): one restart at
n = 1024 took 3.5 s with the released code and 0.08 s patched (0.76 s at n = 2048); with the
released code 100 restarts on the 2048 / 1024 groups alone would take hours.

## Smoke test actually run (local GPU)
Phase 1: run 1 `--epochs 2 --warmup_epochs 1 --max_iters_per_epoch 10 --stop_at_epoch 1`, run 2
the same without `--stop_at_epoch`. Phase 2 from that checkpoint: run 1 `--epochs 2
--max_iters_per_epoch 10 --stop_at_epoch 1 --eval_converted` (crashed first in its own channel
check, which read `out_channels` instead of the weight; fixed, then rerun from the epoch-0
checkpoint it had saved, which also tested the rebuild-without-permutation path), run 2 the same
without `--stop_at_epoch`.
```
base run1: CIFAR ResNet-50: 23,705,252 parameters
           Epoch 0 train time 36.6s, 10 steps, loss 4.6952, ..., lr now 0.01024, steady 0.290 s/step ..., peak mem 9.37 GiB
           Stopping after epoch 0 (--stop_at_epoch 1); rerun to resume.
base run2: Resumed from .../base at epoch 1
           Epoch 1 train time 36.7s, 10 steps, loss 8.5347, ..., lr now 0.19871, steady 0.290 s/step ..., peak mem 9.46 GiB
           (BN recalibrated per width: 53 layers, 20 batches of 256, cumulative)
[smoke_inn_base] 80.3s	val	1.00	-1/2: loss: 6.4787, top1_error: 0.9900, top5_error: 0.9493
inn run1:  Conversion took 265.9s / 37 integral groups ... / BatchNorm layers left after conversion: 0
           Saved checkpoint .../inn/last.pt (next epoch 0)
inn run1b: Conversion took 0.5s (no permutation: weights come from the checkpoint)
           Resumed from .../inn at epoch 0
           width 0.25: 1,523,716 parameters, channel counts = make_divisible(C*w) in all 54 conv/linear layers
           (no BatchNorm layers: evaluated without recalibration)
[smoke_inn_r50_conv] 10.9s	val	0.25	-1/0: loss: 36246819327.1808, top1_error: 0.9900, top5_error: 0.9500
... (16 lines) ...
[smoke_inn_r50_conv] 50.2s	val	1.00	-1/0: loss: 4299.3833, top1_error: 0.9900, top5_error: 0.9500
           Epoch 0 train time 10.6s, 10 steps, loss 475844468.9372, mean width 0.584, lr 0.0001, steady 0.182 s/step ..., peak mem 2.03 GiB
           Stopping after epoch 0 (--stop_at_epoch 1); rerun to resume.
inn run2:  Resumed from .../inn at epoch 1
           Epoch 1 train time 10.4s, 10 steps, loss 3850665.0706, mean width 0.646, lr 0.0001, steady 0.182 s/step ..., peak mem 2.04 GiB
[smoke_inn_r50] 22.1s	val	0.25	-1/2: loss: 6125913582.7968, top1_error: 0.9900, top5_error: 0.9500
... (16 lines) ...
[smoke_inn_r50] 61.5s	val	1.00	-1/2: loss: 889.3367, top1_error: 0.9900, top5_error: 0.9499
```
The smoke base (20 steps at up to lr 0.2) has exploded running statistics (loss 4298.92 with
them, 6.48 after recalibration), so the huge losses are that base, fused into a BN-free network,
not the INN code: its INN at 1.00 reproduces 4298.92 as 4299.38. The conversion check on BX's
trained weights (Method) is the meaningful one. Not run: a full epoch, `--eval_only`, the plateau
schedule, the Linux build.

Conversion only (BX's width-1.0 network, no fine-tuning; a US-Net-trained network, so only
indicative): 1.00 -> 77.95 %, but every narrower width is near chance, 0.95 -> 7.6 %, 0.80 -> 7.3 %,
0.50 -> 1.4 %, 0.25 -> 1.0 % (loss 640). Resampling a converted BN-free ResNet-50 is not
usable by itself; everything below 1.00 must come from phase 2.

## Changes and deviations
Code (all in `inn.patch`):
1. `cifar_resnet.py` (new): plain CIFAR ResNet-50 = our `us_resnet` at width 1.0 (3x3 stride-1
   stem, no max-pool, bottlenecks [3,4,6,3] with the stride on the 3x3, 1x1 conv + BN projection
   where stride or width changes, ReLU after the sum, average pool, linear with bias; our
   initialisation: convs N(0, 2/(k^2 C_out)), BN 1/0, linear N(0, 0.01), bias 0); torchvision
   module names; 23,705,252 parameters, as ours. `make_divisible` copied from ours.
2. `train_base.py` (new): phase 1, our recipe (above), plain loop.
3. `train_inn.py` (new): phase 2 in a plain loop instead of catalyst's `SupervisedRunner`;
   conversion call as in the examples; the group description, the width sampling, the
   resample-and-check evaluation, resume, `--eval_converted`, auto-build of the solver.
4. `protocol.py` (new): CIFAR-100 loaders (RandomCrop 32 pad 4 + HFlip, mean/std
   (0.5071,0.4865,0.4409)/(0.2673,0.2564,0.2762), `INN_WORKERS`, persistent workers; training
   loader keeps the last partial batch, as ours; recalibration and test batches of 256), checkpoint
   save/load, steady step timer, BN recalibration (cumulative, 1/(i+1)), evaluation and the
   protocol lines (as LCS's/DS-Net's `protocol.py`).
5. `.gitignore`: the in-place build outputs (`solver.cpp`, `*.so`, `*.pyd`).
6. `torch_integral/tsp_solver/solver.pyx`, speed only: the distance matrix (the release's
   Python double loop of `np.mean(np.abs(t[i] - t[j]))`) is computed with `torch.cdist(p=1) / D`
   on the tensor's device; `SymmMatrix` reads its packed array through a typed memoryview, and
   `_calculate_path_dist` takes a typed array. Same algorithm, same random restarts: checked
   against the released solver on random tensors (n = 128, 384, 4 tensors, 3 restarts, same seed):
   identical routes, path lengths within 2e-4 (of ~1,700), distance entries within 3e-6. Kept as released:
   2-opt's gain test (`before` counts the edge (before_start, start) twice instead of adding
   (end, after_end)), the relative threshold, 100 restarts.
7. No change to the rest of `torch_integral` (tracer, grids, parametrizations, quadrature,
   wrapper); no `torch.fx` fix was needed on torch 2.11.

Method and recipe (ours vs paper / upstream):
8. Base network: trained by us with our recipe (100 epochs SGD) on CIFAR-100. The paper converts
   pre-trained networks (pytorchcv on CIFAR-10, torchvision on ImageNet).
9. Whole network continuous (37 groups, stem output and all residual streams included); the
   examples make one stage continuous. Classifier classes and RGB input discrete.
10. One width per step for all groups, w ~ U(0.25, 1.0), sizes make_divisible(C*w); upstream
    samples each group independently in a per-group range (their examples: 66-100 % of two groups).
    The paper never resamples a whole network to 25 %.
11. Evaluation at the 16 protocol widths through `get_unparametrized_model()` (an ordinary network
    with exactly our channel counts); upstream evaluates the resized INN directly (same weights).
12. Fine-tuning: Adam 1e-4, wd 1e-8, batch 128, 50 epochs, x 0.33 at 10/20/30/40 (code) rather
    than on a plateau (paper; `--lr_schedule plateau` on training loss). Plain CE, no clip, fp32.
13. Data: CIFAR-100 with our augmentation and normalisation (their CIFAR-10 example: HFlip only,
    CIFAR-10 mean/std).
14. BN fused, so no BN at any width and no recalibration (our other rows recalibrate BN per width).
15. `cudnn.benchmark` True in phase 1, False in phase 2 (upstream does not set it).
16. Seeds: `--seed` seeds Python/NumPy/torch (init, data order, the 2-opt restarts through
    `np.random`, the width draws).

**Not comparable to the 100-epoch from-scratch rows as is:**
- Total training is **150 epochs** (100 SGD epochs of the base + 50 Adam epochs of the INN), and a
  different optimizer in the second part. The INN epochs are cheaper per step (one pass at one
  width) than a US-Net step (four passes), so the compute is not simply 1.5x either.
- The INN's width 1.00 is the fine-tuned network, not the base; the base's 1.00 line is our
  *individual network at width 1.0* reference (100 epochs, same recipe as the solo rows).
- The INN has no BatchNorm at any width; every other row has BN, recalibrated per width.

## Uncertain
- **Whether narrow widths are viable.** After conversion alone, every width below 1.00 is near
  chance on a well-trained network (BX test above), so 0.25-0.95 must be learned in 50 epochs of
  Adam at 1e-4 (then lower), BN-free, from that starting point. The paper itself notes INNs trained
  from scratch lag, "probably caused by absence of batch-normalization layers", and never trains a
  range as wide as 0.25-1.00 over the whole network. I expect the narrow end to be weak; the smoke
  (20 steps on a 20-step base) says nothing about it. A cheap first look on Kaggle: phase 2 with
  `--stop_at_epoch 5` and `--eval_only` on that checkpoint.
- Schedule choice (code's milestones vs paper's plateau) and weight decay 1e-8 (code only; the
  paper names none).
- T4 timing is extrapolated, not measured; the Linux build of the solver was not run here.
- 2-opt restarts are random, so the permutation (and so the INN) depends on the seed.
