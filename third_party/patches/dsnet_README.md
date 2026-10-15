# DS-Net (official code) on CIFAR-100 / CIFAR ResNet-50, Stage I supernet, under our protocol

Papers: Li et al., "Dynamic Slimmable Network", CVPR 2021 (oral), arXiv 2103.13258; journal
extension DS-Net++, TPAMI 2023, arXiv 2109.10060. Upstream: https://github.com/changlin31/DS-Net
Pinned commit: `15cd3036970ec27d2c306014344fd50d9e9b888b` (HEAD, 2021-12-31, "set default ensemble
false"). Our changes: `dsnet.patch` (= `git diff` against that commit, new files `protocol.py`,
`dyn_slim/models/dyn_slim_resnet.py`, `configs/cifar100_ds_resnet50_uniform.yml` included), applied
with `git apply`. Checked: applies cleanly to a fresh checkout and reproduces the tested tree byte
for byte.

## Method
Their Stage I supernet training, `dyn_slim/apis/train_slim.py` with
`configs/mobilenetv1_bn_uniform.yml` as released, kept as it is:
- Sandwich sample list `['largest', 'uniform', 'uniform', 'smallest']` every step, one
  forward/backward per sample (upstream already backpropagates per sample).
- In-place bootstrapping (IB): the largest learns from the label (CE); the two uniform widths
  and the smallest learn from the soft target softmax(EMA(x)), the EMA supernet at its largest
  width, with timm's `SoftTargetCrossEntropy` (T = 1). EMA = their `ModelEma` (all parameters and
  buffers, decay 0.997, updated after every optimizer step, kept in eval mode). `ensemble_ib`
  false, as in their released config.
- Their slimmable ops and BN (`DSConv2d`, `DSpwConv2d`, `DSBatchNorm2d` with shared affine and
  private running statistics per channel choice, `DSLinear`, `DSAdaptiveAvgPool2d`), their
  `efficientnet_init_weights`, their per-sample reseeding, their "drop parameters whose gradient is
  exactly zero before `optimizer.step()`" filter, timm's SGD and cosine scheduler.
- Stage II (the gate) is not run. The supernet is evaluated at fixed uniform widths.

Width space: `ds_resnet50_cifar` (new, built from their ops) gives every slimmable layer the same
sixteen channel choices, `make_divisible(C * w)` for w = 0.25, 0.30, ..., 1.00, so channel choice i
is width w_i in every layer at once (stem, all convs and BNs, shortcuts, pooled feature,
classifier input). Their 'uniform' mode then draws one choice for all layers:
`random.randint(1, 15)`, i.e. 0.30..1.00, exactly as `DSNet.set_mode` draws `randint(1, 13)` over
its 14 choices (the smallest is index 0 and is trained separately as 'smallest'). So each step
trains 1.00 (label), 0.25 (soft target) and two widths drawn uniformly from {0.30, ..., 1.00}
(soft target). The sixteen-width evaluation uses the same sixteen choices.

H-IB (`--hib`, our addition from DS-Net++, not in their code; default off, so the default run is
DS-Net as released): every non-largest sampled width (both uniform ones and the smallest) also
gets the hierarchical in-place bootstrapping loss of DS-Net++ eq. (33) at the last block only (the
paper's CNN setting: "only calculated with the last blocks' outputs"):
`L_HIB = (1/C) * || phi(f_k(x)) - f_L(x) ||^2`, averaged over the batch, where f_k is the
globally pooled output of the last bottleneck at the sampled width (c channels), f_L the pooled
C = 2048-d feature of the EMA supernet at its largest width (no gradient), and phi a slimmable
2-layer MLP predictor of the online supernet, eq. (19): `W2 relu(W1[:, :c] f)` with
W1: C_mlp x 2048 (input sliced to c, their `DSLinear`), W2: 2048 x C_mlp, C_mlp = 2048 (the
paper gives no value; `--hib_hidden`). Eq. (34) with the paper's lambda_IB = lambda_HIB = 0.5:
each non-largest width's loss is `0.5 * L_IB + 0.5 * L_HIB`, so the soft-target term is halved
relative to the default run; the largest keeps its CE unscaled. No feature normalisation (the
paper: "simplistic L2 distance"). The predictor adds 8,392,704 parameters (training only; the
supernet itself is unchanged).

Parameter count of `ds_resnet50_cifar` (100 classes): **23,705,252**, the CIFAR ResNet-50 count
(23,705,152 conv/BN/classifier weights + 100 classifier biases). With `--hib`: 32,097,956.

## Setup (Kaggle or local)
```bash
git clone https://github.com/changlin31/DS-Net && cd DS-Net
git checkout 15cd3036970ec27d2c306014344fd50d9e9b888b
git apply /path/to/dsnet.patch
```
(On Windows with `core.autocrlf=true`, `git apply` writes CRLF files; harmless, but the
byte-for-byte check above was made with `core.autocrlf=false`.) For
`scripts/build_external_notebooks.py` the `UPSTREAMS` entry would be
`('https://github.com/changlin31/DS-Net.git', '15cd3036970ec27d2c306014344fd50d9e9b888b',
'DS-Net', 'dsnet.patch', 'timm')`; keep the extra-package field non-empty: if DS-Net is a notebook's only upstream, the
notebook's pip call gets no requirement at all and fails; and `timm` under the torch constraint is
a no-op when Kaggle already has it.
Do not install their requirements (README pins timm 0.3.2, torch 1.2+). Needed: torch,
torchvision, timm (any recent; preinstalled on Kaggle, else `pip install timm` with a torch
constraint), PyYAML. No apex (AMP stays off: fp32). Tested with Python 3.10.20, torch
2.11.0+cu128, torchvision 0.26.0, timm 1.0.27, numpy 2.2.6, PyYAML 6.0.3 on an RTX 5060 Ti 16 GB.

timm compatibility (the release targets timm 0.3.2): `timm.data.Dataset` is imported as
`ImageDataset` when the old name is missing (only used by the ImageNet path);
`convert_splitbn_model` from `timm.layers` when `timm.models` lacks it; `Swish`, `sigmoid`,
`SelectAdaptivePool2d` from `timm.layers` with a fallback to `timm.models.layers`;
`register_model` from `timm.models`. `train_slim.py`'s `from timm.utils import *` is replaced by
the three names it uses (`AverageMeter, reduce_tensor, accuracy`): under timm >= 0.9 the star
import also binds the submodule `timm.utils.random` over Python's `random`, and the first
`random.seed(seed)` crashes (measured: `AttributeError: module 'timm.utils.random' has no
attribute 'seed'`). Kept from timm, unvendored: `create_model`/registry, `create_optimizer`
(legacy wrapper of `create_optimizer_v2`; 'sgd' = Nesterov SGD, weight decay skipped for 1-D
params and biases, as in 0.3.2), `create_scheduler` (+ `updates_per_epoch`, timm >= 0.9),
`SoftTargetCrossEntropy`, `get_outdir`, `AverageMeter`. Any timm >= 0.9 should do; a timm
without `create_scheduler(..., updates_per_epoch)` (< 0.9) would need `--sched-on-updates` off.

`--data_dir` is the folder that *contains* `cifar-100-python/` (torchvision `CIFAR100(...,
download=True)` only md5-checks existing files; a read-only Kaggle input works). Loader workers:
env `DSNET_WORKERS` (default 2). Run from the repo root (the config path is relative, and
`protocol.py` is imported from there). One GPU per run: pin it with `CUDA_VISIBLE_DEVICES`.

## Commands
The protocol recipe sits in `configs/cifar100_ds_resnet50_uniform.yml` (their config format):
CIFAR-100, `ds_resnet50_cifar`, 16 choices, Nesterov SGD momentum 0.9, lr 0.2, batch 256,
wd 5e-4, clip 5, 100 epochs with 5 warm-up epochs then cosine to 0 (per step), EMA 0.997, IB on,
ensemble IB off, label smoothing 0, dropout 0.
```bash
D=/kaggle/input/cifar100          # contains cifar-100-python/
CK=/kaggle/working/ckpt
COMMON="-c configs/cifar100_ds_resnet50_uniform.yml --data_dir $D --seed 1995 \
  --skip_upstream_test --output /kaggle/working/dsnet_out"
```
(a) `dsnet_ib`, DS-Net Stage I as released (sandwich + IB, EMA target, no ensemble):
```bash
CUDA_VISIBLE_DEVICES=0 DSNET_WORKERS=2 python train.py $COMMON --ckpt_dir $CK/dsnet_ib --log_prefix dsnet_ib
```
(b) `dsnet_hib`, the same plus DS-Net++'s H-IB at the last block (our addition, see Method):
```bash
CUDA_VISIBLE_DEVICES=1 DSNET_WORKERS=2 python train.py $COMMON --hib --ckpt_dir $CK/dsnet_hib --log_prefix dsnet_hib
```
At the end each run prints the 16 protocol lines (tab separated):
`[dsnet_ib] 15912.3s	val	0.25	-1/100: loss: 1.2345, top1_error: 0.4321, top5_error: 0.1234`
(elapsed = seconds since the process started). Per width: choice i in every layer, reset all
running stats, forward `--recal_batches` (default 20) shuffled, augmented training batches of 256
in train mode with a cumulative average (momentum 1/(i+1)), eval on the 10k test set. The online
supernet is evaluated (what our table reads). `--eval_ema` adds 16 lines for the EMA supernet,
tagged `<log_prefix>_ema`; do not use it inside `scripts/build_external_notebooks.py`'s
`run_pinned`, whose `VAL_LINE` ignores the tag and would file the EMA lines under the job's label
(use it in a separate `--eval_only` job instead; `scripts/collect_results.py` keys on the tag and
is fine).

`--skip_upstream_test` drops upstream's per-epoch `validate_slim` (smallest/largest, model and
EMA, on the test set, shared BN stats), its `CheckpointSaver` (`checkpoint-N.pth.tar`, which needs
those metrics) and its final `reset_bn` + 16-choice test. Without it those run as upstream as well.

(c) Evaluation only, from the resume checkpoint (same flags + `--eval_only`):
```bash
CUDA_VISIBLE_DEVICES=0 python train.py $COMMON --ckpt_dir $CK/dsnet_ib --log_prefix dsnet_ib --eval_only [--eval_ema]
```
It raises if no checkpoint exists.

(d) Resume. After every epoch `{ckpt_dir}/last.pt` is written atomically (`.tmp` + `os.replace`):
online model (incl. all BN buffers), EMA model, SGD state (momentum buffers), next epoch, and
Python/NumPy/torch/CUDA RNG states. The LR is a pure function of the update count (timm
scheduler, re-stepped to `start_epoch * 195` on resume), so there is no scheduler state to save.
If `last.pt` exists, the identical command resumes from the next epoch (prints `Resumed from ... at
epoch N`). For Kaggle's 12 h limit add `--stop_at_epoch N` to exit cleanly after epoch N (no
eval); copy `last.pt` into the new session's `--ckpt_dir` and rerun. Granularity is one epoch.

## Timing
RTX 5060 Ti, fp32, batch 256, 2 workers, `cudnn.benchmark = True` (upstream). Per-step times
read from the training log (`--log-interval 1`) in the smoke runs, counting only steps whose
widths cuDNN had already tuned:

| variant | s/step (tuned steps) | 195 steps/epoch | 100 epochs | peak mem |
|---|---|---|---|---|
| (a) dsnet_ib | 1.04-1.54, mean 1.24 (18 steps) | ~242 s | ~6.7 h | 9.47-9.57 GiB |
| (b) dsnet_hib | 1.04-1.40, mean 1.22 (6 steps) | ~240 s | ~6.7 h | 9.56-9.68 GiB |

Each step = 4 forward/backward passes + 1 EMA forward (largest) + the EMA update over the whole
state dict (2,705 tensors: every `DSBatchNorm2d` holds 16 sets of running stats) + upstream's
per-parameter "is the gradient all zero" check (161 GPU syncs). Our US-Net R50 is ~0.59 s/step
here, LCS (BN) 0.80. H-IB costs nothing measurable. Peak memory is `max_memory_allocated` and
includes cuDNN's benchmark trial workspaces.

**cuDNN autotuning:** with `benchmark = True` every new width costs ~17-48 s the first time a
process sees it (steps jump from ~1.2 s to ~20 s). There are only 16 widths, so this is a one-off
of ~5-7 min at the start of every process (each Kaggle session pays it again), plus ~20-40 s per
width the first time the evaluation sees it (recalibration and the last test batch of 16 are new
shapes): the first 16-width evaluation in a process took ~5.5 min (dsnet_ib) and ~7 min
(dsnet_hib); a second one in the same process (EMA) ~1 min. The smoke runs are therefore
dominated by tuning and their epoch averages (26-29 s/step) are not the training speed.

A T4 is ~2-3x slower in fp32: expect ~13-20 h per run, i.e. **at least one, probably two resumes**
per branch. Use `--stop_at_epoch` to end a session cleanly (e.g. 45 then 90 then 100).

## Smoke test actually run (local GPU)
Common flags: the config, `--data_dir D:/.../CL/data --seed 1995 --skip_upstream_test --epochs 2
--warmup-epochs 1 --max_iters_per_epoch 10` (a 2-epoch schedule: epoch 0 is warm-up, epoch 1
cosine). Per branch: run 1 adds `--stop_at_epoch 1` (10 steps, checkpoint, exit); run 2 is the
same command without it (resumes at epoch 1, trains, checkpoints, 16-width evaluation).
dsnet_ib run 2 used `--max_iters_per_epoch 30 --eval_ema` to get tuned steps for the timing and
to run the EMA evaluation.
```
ib  run1: Epoch 0 train time 286.3s, 10 steps, ..., lr now 0.01026, peak mem 9.47 GiB
          Saved checkpoint .../ckpt_dsnet_ib/last.pt (next epoch 1)
          Stopping after epoch 0 (--stop_at_epoch 1); rerun to resume.
ib  run2: Resumed from .../ckpt_dsnet_ib/last.pt at epoch 1
          Train: 1 [   0/29 (  0%)] ... LR: 2.000e-01          (cosine start right after warm-up)
          Epoch 1 train time 400.1s, 30 steps, ..., lr now 0.1885, peak mem 9.57 GiB
[smoke_dsnet_ib] 476.9s	val	0.25	-1/2: loss: 4.6085, top1_error: 0.9901, top5_error: 0.9501
[smoke_dsnet_ib] 500.0s	val	0.30	-1/2: loss: 4.6084, top1_error: 0.9901, top5_error: 0.9501
... (16 lines) ...
[smoke_dsnet_ib] 783.1s	val	0.95	-1/2: loss: 4.5197, top1_error: 0.9724, top5_error: 0.8825
[smoke_dsnet_ib] 806.3s	val	1.00	-1/2: loss: 4.4587, top1_error: 0.9713, top5_error: 0.8720
[smoke_dsnet_ib_ema] 808.4s	val	0.25	-1/2: loss: 4.6927, top1_error: 0.9917, top5_error: 0.9567
... (16 lines) ...
[smoke_dsnet_ib_ema] 867.7s	val	1.00	-1/2: loss: 4.7055, top1_error: 0.9613, top5_error: 0.8704
hib run1: Train: 0 [   0/9 ...] ... Loss_HIB_smallest: 0.003908
          ...      [   9/9 ...] ... Loss_HIB_smallest: 0.003561
          Epoch 0 train time 261.3s, 10 steps, ..., lr now 0.01026, peak mem 9.56 GiB / Stopping after epoch 0
hib run2: Resumed from .../ckpt_dsnet_hib/last.pt at epoch 1
          Train: 1 [   0/9 ...] ... LR: 2.000e-01 ... Loss_HIB_smallest: 0.003395
          Train: 1 [   9/9 ...] ... LR: 1.990e-01 ... Loss_HIB_smallest: 0.000846
          Epoch 1 train time 258.3s, 10 steps, ..., lr now 0.1987, peak mem 9.68 GiB
[smoke_dsnet_hib] 334.4s	val	0.25	-1/2: loss: 4.6339, top1_error: 0.9904, top5_error: 0.9498
... (16 lines) ...
[smoke_dsnet_hib] 741.7s	val	1.00	-1/2: loss: 4.6363, top1_error: 0.9723, top5_error: 0.8935
```
Near chance after 20-40 steps, as expected; width 1.00 has moved off chance and the narrow
widths have not. All lines are in the `scripts/collect_results.py` / notebook `VAL_LINE` format.
Checked in the checkpoints: next epoch 2; online and EMA state dicts both present (2,705 keys,
2,709 with H-IB); 161 SGD momentum buffers (165 with H-IB), Nesterov, two groups (wd 5e-4 on the
54 conv/linear weights, wd 0 on the 107 BN parameters and biases). H-IB really trains: the two
2048x2048 predictor weights, which get gradient only from L_HIB, have non-zero momentum buffers
(norms 0.16, 0.14) and differ from their EMA copies. The LR after resume continues the schedule
(0.2 at the first step of epoch 1, the end of warm-up).
Not run locally: a full epoch, `--eval_only` (same code path as the final evaluation, after
`load_checkpoint`), the ImageNet path, upstream's `validate_slim`/`CheckpointSaver` path (i.e.
a run without `--skip_upstream_test`).

## Changes and deviations
Code (all in `dsnet.patch`):
1. `dyn_slim/models/dyn_slim_resnet.py` (new): `ds_resnet50_cifar`, a DS-ResNet-50 for 32x32 built
   from their `DSConv2d`/`DSpwConv2d`/`DSBatchNorm2d`/`DSLinear`/`DSAdaptiveAvgPool2d` and their
   `set_exist_attr`/`set_mode` conventions (the release ships only DS-MBNet). Against our
   `models/us_resnet.py` (depth 50):
   - Same: 3x3 stride-1 stem, no max-pool, bottlenecks [3,4,6,3] (1x1, 3x3 with the stride, 1x1
     x4), 1x1 conv + BN projection shortcut where stride or width changes, ReLU after the sum,
     global average pool, linear classifier with bias; channels at width w are
     `make_divisible(C*w)` (divisor 8; their `min_value` is 8 instead of our 1, which never binds
     since the smallest layer has 16 channels). Checked: identical channel counts at all 16
     widths in every layer (e.g. stem 16/24/24/24/32/.../64, pooled 512/616/.../2048), and the same
     parameter count, 23,705,252.
   - BN: `DSBatchNorm2d` = one shared affine (sliced) plus private running statistics per channel
     choice (the stats are found by channel count, so widths that round to the same count in a
     layer, e.g. 0.30/0.35/0.40 in the stem, share them). Ours: shared affine, batch statistics in
     training. Training uses batch statistics in both and the evaluation recalibrates per width,
     so only the stored stats differ.
   - Initialisation is their `efficientnet_init_weights` (convs N(0, 2/(k^2 * C_out)), as ours;
     BN 1/0; classifier U(+-1/sqrt(100)) with zero bias, ours N(0, 0.01)) plus their
     `init_residual_norm`: gamma of the last BN of every bottleneck = 0 (their residual blocks do
     this, and DS-Net++ sec. 5.3 states it for the CNN supernets). Ours starts every gamma at 1.
   - No SE / `MultiHeadGate`: their MobileNet blocks always carry an SE-style attention module
     (the gate's trunk); a standard ResNet-50 and ours have none, and no gate is trained here.
   - Width space: one channel choice for all layers, stem included. The paper's DS-ResNet keeps
     the stem at 64 and has one gate per stage with ratios {0.25, 0.5, 0.75, 1} (256 paths); their
     MBNet 'uniform' mode slims only the gated stages and keeps the first ones at their smallest
     width. Here the 16 protocol widths in every layer, as the protocol needs.
2. `dyn_slim/apis/train_slim.py`: `--max_iters_per_epoch` (smoke only); `--hib` (H-IB term, see
   Method; logs `Loss_HIB_smallest`); `--clip-grad` (global norm, after their zero-gradient filter,
   before `optimizer.step()`); the step timer; the per-sample reseeding gets the run's seed as an
   offset (item 13); `from timm.utils import *` replaced by explicit names (the timm >= 0.9
   `random` shadowing crash, see Setup).
3. `train.py`: timm import shims; `--dataset cifar100` path (CIFAR loaders from `protocol.py`
   built before the scheduler, ImageNet loaders unchanged under `if`); the positional `data`
   is optional; `--sched-on-updates` / `--warmup-prefix` (timm's own scheduler options, item 8);
   resume via `protocol.load_checkpoint` after the EMA model exists, with the per-update LR
   re-stepped; per-epoch checkpoint, `--stop_at_epoch`, `--eval_only`, `--skip_upstream_test`;
   epoch timing / peak-memory line; model profiling at 32 px; the 16-width evaluation
   (+ `--eval_ema`). Upstream's `--resume`/`CheckpointSaver` path is still there and untouched.
4. `dyn_slim/models/dyn_slim_blocks.py`, `dyn_slim_net.py`, `dyn_slim/utils/model_profiling.py`:
   timm imports from `timm.layers` with a fallback to `timm.models.layers`; `register_model` from
   `timm.models`. `dyn_slim/models/__init__.py` imports the new model so it registers.
5. `protocol.py` (new): CIFAR-100 loaders (RandomCrop 32 pad 4 + HFlip, mean/std
   (0.5071,0.4865,0.4409)/(0.2673,0.2564,0.2762), `DSNET_WORKERS`, persistent workers), checkpoint
   save/load, BN recalibration, evaluation and the protocol lines (as LCS's `protocol.py`).
6. `configs/cifar100_ds_resnet50_uniform.yml` (new): the recipe below, in their config format.

Recipe (ours vs paper / upstream):
7. CIFAR-100 32x32, 100 epochs (paper 150 on ImageNet, upstream config 150 + 10 cool-down),
   batch 256 (512), lr 0.2 (paper: 0.2 for DS-ResNet at 512, 0.08 for DS-MBNet), Nesterov SGD
   momentum 0.9 (timm 'sgd', as upstream), wd 5e-4 (1e-4), not applied to BN parameters and biases
   (timm `create_optimizer`, as upstream; ours does the same), grad clip 5 (none upstream),
   label smoothing 0 (0.1), no RandAugment / random erasing / colour jitter (upstream
   rand-m9-mstd0.5, reprob 0.2), dropout 0 (0.2 before the classifier on the largest), fp32.
8. LR per step: timm `CosineLRScheduler` in per-update mode with warm-up prefix: linear from 0
   to 0.2 over 5 epochs (975 updates; the very first update runs at lr 0), then cosine to 0 over
   the other 95 epochs, as our train.py. Upstream: per epoch, warm-up from 1e-5 (so epoch 0 runs
   at 1e-5), cosine over the full length including warm-up, min lr 1e-5, then 10 cool-down epochs.
9. `ensemble_ib` false, as in their released configs and the code default. The paper trains
   DS-ResNet with E-IB (ensemble target for the smallest); its ablation (DS-MBNet) gives +0.2 on
   the smallest and +0.3 on the largest over IB.
10. Uniform widths drawn from {0.30, ..., 1.00} (`randint(1, 15)`, their rule), not from all
    sixteen; 0.25 and 1.00 are trained every step anyway.
11. Training loader `drop_last=True` (timm's training loader): 195 steps of 256 per epoch, 80
    images dropped per epoch. Ours keeps the last partial batch.
12. Evaluation: the online supernet at the 16 widths after per-width recalibration (20 batches of
    256, cumulative). Upstream reports the EMA supernet in its per-epoch validation and tests both
    at the end, after `reset_bn` (all running stats reset, then up to 1,001 batches of 4 x batch
    size forwarded at all choices in turn, momentum 0.1). `--eval_ema` prints the EMA numbers too.
13. Seeding: upstream reseeds `random`, NumPy and torch before every sample with
    `seed = epoch * len(loader) + k`, the same stream in every run, which also fixes the data order
    and augmentation from epoch 1 on regardless of `--seed`. We add `seed * 10^6` to it (and take
    the NumPy seed mod 2^32), so seed repeats differ in widths and data; the sampling rule is
    unchanged. `--seed` also seeds the initialisation as upstream.
14. `cudnn.benchmark = True` kept (upstream, and ours): see Timing for the one-off cost.
15. Kept from upstream although unusual: parameters whose gradient is exactly zero in a step get
    `grad = None` before `optimizer.step()` (no weight decay, no momentum update that step; at
    initialisation this holds for every conv feeding a zero-gamma BN); the EMA averages the whole
    state dict, integer `num_batches_tracked` included (unused by `F.batch_norm`).
16. H-IB (`--hib` only), our reading of DS-Net++ where the paper is silent: C_mlp = 2048; the
    predictor has biases (eq. 19 writes weights only) and is initialised by their
    `efficientnet_init_weights`; it gets weight decay on its weights like any linear layer; the
    target is the EMA supernet's pooled feature after the last block's ReLU; the term is applied
    to the two uniform widths and the smallest; MvCo (multi-view consistency, used with H-IB for
    DS-Net++) and the compound routing space are not used; nothing of DS-Net++ is in the code
    release, so this is written from the paper only.

## Uncertain
- The H-IB term is small at this scale: (1/C) * squared L2 between pooled features is ~4e-3 at
  initialisation and fell to ~8e-4 within 20 steps, against a soft-target CE of ~4.6, both
  weighted 0.5. So dsnet_hib may behave mostly like dsnet_ib with the soft-target weight halved.
  That is the paper's formula and weights; we did not rescale it.
- Speed: 1.24 s/step here is twice our US-Net. On a T4 each branch needs one or two resumes.
- Steady s/step is from 18 (6) tuned steps with per-step logging, not a full epoch.
