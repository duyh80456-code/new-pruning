# DS-Net (official code), Stage I supernet, under our protocol: CIFAR-100 / ResNet-50, Tiny ImageNet / ResNet-18, CIFAR-100 / MobileNetV2 (slim_last)

Papers: Li et al., "Dynamic Slimmable Network", CVPR 2021 (oral), arXiv 2103.13258; journal
extension DS-Net++, TPAMI 2023, arXiv 2109.10060. Upstream: https://github.com/changlin31/DS-Net
Pinned commit: `15cd3036970ec27d2c306014344fd50d9e9b888b` (HEAD, 2021-12-31, "set default ensemble
false"). Our changes: `dsnet.patch` (= `git diff` against that commit, new files `protocol.py`,
`dyn_slim/models/dyn_slim_resnet.py`, `dyn_slim/models/dyn_slim_mobilenetv2.py`,
`configs/cifar100_ds_resnet50_uniform.yml`, `configs/tiny_ds_resnet18_uniform.yml`,
`configs/cifar100_ds_mobilenetv2_slimlast_uniform.yml` included), applied with `git apply`.
Checked: applies cleanly to a fresh clone from GitHub and reproduces the tested tree byte for byte.

Three settings, one model and one config each:

| setting | model (new) | config (new) | matches our |
|---|---|---|---|
| CIFAR-100, ResNet-50 | `ds_resnet50_cifar` | `configs/cifar100_ds_resnet50_uniform.yml` | `models/us_resnet.py` depth 50 |
| Tiny ImageNet, ResNet-18 | `ds_resnet18_tiny` | `configs/tiny_ds_resnet18_uniform.yml` | `models/us_resnet.py` depth 18, `stem_stride: 2`; `data/tinyimagenet.py`; `apps/cifar100_es_a_r18_tiny.yml` |
| CIFAR-100, MobileNetV2 slim_last | `ds_mobilenetv2_cifar_slimlast` | `configs/cifar100_ds_mobilenetv2_slimlast_uniform.yml` | `models/us_mobilenet_v2_cifar.py` with `slim_last: True`; `apps/cifar100_ev_a_mbv2_slimlast.yml` |

## Method
Their Stage I supernet training, `dyn_slim/apis/train_slim.py` with
`configs/mobilenetv1_bn_uniform.yml` as released, kept as it is (all three settings):
- Sandwich sample list `['largest', 'uniform', 'uniform', 'smallest']` every step, one
  forward/backward per sample (upstream already backpropagates per sample).
- In-place bootstrapping (IB): the largest learns from the label (CE); the two uniform widths
  and the smallest learn from the soft target softmax(EMA(x)), the EMA supernet at its largest
  width, with timm's `SoftTargetCrossEntropy` (T = 1). EMA = their `ModelEma` (all parameters and
  buffers, decay 0.997, updated after every optimizer step, kept in eval mode). `ensemble_ib`
  false, as in their released config.
- Their slimmable ops and BN (`DSConv2d`, `DSpwConv2d`, `DSdwConv2d` (MobileNetV2),
  `DSBatchNorm2d` with shared affine and private running statistics per channel choice,
  `DSLinear`, `DSAdaptiveAvgPool2d`), their `efficientnet_init_weights`, their per-sample
  reseeding, their "drop parameters whose gradient is exactly zero before `optimizer.step()`"
  filter, timm's SGD and cosine scheduler.
- Stage II (the gate) is not run. The supernet is evaluated at fixed uniform widths.

Width space (all three models): every slimmable layer has the same sixteen channel choices,
`make_divisible(C * w)` for w = 0.25, 0.30, ..., 1.00, so channel choice i is width w_i in every
layer at once (stem, all convs and BNs, shortcuts, the MobileNetV2's last 1x1 to 1280, pooled
feature, classifier input). Their 'uniform' mode then draws one choice for all layers:
`random.randint(1, 15)`, i.e. 0.30..1.00, exactly as `DSNet.set_mode` draws `randint(1, 13)` over
its 14 choices (the smallest is index 0 and is trained separately as 'smallest'). So each step
trains 1.00 (label), 0.25 (soft target) and two widths drawn uniformly from {0.30, ..., 1.00}
(soft target). The sixteen-width evaluation uses the same sixteen choices. The three models share
this code (`UniformChoiceMixin` in `dyn_slim_resnet.py`: `set_mode`, `set_width`, pooled feature,
classifier, H-IB predictor).

H-IB (`--hib`, our addition from DS-Net++, not in their code; default off, so the default run is
DS-Net as released; **used for R50 only**, `dsnet_hib`): every non-largest sampled width (both
uniform ones and the smallest) also gets the hierarchical in-place bootstrapping loss of DS-Net++
eq. (33) at the last block only (the paper's CNN setting: "only calculated with the last blocks'
outputs"): `L_HIB = (1/C) * || phi(f_k(x)) - f_L(x) ||^2`, averaged over the batch, where f_k is
the globally pooled output of the last bottleneck at the sampled width (c channels), f_L the
pooled C = 2048-d feature of the EMA supernet at its largest width (no gradient), and phi a
slimmable 2-layer MLP predictor of the online supernet, eq. (19): `W2 relu(W1[:, :c] f)` with
W1: C_mlp x 2048 (input sliced to c, their `DSLinear`), W2: 2048 x C_mlp, C_mlp = 2048 (the
paper gives no value; `--hib_hidden`). Eq. (34) with the paper's lambda_IB = lambda_HIB = 0.5:
each non-largest width's loss is `0.5 * L_IB + 0.5 * L_HIB`, so the soft-target term is halved
relative to the default run; the largest keeps its CE unscaled. No feature normalisation (the
paper: "simplistic L2 distance"). The predictor adds 8,392,704 parameters (training only; the
supernet itself is unchanged). The flag also builds on the two new models (predictor on the
pooled 512-d / 1280-d feature: 13,371,144 / 7,598,180 parameters with it; checked to build and
run a forward), but no H-IB run is planned or smoked there.

Parameter counts (checked equal to our in-repo models at width 1.0, see Changes item 1):

| model | classes | params | with `--hib` |
|---|---|---|---|
| `ds_resnet50_cifar` | 100 | **23,705,252** | 32,097,956 |
| `ds_resnet18_tiny` | 200 | **11,271,432** | 13,371,144 |
| `ds_mobilenetv2_cifar_slimlast` | 100 | **2,351,972** | 7,598,180 |

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
constraint), PyYAML, Pillow. No apex (AMP stays off: fp32). Tested with Python 3.10.20, torch
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

Data (`--dataset` comes from the config; `--data_dir` as our `dataset_dir`):
- CIFAR-100 (`dataset: cifar100`): `--data_dir` is the folder that *contains*
  `cifar-100-python/` (torchvision `CIFAR100(..., download=True)` only md5-checks existing files;
  a read-only Kaggle input works).
- Tiny ImageNet (`dataset: tinyimagenet`): as our `data/tinyimagenet.py`, first hit wins:
  `--data_dir/tiny-imagenet-200`, any folder with the official layout under `/kaggle/input` (also
  a zip of it, unpacked into `--data_dir`), else the Stanford zip downloaded into `--data_dir`.
  The class ids and val labels are md5-checked against the official release; the JPEGs are
  decoded once into `--data_dir/tinyimagenet_cache/{train_x,train_y,val_x,val_y}.npy` (1.35 GB,
  the same files our pipeline writes, so a cache written by our code is read as is). So
  `--data_dir` must be writable unless the cache is already there: on Kaggle use e.g.
  `--data_dir /kaggle/working/data` with the dataset attached as an input (our notebooks do the
  same with `dataset_dir: data`).
Loader workers: env `DSNET_WORKERS` (default 2). Run from the repo root (the config path is
relative, and `protocol.py` is imported from there). One GPU per run: pin it with
`CUDA_VISIBLE_DEVICES`.

## Commands
Recipe (same for all three, in their config format; only dataset, model and classes differ):
Nesterov SGD momentum 0.9, lr 0.2, batch 256, wd 5e-4, clip 5, 100 epochs with 5 warm-up epochs
then cosine to 0 (per step), EMA 0.997, IB on, ensemble IB off, label smoothing 0, dropout 0.
This is the recipe of `apps/cifar100_es_a_r18_tiny.yml` and `apps/cifar100_ev_a_mbv2_slimlast.yml`
(identical to our R50 one).
```bash
C100=/kaggle/input/cifar100              # contains cifar-100-python/
TINY=/kaggle/working/data                # tiny-imagenet-200/ here or under /kaggle/input; cache written here
CK=/kaggle/working/ckpt
OUT=/kaggle/working/dsnet_out
SEED=1995                                # labels below are run with --seed 1995, 2026 and 2006
R50="-c configs/cifar100_ds_resnet50_uniform.yml --data_dir $C100"
R18T="-c configs/tiny_ds_resnet18_uniform.yml --data_dir $TINY"
MBV2="-c configs/cifar100_ds_mobilenetv2_slimlast_uniform.yml --data_dir $C100"
COMMON="--seed $SEED --skip_upstream_test --output $OUT"
```
One `--ckpt_dir` per label and seed (the resume checkpoint is `{ckpt_dir}/last.pt`):

(a) `dsnet_ib_r50`, DS-Net Stage I as released (sandwich + IB, EMA target, no ensemble), R50:
```bash
CUDA_VISIBLE_DEVICES=0 DSNET_WORKERS=2 python train.py $R50 $COMMON --ckpt_dir $CK/dsnet_ib_r50_s$SEED --log_prefix dsnet_ib_r50
```
(b) `dsnet_ib_r18_tiny`, the same on Tiny ImageNet / ResNet-18:
```bash
CUDA_VISIBLE_DEVICES=0 DSNET_WORKERS=2 python train.py $R18T $COMMON --ckpt_dir $CK/dsnet_ib_r18_tiny_s$SEED --log_prefix dsnet_ib_r18_tiny
```
(c) `dsnet_ib_mbv2_slimlast`, the same on CIFAR-100 / MobileNetV2 slim_last:
```bash
CUDA_VISIBLE_DEVICES=0 DSNET_WORKERS=2 python train.py $MBV2 $COMMON --ckpt_dir $CK/dsnet_ib_mbv2_slimlast_s$SEED --log_prefix dsnet_ib_mbv2_slimlast
```
(d) `dsnet_hib`, R50 only: (a) plus DS-Net++'s H-IB at the last block (our addition, see Method):
```bash
CUDA_VISIBLE_DEVICES=1 DSNET_WORKERS=2 python train.py $R50 $COMMON --hib --ckpt_dir $CK/dsnet_hib_s$SEED --log_prefix dsnet_hib
```
Each of (a)-(c) with `SEED=1995`, `2026`, `2006` (same label, the seed tells the runs apart).
`--seed` sets the initialisation seed and offsets upstream's per-sample reseeding by
`seed * 10^6` (Changes item 13): without that offset the width draws and data order from
epoch 1 on would be the same in every seed.

At the end each run prints the 16 protocol lines (tab separated):
`[dsnet_ib_r50] 15912.3s	val	0.25	-1/100: loss: 1.2345, top1_error: 0.4321, top5_error: 0.1234`
(elapsed = seconds since the process started). Per width: choice i in every layer, reset all
running stats, forward `--recal_batches` (default 20) shuffled, augmented training batches of 256
in train mode with a cumulative average (momentum 1/(i+1)), eval on the 10k test set (CIFAR-100
test / Tiny ImageNet official val). The online supernet is evaluated (what our table reads).
`--eval_ema` adds 16 lines for the EMA supernet, tagged `<log_prefix>_ema`; do not use it inside
`scripts/build_external_notebooks.py`'s `run_pinned`, whose `VAL_LINE` ignores the tag and would
file the EMA lines under the job's label (use it in a separate `--eval_only` job instead;
`scripts/collect_results.py` keys on the tag and is fine).

`--skip_upstream_test` drops upstream's per-epoch `validate_slim` (smallest/largest, model and
EMA, on the test set, shared BN stats), its `CheckpointSaver` (`checkpoint-N.pth.tar`, which needs
those metrics) and its final `reset_bn` + 16-choice test. Without it those run as upstream as well.

(e) Evaluation only, from the resume checkpoint (same flags + `--eval_only`):
```bash
CUDA_VISIBLE_DEVICES=0 python train.py $R50 $COMMON --ckpt_dir $CK/dsnet_ib_r50_s$SEED --log_prefix dsnet_ib_r50 --eval_only [--eval_ema]
```
It raises if no checkpoint exists.

(f) Resume. After every epoch `{ckpt_dir}/last.pt` is written atomically (`.tmp` + `os.replace`):
online model (incl. all BN buffers), EMA model, SGD state (momentum buffers), next epoch, and
Python/NumPy/torch/CUDA RNG states. The LR is a pure function of the update count (timm
scheduler, re-stepped to `start_epoch * updates_per_epoch` on resume: 195 on CIFAR-100, 390 on
Tiny ImageNet), so there is no scheduler state to save. If `last.pt` exists, the identical
command resumes from the next epoch (prints `Resumed from ... at epoch N`). For Kaggle's 12 h
limit add `--stop_at_epoch N` to exit cleanly after epoch N (no eval); copy `last.pt` into the new
session's `--ckpt_dir` and rerun. Granularity is one epoch.

## Timing
RTX 5060 Ti, fp32, batch 256, 2 workers, `cudnn.benchmark = True` (upstream). Per-step times
read from the training log (`--log-interval 1`) in the smoke runs, counting only steps whose
widths cuDNN had already tuned (the fast steps; every other step pays tuning, see below):

| setting / variant | s/step (tuned steps) | steps/epoch | 100 epochs (train) | peak mem |
|---|---|---|---|---|
| R50 (a) dsnet_ib_r50 | 1.04-1.54, mean 1.24 (18 steps) | 195 | ~6.7 h | 9.47-9.57 GiB |
| R50 (d) dsnet_hib | 1.04-1.40, mean 1.22 (6 steps) | 195 | ~6.7 h | 9.56-9.68 GiB |
| Tiny R18 (b) dsnet_ib_r18_tiny | 0.22-0.33, mean 0.28 (5 steps) | 390 | ~3.1 h | 3.37-3.41 GiB |
| MBV2 slim_last (c) dsnet_ib_mbv2_slimlast | 0.42-0.47, mean 0.44 (6 steps) | 195 | ~2.4 h | 5.74 GiB |

Each step = 4 forward/backward passes + 1 EMA forward (largest) + the EMA update over the whole
state dict (every `DSBatchNorm2d` holds 16 sets of running stats; 2,705 tensors on R50) +
upstream's per-parameter "is the gradient all zero" check (one GPU sync per parameter tensor:
161 on R50). Our US-Net R50 is ~0.59 s/step here, LCS (BN) 0.80. H-IB costs nothing measurable.
Peak memory is `max_memory_allocated` and includes cuDNN's benchmark trial workspaces. The Tiny
and MBV2 tuned-step means rest on 5 and 6 steps only (the smoke is 2 x 10 steps, most of them
tuning); treat them as +-20%.

**cuDNN autotuning:** with `benchmark = True` every new width costs time the first time a
process sees it (R50: ~17-48 s, steps jump from ~1.2 s to ~20 s; Tiny R18 and MBV2: steps of
3-9 s instead of 0.3-0.5 s). There are only 16 widths, so this is a one-off of a few minutes at
the start of every process (each Kaggle session pays it again), plus a few seconds to ~40 s per
width the first time the evaluation sees it: the first 16-width evaluation in a process took
~5.5 min (R50 dsnet_ib), ~7 min (R50 dsnet_hib), ~1.3 min (Tiny R18), ~1.3 min (MBV2).

A T4 is ~2-3x slower in fp32: R50 ~13-20 h per run, i.e. **at least one, probably two resumes**
per branch (use `--stop_at_epoch`, e.g. 45 then 90 then 100); Tiny R18 ~6-10 h and MBV2
~5-7 h, which may fit one 12 h session but keep `--stop_at_epoch` as a safeguard (e.g. 60).
On Kaggle's 2-CPU machines the loaders (2 workers, PIL crop/flip/ToTensor) may limit the faster
models; not measured there.

## Smoke tests actually run (local GPU)
Common flags: the config, `--data_dir <data> --seed 1995 --skip_upstream_test --epochs 2
--warmup-epochs 1 --max_iters_per_epoch 10 --log-interval 1` (a 2-epoch schedule: epoch 0 is
warm-up, epoch 1 cosine). Per setting: run 1 adds `--stop_at_epoch 1` (10 steps, checkpoint,
exit); run 2 is the same command without it (resumes at epoch 1, trains, checkpoints, 16-width
evaluation). R50 dsnet_ib run 2 used `--max_iters_per_epoch 30 --eval_ema` to get tuned steps for
the timing and to run the EMA evaluation.
```
R50 ib  run1: Epoch 0 train time 286.3s, 10 steps, ..., lr now 0.01026, peak mem 9.47 GiB
              Saved checkpoint .../ckpt_dsnet_ib/last.pt (next epoch 1)
              Stopping after epoch 0 (--stop_at_epoch 1); rerun to resume.
R50 ib  run2: Resumed from .../ckpt_dsnet_ib/last.pt at epoch 1
              Train: 1 [   0/29 (  0%)] ... LR: 2.000e-01          (cosine start right after warm-up)
              Epoch 1 train time 400.1s, 30 steps, ..., lr now 0.1885, peak mem 9.57 GiB
[smoke_dsnet_ib] 476.9s	val	0.25	-1/2: loss: 4.6085, top1_error: 0.9901, top5_error: 0.9501
... (16 lines) ...
[smoke_dsnet_ib] 806.3s	val	1.00	-1/2: loss: 4.4587, top1_error: 0.9713, top5_error: 0.8720
[smoke_dsnet_ib_ema] 808.4s	val	0.25	-1/2: loss: 4.6927, top1_error: 0.9917, top5_error: 0.9567
... (16 lines) ...
[smoke_dsnet_ib_ema] 867.7s	val	1.00	-1/2: loss: 4.7055, top1_error: 0.9613, top5_error: 0.8704
R50 hib run1: Train: 0 [   0/9 ...] ... Loss_HIB_smallest: 0.003908
              ...      [   9/9 ...] ... Loss_HIB_smallest: 0.003561
              Epoch 0 train time 261.3s, 10 steps, ..., lr now 0.01026, peak mem 9.56 GiB / Stopping after epoch 0
R50 hib run2: Resumed from .../ckpt_dsnet_hib/last.pt at epoch 1
              Train: 1 [   0/9 ...] ... LR: 2.000e-01 ... Loss_HIB_smallest: 0.003395
              Train: 1 [   9/9 ...] ... LR: 1.990e-01 ... Loss_HIB_smallest: 0.000846
              Epoch 1 train time 258.3s, 10 steps, ..., lr now 0.1987, peak mem 9.68 GiB
[smoke_dsnet_hib] 334.4s	val	0.25	-1/2: loss: 4.6339, top1_error: 0.9904, top5_error: 0.9498
... (16 lines) ...
[smoke_dsnet_hib] 741.7s	val	1.00	-1/2: loss: 4.6363, top1_error: 0.9723, top5_error: 0.8935
```
Tiny ImageNet / ResNet-18 (`--data_dir` = our repo's `data/`, which holds `tiny-imagenet-200/`
and the decoded cache):
```
run1: Model ds_resnet18_tiny created, param count: 11271432
      Tiny ImageNet at .../data/tiny-imagenet-200: official classes and val labels
      Train: 0 [   1/9 ...] ... LR: 5.128e-04        (= 0.2 / 390: 390 steps per epoch)
      Train: 0 [   3/9 ...] ... Time: 0.225s ...      (tuned steps: 0.225, 0.267, 0.300 s)
      Epoch 0 train time 58.1s, 10 steps, ..., lr now 0.005128, peak mem 3.37 GiB
      Saved checkpoint .../ck_r18_tiny/last.pt (next epoch 1) / Stopping after epoch 0
run2: Resumed from .../ck_r18_tiny/last.pt at epoch 1
      Train: 1 [   0/9 ...] ... LR: 2.000e-01
      Train: 1 [   8/9 ...] ... Time: 0.326s ...      (tuned steps: 0.326, 0.306 s)
      Epoch 1 train time 65.7s, 10 steps, ..., lr now 0.1997, peak mem 3.41 GiB
      Saved checkpoint .../ck_r18_tiny/last.pt (next epoch 2)
[smoke_dsnet_ib_r18_tiny] 91.5s	val	0.25	-1/2: loss: 5.3009, top1_error: 0.9911, top5_error: 0.9658
[smoke_dsnet_ib_r18_tiny] 100.0s	val	0.30	-1/2: loss: 5.3066, top1_error: 0.9921, top5_error: 0.9666
[smoke_dsnet_ib_r18_tiny] 104.4s	val	0.35	-1/2: loss: 5.3099, top1_error: 0.9934, top5_error: 0.9666
[smoke_dsnet_ib_r18_tiny] 107.8s	val	0.40	-1/2: loss: 5.3063, top1_error: 0.9904, top5_error: 0.9634
[smoke_dsnet_ib_r18_tiny] 113.0s	val	0.45	-1/2: loss: 5.3042, top1_error: 0.9912, top5_error: 0.9655
[smoke_dsnet_ib_r18_tiny] 117.5s	val	0.50	-1/2: loss: 5.2970, top1_error: 0.9913, top5_error: 0.9646
[smoke_dsnet_ib_r18_tiny] 121.9s	val	0.55	-1/2: loss: 5.2948, top1_error: 0.9917, top5_error: 0.9636
[smoke_dsnet_ib_r18_tiny] 127.2s	val	0.60	-1/2: loss: 5.2827, top1_error: 0.9903, top5_error: 0.9570
[smoke_dsnet_ib_r18_tiny] 130.6s	val	0.65	-1/2: loss: 5.2859, top1_error: 0.9906, top5_error: 0.9583
[smoke_dsnet_ib_r18_tiny] 135.9s	val	0.70	-1/2: loss: 5.2738, top1_error: 0.9898, top5_error: 0.9555
[smoke_dsnet_ib_r18_tiny] 144.0s	val	0.75	-1/2: loss: 5.2680, top1_error: 0.9873, top5_error: 0.9542
[smoke_dsnet_ib_r18_tiny] 148.7s	val	0.80	-1/2: loss: 5.2636, top1_error: 0.9866, top5_error: 0.9498
[smoke_dsnet_ib_r18_tiny] 154.2s	val	0.85	-1/2: loss: 5.2392, top1_error: 0.9859, top5_error: 0.9448
[smoke_dsnet_ib_r18_tiny] 158.0s	val	0.90	-1/2: loss: 5.2381, top1_error: 0.9850, top5_error: 0.9430
[smoke_dsnet_ib_r18_tiny] 163.8s	val	0.95	-1/2: loss: 5.2210, top1_error: 0.9815, top5_error: 0.9362
[smoke_dsnet_ib_r18_tiny] 168.8s	val	1.00	-1/2: loss: 5.2090, top1_error: 0.9808, top5_error: 0.9301
```
CIFAR-100 / MobileNetV2 slim_last:
```
run1: Model ds_mobilenetv2_cifar_slimlast created, param count: 2351972
      Train: 0 [   1/9 ...] ... LR: 1.026e-03        (= 0.2 / 195)
      Train: 0 [   3/9 ...] ... Time: 0.433s ...      (tuned steps: 0.433, 0.439, 0.465 s)
      Epoch 0 train time 61.4s, 10 steps, ..., lr now 0.01026, peak mem 5.74 GiB
      Saved checkpoint .../ck_mbv2/last.pt (next epoch 1) / Stopping after epoch 0
run2: Resumed from .../ck_mbv2/last.pt at epoch 1
      Train: 1 [   0/9 ...] ... LR: 2.000e-01
      Train: 1 [   8/9 ...] ... Time: 0.421s ...      (tuned steps: 0.421, 0.421, 0.436 s)
      Epoch 1 train time 60.3s, 10 steps, ..., lr now 0.1987, peak mem 5.74 GiB
[smoke_dsnet_ib_mbv2_slimlast] 93.2s	val	0.25	-1/2: loss: 4.6951, top1_error: 0.9910, top5_error: 0.9479
[smoke_dsnet_ib_mbv2_slimlast] 100.7s	val	0.30	-1/2: loss: 4.7212, top1_error: 0.9907, top5_error: 0.9546
[smoke_dsnet_ib_mbv2_slimlast] 103.4s	val	0.35	-1/2: loss: 4.7347, top1_error: 0.9909, top5_error: 0.9537
[smoke_dsnet_ib_mbv2_slimlast] 110.3s	val	0.40	-1/2: loss: 4.7505, top1_error: 0.9909, top5_error: 0.9555
[smoke_dsnet_ib_mbv2_slimlast] 113.9s	val	0.45	-1/2: loss: 4.7706, top1_error: 0.9905, top5_error: 0.9482
[smoke_dsnet_ib_mbv2_slimlast] 120.2s	val	0.50	-1/2: loss: 4.7964, top1_error: 0.9912, top5_error: 0.9531
[smoke_dsnet_ib_mbv2_slimlast] 124.2s	val	0.55	-1/2: loss: 4.8365, top1_error: 0.9906, top5_error: 0.9505
[smoke_dsnet_ib_mbv2_slimlast] 131.7s	val	0.60	-1/2: loss: 4.8343, top1_error: 0.9910, top5_error: 0.9513
[smoke_dsnet_ib_mbv2_slimlast] 136.1s	val	0.65	-1/2: loss: 4.8556, top1_error: 0.9874, top5_error: 0.9534
[smoke_dsnet_ib_mbv2_slimlast] 140.0s	val	0.70	-1/2: loss: 4.8488, top1_error: 0.9879, top5_error: 0.9461
[smoke_dsnet_ib_mbv2_slimlast] 145.7s	val	0.75	-1/2: loss: 4.8326, top1_error: 0.9877, top5_error: 0.9423
[smoke_dsnet_ib_mbv2_slimlast] 150.4s	val	0.80	-1/2: loss: 4.8263, top1_error: 0.9855, top5_error: 0.9392
[smoke_dsnet_ib_mbv2_slimlast] 156.3s	val	0.85	-1/2: loss: 4.7965, top1_error: 0.9829, top5_error: 0.9355
[smoke_dsnet_ib_mbv2_slimlast] 163.5s	val	0.90	-1/2: loss: 4.7573, top1_error: 0.9808, top5_error: 0.9203
[smoke_dsnet_ib_mbv2_slimlast] 168.0s	val	0.95	-1/2: loss: 4.7143, top1_error: 0.9761, top5_error: 0.9041
[smoke_dsnet_ib_mbv2_slimlast] 172.9s	val	1.00	-1/2: loss: 4.7644, top1_error: 0.9727, top5_error: 0.8819
```
The R50 path was not re-smoked after the refactor (to keep the machine light); instead its
construction was checked to be unchanged (Changes item 1: same keys, bit-identical initial
values), and it shares `train.py` / `protocol.py` / `train_slim.py` with the two settings above,
whose runs exercised the same training, checkpoint, resume and evaluation code.
Near chance after 20 steps, as expected; the widest widths have moved off chance first. All lines
are in the `scripts/collect_results.py` / notebook `VAL_LINE` format.
Checked in the R50 checkpoints (earlier smoke): next epoch 2; online and EMA state dicts both
present (2,705 keys, 2,709 with H-IB); 161 SGD momentum buffers (165 with H-IB), Nesterov, two
groups (wd 5e-4 on the 54 conv/linear weights, wd 0 on the 107 BN parameters and biases). H-IB
really trains: the two 2048x2048 predictor weights, which get gradient only from L_HIB, have
non-zero momentum buffers (norms 0.16, 0.14) and differ from their EMA copies. The LR after resume
continues the schedule (0.2 at the first step of epoch 1, the end of warm-up).
Not run locally: a full epoch, `--eval_only` (same code path as the final evaluation, after
`load_checkpoint`), the ImageNet path, upstream's `validate_slim`/`CheckpointSaver` path (i.e.
a run without `--skip_upstream_test`), a Tiny ImageNet run that has to decode the JPEGs or find the
data under `/kaggle/input` (the local run read our existing cache), `--hib` on the two new models.

## Changes and deviations
Code (all in `dsnet.patch`):
1. `dyn_slim/models/dyn_slim_resnet.py` (new): `ds_resnet50_cifar` (DS-ResNet-50 for 32x32) and
   `ds_resnet18_tiny` (DS-ResNet-18 for 64x64), and `dyn_slim/models/dyn_slim_mobilenetv2.py`
   (new): `ds_mobilenetv2_cifar_slimlast`; all built from their
   `DSConv2d`/`DSpwConv2d`/`DSdwConv2d`/`DSBatchNorm2d`/`DSLinear`/`DSAdaptiveAvgPool2d` and their
   `set_exist_attr`/`set_mode` conventions (the release ships only DS-MBNet). Checked for all
   three against our in-repo model (ours built with `utils.config.FLAGS` set as in the reference
   configs: depth 50 / depth 18 + `stem_stride 2` / `slim_last True`): at each of the 16 widths,
   every executed conv / BN / linear layer (41 on R18, 105 on MBV2, 107 on R50) has the same input
   and output channels, output spatial size, groups, kernel, stride and padding; the parameter
   counts at 1.0 are equal (23,705,252 / 11,271,432 / 2,351,972); and with our model's weights
   copied into the DS model (same registration order and shapes), the logits of a random batch
   agree exactly (max difference 0) at all 16 widths in train-mode BN. The check was confirmed to
   fail on purpose-made errors (stem stride 1 on R18: shape mismatches, logits off by 40%; one
   MobileNetV2 skip removed: logits off by 75%). The R50 refactor (shared `UniformChoiceMixin`,
   `block`/`stem_stride` arguments) leaves `ds_resnet50_cifar` unchanged: same 2,705 state-dict
   keys (2,709 with H-IB) and bit-identical initial values for the same seed as the previous
   patch.
   - R50 (against `models/us_resnet.py`, depth 50): 3x3 stride-1 stem, no max-pool, bottlenecks
     [3,4,6,3] (1x1, 3x3 with the stride, 1x1 x4), 1x1 conv + BN projection shortcut where stride
     or width changes, ReLU after the sum, global average pool, linear classifier with bias;
     channels at width w are `make_divisible(C*w)` (divisor 8; their `min_value` is 8 instead of
     our 1, which never binds in any of the three models: the smallest rounded value is 4 -> 8 in
     both). E.g. stem 16/24/24/24/32/.../64, pooled 512/616/.../2048.
   - R18 Tiny (against `models/us_resnet.py`, depth 18, `stem_stride 2`): 3x3 stem at stride 2,
     no max-pool (stages at 32, 16, 8, 4 px from 64x64), basic blocks [2,2,2,2] (3x3 with the
     stride + BN + ReLU, 3x3 + BN), identity skip in the first block (64 -> 64, stride 1), 1x1 conv
     + BN projection in the first block of stages 2-4, ReLU after the sum, global average pool,
     linear classifier (200 classes). Stem 16/24/24/24/32/.../64, pooled 128/152/.../512.
   - MBV2 slim_last (against `models/us_mobilenet_v2_cifar.py`, `slim_last True`): 3x3 stride-1
     stem to 32, inverted residuals (1,16,1,1), (6,24,2,1), (6,32,3,2), (6,64,4,2), (6,96,3,1),
     (6,160,3,2), (6,320,1,1) (CIFAR: the second stage keeps stride 1), ReLU6 everywhere, identity
     skip only where stride 1 and in == out (no projection, as ours; upstream's
     `DSInvertedResidual` would add an avg-pool + 1x1 projection everywhere else, so it is not
     used), last 1x1 to 1280 slimmed (`make_divisible(1280*w)`, 320/384/.../1280) with its BN and
     the classifier input. Expansion and depthwise channels use our ratio rounding,
     `make_divisible(C_in*t*w/t)*t` (= t x the block's input channels at that width), written
     with the same floating-point expression as ours. The depthwise conv is their `DSdwConv2d`
     (groups = the running channel count, as in their DS-MobileNetV1: `DSConv2d` slices
     `weight[:c_out, :c_in]` and sets `groups = c_out` when built with groups = channels). Stem
     8/16/16/.../32 (32*0.25 = 8). Upstream's `DSInvertedResidual` also carries an SE attention
     module (`MultiHeadGate`), which a standard MobileNetV2 and ours do not have: not used.
   - BN: `DSBatchNorm2d` = one shared affine (sliced) plus private running statistics per channel
     choice (the stats are found by channel count, so widths that round to the same count in a
     layer, e.g. 0.30/0.35/0.40 in the stem, share them). Ours: shared affine, batch statistics in
     training. Training uses batch statistics in both and the evaluation recalibrates per width,
     so only the stored stats differ.
   - Initialisation is their `efficientnet_init_weights` (convs N(0, 2/(k^2 * C_out / groups)):
     as ours for regular convs, but the depthwise convs of MBV2 get N(0, 2/9) where ours use
     N(0, 2/(9 C)); BN 1/0; classifier U(+-1/sqrt(classes)) with zero bias, ours N(0, 0.01)) plus
     their `init_residual_norm`: gamma of the last BN of every residual block = 0 (bottleneck bn3,
     basic block bn2, and the MBV2 blocks that have an identity skip; their residual blocks do
     this, and DS-Net++ sec. 5.3 states it for the CNN supernets; an MBV2 block without a skip
     keeps gamma 1, since zeroing it would zero its output). Ours starts every gamma at 1.
   - No SE / `MultiHeadGate` in any of the three: no gate is trained here.
   - Width space: one channel choice for all layers, stem included. The paper's DS-ResNet keeps
     the stem at 64 and has one gate per stage with ratios {0.25, 0.5, 0.75, 1} (256 paths); their
     MBNet 'uniform' mode slims only the gated stages and keeps the first ones at their smallest
     width. Here the 16 protocol widths in every layer, as the protocol needs.
2. `dyn_slim/apis/train_slim.py`: `--max_iters_per_epoch` (smoke only); `--hib` (H-IB term, see
   Method; logs `Loss_HIB_smallest`); `--clip-grad` (global norm, after their zero-gradient filter,
   before `optimizer.step()`); the step timer; the per-sample reseeding gets the run's seed as an
   offset (item 13); `from timm.utils import *` replaced by explicit names (the timm >= 0.9
   `random` shadowing crash, see Setup).
3. `train.py`: timm import shims; `--dataset cifar100 | tinyimagenet` path (protocol loaders from
   `protocol.py` built before the scheduler, `--num-classes` checked against the dataset,
   ImageNet loaders unchanged under `if`); the positional `data` is optional;
   `--sched-on-updates` / `--warmup-prefix` (timm's own scheduler options, item 8); resume via
   `protocol.load_checkpoint` after the EMA model exists, with the per-update LR re-stepped;
   per-epoch checkpoint, `--stop_at_epoch`, `--eval_only`, `--skip_upstream_test`; epoch timing /
   peak-memory line; model profiling at the dataset's image size (32 / 64 px); the 16-width
   evaluation (+ `--eval_ema`). Upstream's `--resume`/`CheckpointSaver` path is still there and
   untouched.
4. `dyn_slim/models/dyn_slim_blocks.py`, `dyn_slim_net.py`, `dyn_slim/utils/model_profiling.py`:
   timm imports from `timm.layers` with a fallback to `timm.models.layers`; `register_model` from
   `timm.models`. `dyn_slim/models/__init__.py` imports the new models so they register.
5. `protocol.py` (new): CIFAR-100 loaders (RandomCrop 32 pad 4 + HFlip, mean/std
   (0.5071,0.4865,0.4409)/(0.2673,0.2564,0.2762)); Tiny ImageNet loaders copied from our
   `data/tinyimagenet.py` (RandomCrop 64 pad 8 + HFlip, mean/std (0.4802,0.4481,0.3975)/
   (0.2770,0.2691,0.2821); test = the official val split with labels from `val_annotations.txt`,
   label = index of the class id in sorted `wnids.txt`; the same locate / md5 check / decoded
   cache, see Setup); one difference in mechanics only: the cached images are opened as a
   read-only memory map in each process instead of being held in memory, so a spawned loader
   worker gets a file path, not a 1.2 GB copy (same images, labels and order); `DSNET_WORKERS`,
   persistent workers; checkpoint save/load, BN recalibration, evaluation and the protocol lines
   (as LCS's `protocol.py`).
6. `configs/cifar100_ds_resnet50_uniform.yml`, `configs/tiny_ds_resnet18_uniform.yml`,
   `configs/cifar100_ds_mobilenetv2_slimlast_uniform.yml` (new): the recipe below, in their
   config format; they differ only in `dataset`, `model`, `num_classes`.

Recipe (ours vs paper / upstream):
7. CIFAR-100 32x32 or Tiny ImageNet 64x64, 100 epochs (paper 150 on ImageNet, upstream config
   150 + 10 cool-down), batch 256 (512), lr 0.2 (paper: 0.2 for DS-ResNet at 512, 0.08 for
   DS-MBNet; we keep 0.2 for the MobileNetV2 too, as our MBV2 runs do), Nesterov SGD momentum 0.9
   (timm 'sgd', as upstream), wd 5e-4 (1e-4), not applied to BN parameters and biases (timm
   `create_optimizer`, as upstream; ours does the same), grad clip 5 (none upstream), label
   smoothing 0 (0.1), no RandAugment / random erasing / colour jitter (upstream rand-m9-mstd0.5,
   reprob 0.2), dropout 0 (0.2 before the classifier on the largest), fp32.
8. LR per step: timm `CosineLRScheduler` in per-update mode with warm-up prefix: linear from 0
   to 0.2 over 5 epochs (975 updates on CIFAR-100, 1,950 on Tiny ImageNet; the very first update
   runs at lr 0), then cosine to 0 over the other 95 epochs, as our train.py. Upstream: per epoch,
   warm-up from 1e-5 (so epoch 0 runs at 1e-5), cosine over the full length including warm-up,
   min lr 1e-5, then 10 cool-down epochs.
9. `ensemble_ib` false, as in their released configs and the code default. The paper trains
   DS-ResNet with E-IB (ensemble target for the smallest); its ablation (DS-MBNet) gives +0.2 on
   the smallest and +0.3 on the largest over IB.
10. Uniform widths drawn from {0.30, ..., 1.00} (`randint(1, 15)`, their rule), not from all
    sixteen; 0.25 and 1.00 are trained every step anyway.
11. Training loader `drop_last=True` (timm's training loader): 195 steps of 256 per epoch on
    CIFAR-100 (80 images dropped per epoch), 390 on Tiny ImageNet (160 dropped). Ours keeps the
    last partial batch.
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
- Speed: 1.24 s/step on R50 is twice our US-Net. On a T4 each R50 branch needs one or two resumes.
- Steady s/step is from few tuned steps with per-step logging, not a full epoch (R50: 18 / 6;
  Tiny R18: 5; MBV2: 6).
- ReLU vs ReLU6: the logit-equality check cannot tell them apart at initialisation (activations
  rarely exceed 6 there); the MobileNetV2's ReLU6 was checked by reading the code.
- lr 0.2 on MobileNetV2 is our recipe, not DS-Net's (0.08 for DS-MBNet at batch 512).
