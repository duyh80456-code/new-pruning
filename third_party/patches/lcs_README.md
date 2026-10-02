# LCS (official code) on CIFAR-100 / CIFAR ResNet-50, structured width, under our protocol

Paper: Nunez et al., "LCS: Learning Compressible Subspaces for Adaptive Network Compression at
Inference Time", WACV 2023. Upstream: https://github.com/apple/learning-compressible-subspaces
Pinned commit: `e6d3924368faccbdfd3d89c4a4735dba947275c9` (HEAD, "Update root files"; code is the
2021-10-27 initial release `308f521`). Our changes: `lcs.patch` (= `git diff` against that commit,
new file `protocol.py` included), applied with `git apply`. Checked: applies cleanly to a fresh
checkout and reproduces the tested tree byte for byte.

## Method
`--method lcs_l` = the paper's LCS+L, the line subspace (`configs/structured_sparsity/lcs_l.yaml`,
trained by `train_curve.py`, regime "us", `num_points: 2`). Width w in [0.25, 1.0] maps to the line
coordinate alpha = (w - 0.25)/0.75; w = 0.25 is endpoint 0, w = 1.0 is endpoint 1. Each step is
the sandwich rule with 4 samples (alpha 0, 1 and two U(0,1)), loss = **sum** of the 4 CE losses
(no in-place distillation), plus the paper's eq. 3 regulariser beta*cos^2(endpoint0, endpoint1),
beta = 1, over the norm layers (`apply_beta_to_norm: True`).

Other structured methods in the code: `lcs_p` (LCS+P, the "degenerate" point subspace) is
`train_indep.py` regime "us" with `AdaptiveIN`, i.e. US-Net-style sandwich training with instance
norm; `us` is the in-codebase US-Net (same trainer and sandwich of 4, summed CE, `AdaptiveBN` with
one set of running stats shared by all widths and **no** BN recalibration, no in-place
distillation); `ns` (slimmable, 4 fixed widths) and `lec` (network slimming) also exist. So
`lcs_p` and `us` differ only in the norm layer.

**Parameter note (from code).** In structured `lcs_l` the convolutions are plain `AdaptiveConv2d`
(one weight tensor, sliced to the first round(w*C) channels); only the norm layers are lines
(`LinesAdaptiveIN`/`LinesAdaptiveBN`: `weight,bias` = endpoint 0, `weight1,bias1` = endpoint 1,
interpolated by alpha). The beta term likewise only touches norms (it skips `AdaptiveConv2d`).
Stored parameters, CIFAR R50 / 100 classes: **23,758,272** = 23,705,152 for one point
(23,652,032 conv incl. the 1x1-conv classifier, which has no bias) + 53,120 for the second norm
endpoint (+0.22 %). Not 2x. (`LinesAdaptiveConv2d` exists in modules.py but no structured config
uses it.)

## Setup (Kaggle or local)
```bash
git clone https://github.com/apple/learning-compressible-subspaces && cd learning-compressible-subspaces
git checkout e6d3924368faccbdfd3d89c4a4735dba947275c9
git apply /path/to/lcs.patch
pip install pyyaml        # only extra dependency; preinstalled on Kaggle
```
Do not `pip install -r requirements.txt` (pins torch 1.8.1+cu111). Tested with Python 3.10.20,
torch 2.11.0+cu128, torchvision 0.26.0, numpy 2.2.6, PyYAML 6.0.3 on an RTX 5060 Ti 16 GB.

`--data_dir` is the folder that *contains* `cifar-100-python/` (torchvision `CIFAR100(...,
download=True)` only md5-checks existing files; a read-only Kaggle input works).
Loader workers: env `LCS_WORKERS` (default 2). Run from the repo root (configs are relative paths).

## Commands
The model is wrapped in `nn.DataParallel` upstream, so it would spread over every visible GPU:
pin each run with `CUDA_VISIBLE_DEVICES`.
```bash
D=/kaggle/input/cifar100          # contains cifar-100-python/
CK=/kaggle/working/ckpt
W=0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00
COMMON="--model cresnet50 --dataset cifar100 --method lcs_l --data_dir $D \
  --epochs 100 --batch_size 256 --learning_rate 0.2 --momentum 0.9 --nesterov --weight_decay 5e-4 \
  --width_factor_limits 0.25,1.0 --eval_width_factors $W --skip_upstream_test"
```
(a) Published default norm (line-subspace instance norm, `LinesAdaptiveIN`; no running stats, so no
recalibration exists or is needed):
```bash
CUDA_VISIBLE_DEVICES=0 LCS_WORKERS=2 python train_structured.py $COMMON \
  --save_dir /kaggle/working/lcs_l_in --ckpt_dir $CK/lcs_l_in --log_prefix lcs_l_in
```
(b) BatchNorm (`--norm BN` -> `LinesAdaptiveBN`, added by the patch) + per-width recalibration at eval:
```bash
CUDA_VISIBLE_DEVICES=1 LCS_WORKERS=2 python train_structured.py $COMMON --norm BN --recal_batches 20 \
  --save_dir /kaggle/working/lcs_l_bn --ckpt_dir $CK/lcs_l_bn --log_prefix lcs_l_bn
```
At the end of training each run prints the 16 protocol lines (tab separated):
`[lcs_l_bn] 15912.3s	val	0.25	-1/100: loss: 1.2345, top1_error: 0.4321, top5_error: 0.1234`
(elapsed = seconds since the process started). Recalibration per width: reset running stats,
forward `--recal_batches` (default 20) shuffled, augmented training batches of 256 in train mode
with a cumulative average (momentum 1/(i+1), what momentum=None does), then eval on the 10k test
set. `--recal_batches 0` evaluates with the trained shared running stats (upstream BN behaviour).

`--skip_upstream_test` drops upstream's `test()` at epochs 0,20,...,80 and at the end (16 widths
each, with per-width model profiling, and `model_N` saves); without it those also run and print
`Test set (width_factor = ...)` lines (for BN: shared stats, no recalibration).

(c) Evaluation only, from the resume checkpoint (same flags + `--eval_only`):
```bash
CUDA_VISIBLE_DEVICES=1 python train_structured.py $COMMON --norm BN --save_dir /kaggle/working/lcs_l_bn \
  --ckpt_dir $CK/lcs_l_bn --log_prefix lcs_l_bn --eval_only [--recal_batches 0]
```
It raises if no checkpoint exists. ~100 s for 16 widths (IN), ~65 s (BN incl. recalibration).

(d) Resume. After every epoch `{ckpt_dir}/last.pt` is written atomically (`.tmp` + `os.replace`):
model (incl. BN buffers), SGD state (momentum buffers), next epoch, and Python/NumPy/torch/CUDA
RNG states. The LR schedule is a pure function of the epoch (upstream `cosine_lr`), so there is no
scheduler state to save. If `last.pt` exists, the identical command resumes from the next epoch
(prints `Resumed from ... at epoch N`). For Kaggle's 12 h limit add `--stop_at_epoch N` to exit
cleanly after epoch N (no eval); copy `last.pt` into the new session's `--ckpt_dir` and rerun.
Granularity is one epoch.

## Timing (RTX 5060 Ti, fp32, batch 256, 2 workers, steady state over steps 4-10 of each epoch)
| variant | s/step | 196 steps/epoch | 100 epochs |
|---|---|---|---|
| (a) IN | 1.63-1.71 | ~330 s | ~9.2 h |
| (b) BN | 0.80-0.81 | ~158 s | ~4.4 h |

Peak memory 6.5-6.6 GiB. Each step is 4 forward/backward passes. Our US-Net R50 is ~0.59 s/step
here. IN is 2x slower than BN (`F.instance_norm` on N*C channels). A T4 is ~2-3x slower: expect
~9-13 h for BN and ~18-27 h for IN, i.e. **at least one resume for BN and two or more for IN**.

## Smoke test actually run (local GPU)
Per variant: run 1 `--epochs 2 --max_iters_per_epoch 10 --stop_at_epoch 1` (10 steps, checkpoint,
exit), run 2 same command without `--stop_at_epoch` (resumes at epoch 1 with lr 0.08 = warm-up
epoch 1, 10 steps, checkpoint, 16-width eval). BN run 2 also ran upstream's final `test()`.
```
IN  run1: Epoch 0 ... (steady 1.710 s/step over 7 steps), peak mem 6.57 GiB / Stopping after epoch 0
IN  run2: Resumed from .../last.pt at epoch 1 ... (steady 1.630 s/step)
[smoke_lcs_l_in] 40.9s	val	0.25	-1/2: loss: 6.7557, top1_error: 0.9900, top5_error: 0.9505
... 16 lines ...
[smoke_lcs_l_in] 140.3s	val	1.00	-1/2: loss: 15.7692, top1_error: 0.9900, top5_error: 0.9494
BN  run1: (steady 0.808 s/step), run2: Resumed ... at epoch 1 (steady 0.803 s/step)
upstream test, shared stats:  width 0.25 loss 4.61 ... width 1.0 loss 45254.99
[smoke_lcs_l_bn] 104.3s	val	0.25	-1/2: loss: 5.0420, top1_error: 0.9878, top5_error: 0.9467
... 16 lines ...
[smoke_lcs_l_bn] 168.3s	val	1.00	-1/2: loss: 6.8287, top1_error: 0.9937, top5_error: 0.9439
```
(near-chance after 20 steps, as expected). The gap between the upstream test and the recalibrated
eval at width 1.0 shows the recalibration branch runs. The checkpoint holds 266 momentum buffers,
nesterov=True, wd 5e-4, epoch 2. `--eval_only` on the BN checkpoint also ran (16 lines, width 1.00
loss 6.8952; differs slightly from run 2 because the 20 recalibration batches are random).
Not run locally: a full epoch, `lcs_p`/`us` after the edits.

## Changes and deviations
Code (all in `lcs.patch`):
1. `models/networks/resnet.py`: `CResNet50` = upstream `ResNet` with `Bottleneck` [3,4,6,3]
   (stride in the 3x3, 1x1 conv + norm projection shortcuts) and a CIFAR stem: 3x3 stride-1 conv,
   no max-pool (32/16/8/4 px); `cifar100: 100` classes. Built from LCS's own builder/layers.
   Upstream ResNet details kept: classifier is a 1x1 `AdaptiveConv2d` without bias, kaiming-normal
   init everywhere, norm gammas = 1 at both endpoints, no zero-init of the last norm.
2. `models/modules.py`: `LinesAdaptiveBN` (BN twin of `LinesAdaptiveIN`); upstream `--norm BN` with
   `lcs_l` would ask for this missing class. `utils.py`: the "us" regime's assert that BN
   norm_kwargs carry `width_factors_list` becomes a default of None (lcs_l.yaml has none).
3. `utils.py`/`args.py`: CIFAR-100 loader (RandomCrop 32 pad 4 + HFlip, mean/std
   (0.5071,0.4865,0.4409)/(0.2673,0.2564,0.2762), `--data_dir`, `LCS_WORKERS`), dataset/model
   validation, `--nesterov`, `--seed`.
4. **Per-width backward** in the sandwich loop (`train_curve.py`, also `train_indep.py`): upstream
   sums the 4 losses and calls backward once, which keeps 4 sets of activations and overflows 16 GB
   at R50/bs256 (spills into shared memory, minutes per step). Backward per width gives the same
   gradient (backward of a sum); the beta term gets its own backward.
5. **`cudnn.benchmark = False`** (upstream True): with continuous random widths almost every step
   has new conv shapes; autotuning took ~20 s per new shape here (measured).
6. `get_training_params.py`: upstream sets `test_freq = args.epochs` (bug); now `args.test_freq`.
7. `protocol.py` + trainer hooks: resume checkpoint, `--stop_at_epoch`, `--eval_only`,
   `--skip_upstream_test`, `--max_iters_per_epoch` (smoke only), BN recalibration, protocol lines,
   steady s/step print. `ns_logging` profiling made non-verbose.

Recipe (ours vs paper/upstream):
8. 100 epochs (upstream 200), batch 256 (128), lr 0.2 (0.1), nesterov (upstream plain momentum),
   momentum 0.9, wd 5e-4 on **all** parameters incl. norms and both endpoints (upstream).
9. LR schedule is upstream's, **per epoch**: warm-up 5 epochs at 0.04, 0.08, ..., 0.2 (starts at
   lr/5, not 0, no per-step ramp), then cosine per epoch to ~0 at epoch 99.
10. Loss is the sum of 4 CE terms, no in-place distillation, plus beta = 1 cosine regulariser.
11. Width slicing `round(w*C)` per layer (upstream `AdaptiveConv2d`), stem included, classifier
    input sliced; no rounding to multiples of 8. Middle widths are continuous.
12. Default norm is IN (published); IN has no running stats, so variant (a) has no recalibration.
    Variant (b) trains BN with shared running stats (momentum 0.1) and recalibrates at eval.
13. Upstream's DataLoader has no `drop_last` (196 steps/epoch, last batch 80); test loader shuffled
    (upstream, needed by its BN statistics probe; no effect on accuracy).
14. fp32, single GPU; DataParallel wrapper kept (pass-through on one GPU).
