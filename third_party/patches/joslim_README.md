# Joslim (official code) on CIFAR-100 / CIFAR ResNet-50 under our protocol

Upstream: https://github.com/enyac-group/Joslim (formerly cmu-enyac/Joslim)
Pinned commit: `9b743d9b50ade9fc2cd33d255dfbb936e4894733` (2021-06-30, "update", the last commit)
Our changes: `joslim.patch` (= `git diff` against that commit), applied with `git apply`.

## Setup (Kaggle or local)

```bash
git clone https://github.com/enyac-group/Joslim && cd Joslim
git checkout 9b743d9b50ade9fc2cd33d255dfbb936e4894733
git apply /path/to/joslim.patch
pip install botorch==0.16.1 gpytorch==1.15.2 linear_operator==0.6.1   # tensorboard is preinstalled on Kaggle; else: pip install tensorboard
```

Tested with: Python 3.10.20, torch 2.11.0+cu128, torchvision 0.26.0, botorch 0.16.1, gpytorch 1.15.2,
linear_operator 0.6.1 (pulls pyro-ppl 1.9.1, multipledispatch 1.0.0, pyre-extensions 0.0.32),
tensorboard 2.21.0, numpy 2.2.6, scipy 1.15.3. botorch 0.16.1 needs torch >= 2.0.1 and Python >= 3.10.

botorch: Joslim imports `botorch.fit.fit_gpytorch_model`, removed in botorch >= 0.12. We use a
shim, not an old pin: `try: from botorch.fit import fit_gpytorch_model / except ImportError: from
botorch.fit import fit_gpytorch_mll as fit_gpytorch_model`. `fit_gpytorch_model` was a thin
deprecated wrapper around `fit_gpytorch_mll` (same MLL fit, L-BFGS-B via scipy), so the fit is the
same procedure. An old botorch (0.6-0.9) would pin an old gpytorch/linear_operator against torch 2.x
and is more fragile than the shim.

`--datapath` is the folder that *contains* `cifar-100-python/` (torchvision `CIFAR100(root, download=True)`:
it only verifies md5s when the files exist, downloads otherwise). A read-only Kaggle input dataset works.

## Commands

Common flags (both runs). Launch without a distributed launcher: with no `WORLD_SIZE` in the
environment Joslim runs single-process on `cuda:current_device()`. One run per T4 via
`CUDA_VISIBLE_DEVICES`. (`--local-rank` is also accepted now, for `torch.distributed.run`.)

```bash
D=/kaggle/input/cifar100          # contains cifar-100-python/
CK=/kaggle/working/ckpt           # persisted; see "Resume"
COMMON="--dataset CIFAR100 --datapath $D --network slim_resnet50_cifar \
  --epochs 100 --warmup 5 --baselr 0.2 --scheduler cosine_decay --batch_size 256 \
  --wd 5e-4 --mmt 0.9 --nesterov --label_smoothing 0 --lower_channel 0.25 \
  --num_sampled_arch 2 --baseline -3 --print_freq 100 --ckpt_dir $CK"
```

(a) Joslim:
```bash
CUDA_VISIBLE_DEVICES=0 python joslim.py --name joslim_r50 $COMMON --tau 195 --prior_points 20
```

(b) Slim = US-Net, the paper's "Slim" baseline:
```bash
CUDA_VISIBLE_DEVICES=1 python joslim.py --name slim_r50 $COMMON --slim --slim_uniform --tau 1
```
(`--slim` alone is the code's literal slim mode: independent random width per layer. See deviations.)

(c) Evaluation. 16 uniform widths 0.25..1.00 step 0.05 (every layer at `int(w*C)` via
`decode_wm(np.ones(21)*w)`), BN recalibrated per width with Joslim's own procedure (BN stats reset,
momentum=None i.e. exact cumulative average, train mode, augmented shuffled train loader) but on
`--bn_cal_batch_num 20` batches of 256 (our protocol; the default). `--bn_cal_batch_num 0` restores
upstream's full-train-set pass (~35 s/width locally instead of ~18 s). The cap applies to every
recalibration in eval_checkpoints.py (Full, Smallest, Pareto, uniform). Then test top-1/top-5/CE.
Prints our parser lines:
```bash
python eval_checkpoints.py --name joslim_r50 --dataset CIFAR100 --datapath $D --network slim_resnet50_cifar \
  --batch_size 256 --lower_channel 0.25 --ckpt_dir $CK --uniform --tag joslim_r50
python eval_checkpoints.py --name slim_r50   --dataset CIFAR100 --datapath $D --network slim_resnet50_cifar \
  --batch_size 256 --lower_channel 0.25 --ckpt_dir $CK --uniform --tag slim_r50
```
Line format (tab separated; `[tag] ` prefix only when `--tag` is given, which `scripts/collect_results.py`
needs to name the run):
`[joslim_r50] 123.4s	val	0.25	-1/100: loss: 1.2345, top1_error: 0.4321, top5_error: 0.1234`
followed by `WM: 0.25 MFLOPs: ... (... %)`.

Joslim's Pareto set (Joslim run only; the slim run has no visited-architecture history and this mode
crashes on it): non-dominated sort of all visited per-layer configs by (recorded loss, FLOPs), each
one BN-recalibrated and tested; prints `(i/n) Acc: top1 top5, MFLOPs: x (y %)` per config and writes
`results/joslim_r50_eval_pareto.txt` (rows: FLOPs ratio, top-1, top-5) and
`$CK/joslim_r50_sample_pool.pt` (the widths):
```bash
python eval_checkpoints.py --name joslim_r50 --dataset CIFAR100 --datapath $D --network slim_resnet50_cifar \
  --batch_size 256 --lower_channel 0.25 --ckpt_dir $CK
```
FLOPs are conv MACs (classifier is a 1x1 conv, included). Full CIFAR R50 = 1298.0 MMac; width 0.25 = 81.5 MMac (6.3 %).

(d) Resume. `joslim.py` saves `$CK/{name}.pt` after every epoch (now atomically, `.tmp` + `os.replace`)
and, if that file exists at start, resumes from the next epoch. To continue in a new Kaggle session,
copy the previous session's `{name}.pt` into `$CK` and rerun the identical command:
```bash
mkdir -p $CK && cp /kaggle/input/<previous-output>/ckpt/joslim_r50.pt $CK/
CUDA_VISIBLE_DEVICES=0 python joslim.py --name joslim_r50 $COMMON --tau 195 --prior_points 20
# prints "Loading checkpoint from epoch N"
```
What is restored: model weights incl. BN buffers, SGD state (momentum buffers), epoch (the LR is
a pure function of the global step, so warm-up/cosine continue exactly), and the full BO state: `X`
(visited width vectors), `Y` (loss, FLOPs ratio), `population_data`. The GPs are refit from X/Y at
every sampling event anyway, so nothing else of the GP needs saving. Not restored: Python/NumPy/torch
RNG states (the resumed run draws a different but statistically equivalent random stream) and the
TensorBoard writer (logs go to ./runs/{name} in the working directory). Granularity is one epoch.

## Choice of --tau

Iterations per epoch = floor(50000/256) = 195 (drop_last); 100 epochs = 19,500 steps.
Visited architectures = 19,500 / tau x num_sampled_arch. tau = 195 gives one sampling event per epoch,
100 events, **200 visited architectures** (the first 20 = 10 events are uniform-width priors,
`--prior_points 20` as in all upstream commands, then BO). The paper uses |H| = 1000
(upstream CIFAR command: 300 ep x 390 it / tau 235 x 2 = 996). We stay at 200 because:
(1) every event re-evaluates the loss of *every* visited architecture on one batch (temporal-
similarity calibration), so the cost is quadratic in |H|: with 1000 archs over 19,500 steps,
tau = 39 and the calibration alone would be ~500 events x ~500 forwards = 250k R50 forwards,
several times the training compute; (2) each event runs up to 2 x 10 acquisition optimisations
(binary search over the scalarisation weight; see Timing); (3) tau = 195 keeps
images-per-architecture (50k) close to the paper's CIFAR setting (235 x 128 = 30k), and
(4) tau = iters/epoch makes every resume point a sampling boundary, so the resumed epoch starts
with freshly sampled architectures (upstream restores the in-flight archs without the smallest one).

## Timing (RTX 5060 Ti, fp32, batch 256, 2 loader workers, measured over steps 10-50)

| mode | s/step | 195 steps | per-epoch extras | 100 epochs |
|---|---|---|---|---|
| Joslim (`--tau 195`) | 0.718 | 140 s | one sampling event: ~1-2.4 min BO + loss of every visited arch on one batch | ~5.5-7.5 h |
| Slim/US-Net (`--slim --slim_uniform --tau 1`) | 0.793 | 155 s | none | ~4.3 h |

Each step is 4 forward/backward passes (full width with CE; 2 sampled and the smallest, distilled
from the full-width logits). For reference our US-Net R50 does ~0.59 s/step here. Slim is slower per
step than Joslim because its widths change every step (Joslim reuses the same archs for tau steps).
BO cost per sampled architecture (2 per event; each up to 10 `optimize_acqf` calls in a binary
search over the scalarisation weight; float32 GP, scipy L-BFGS often ends "ABNORMAL" and botorch
retries), measured standalone: 32 s at 10 visited points, 52 s at 50, 71 s at 100, 61 s at 200.
That is ~1.5-3.5 h of BO over a run. T4 is roughly 2-3x slower in fp32: expect ~10-13 h for Slim and
more for Joslim (BO is scipy on the CPU plus small GPU kernels, so it does not scale with the GPU),
i.e. **both runs need at least one resume on Kaggle**. Evaluation: ~18 s per width with
`--bn_cal_batch_num 20` (16 widths + Full + Smallest ~ 6 min); the Pareto set adds ~18 s per config.

## Smoke test actually run (local GPU)

Joslim mode with `--max_iters 4 --tau 4 --prior_points 2` (so the GP/BO branch runs), 2 epochs, then
resume with `--epochs 3`, then `eval_checkpoints.py --uniform --tag smoke_joslim`:
```
Lower flops based on lower channel: 0.0627851864006715
Full MFLOPs: 1298.014
Epoch 0 | Time: 15.86s
Epoch 1 | Time: 69.43s                      (BO events)
--- rerun with --epochs 3
Loading checkpoint from epoch 1
Batch 0/195 | SuperLoss: 13.077, MinLoss: 4.378, LR: 0.0800   (step 390 of warm-up: 390/975*0.2)
Epoch 2 | Time: 178.97s
checkpoint: epoch 1 -> 2, X 4 -> 6 rows with the first 4 identical, Y/population 4 -> 6,
            160 SGD momentum buffers stored, weights changed
--- eval (near-chance numbers: 12 training steps)
[smoke_joslim] 16.7s	val	0.25	-1/100: loss: 18.1704, top1_error: 0.9900, top5_error: 0.9477
...  (16 lines)
[smoke_joslim] 306.6s	val	1.0	-1/100: loss: 30.6554, top1_error: 0.9909, top5_error: 0.9495
```
All 16 lines match the regex in `scripts/collect_results.py`, widths exactly 0.25..1.0.
Slim mode (`--slim --slim_uniform --tau 1`) ran 51 steps through epoch end and checkpoint save.
Not executed locally (the machine was overloaded): slim-mode resume and eval, and the Pareto eval
(without `--uniform`). Those paths are upstream code apart from the `--ckpt_dir`/`weights_only`
edits. A `--slim` checkpoint cannot go through the Pareto eval (it has no history and crashes), so
use `--uniform` only for it.

## Deviations from the paper's recipe / upstream defaults

Code changes (all in `joslim.patch`):
1. `model/slim_resnet.py`: `slim_resnet50_cifar` = upstream `slim_resnet50` ([3,4,6,3] bottleneck,
   1x1 conv + BN projection shortcuts, stride in the 3x3 conv) with a CIFAR stem: 3x3 conv stride 1,
   no maxpool (feature maps 32/16/8/4). 23.7M params, 1298.0 MMac.
2. `utils/drivers.py`: CIFAR-100 normalization changed from ImageNet mean/std (upstream) to
   (0.5071,0.4865,0.4409)/(0.2673,0.2564,0.2762). The ImageNet-LMDB import is optional (avoids
   fire/lmdb/umsgpack/pyarrow). The `JOSLIM_WORKERS` env var (default 8, as upstream) sets loader workers.
3. `joslim.py`: botorch shim; `--ckpt_dir`; atomic checkpoint save; `torch.load(...,
   weights_only=False, map_location='cpu')` (torch >= 2.6 defaults to weights_only=True, which cannot
   load the numpy arrays in the checkpoint); `--local-rank` alias; `--slim_uniform`; `--max_iters`
   (smoke tests only, default 0 = off).
4. `eval_checkpoints.py`: botorch shim, `--ckpt_dir`, `weights_only=False`; removed an
   `args.local_rank` reference that crashed at the very end (eval has no such arg); rewrote the
   `--uniform` branch, which upstream cannot run (`selfmodel`, `args.upper_flops`, `ratio` undefined),
   into the 16-width protocol with CE loss and our line format; `--tag`; `--bn_cal_batch_num`
   (default 20; upstream = whole train set).

Recipe (ours vs paper/upstream):
5. 100 epochs, batch 256, lr 0.2 (`--baselr 0.2`; lr = baselr*batch/256), nesterov, wd 5e-4,
   5-epoch per-step linear warm-up from 0, per-step cosine to 0, label smoothing 0, fp32.
   Upstream CIFAR: 300 epochs (paper 200), batch 128, baselr 0.1 (= lr 0.05), linear decay,
   lower_channel 0.42.
6. Weight decay skips BN parameters and biases (Joslim groups params by the names `.bn`/`.bias`).
   The stem BN (`features.0.norm`) does get wd because its name has no `.bn`. Conv biases exist as
   parameters but SlimConv2d never uses them, so the classifier (a 1x1 conv) has no bias.
7. Initialisation is upstream's: kaiming fan_out for all convs including the classifier, and the last
   BN of each block is not zero-initialised. The initial CE is ~16 instead of ln(100)=4.6. In the
   50-step timing run it fell to ~6 by step 40 during warm-up, so the network trains, but this head
   differs from a standard Linear.
8. Width slicing: `int(w*C)` (floor) per layer, no rounding to multiples of 8; the stem shares the
   first width; in per-layer configs a stage's residual width is forced >= the previous stage's. For
   uniform widths this reduces to plain slicing. Reported FLOPs are conv MACs.
9. `--num_sampled_arch 2`, `--baseline -3`, `--prior_points 20`, `--beta 0.1` (default), as in every
   upstream command; `--tau 195` (200 visited archs, not 1000; see above).
10. Joslim's per-architecture losses (the temporal-similarity calibration that feeds the GP, and
    through it the Pareto selection in eval) are computed on one batch of `val_loader`, which upstream
    sets to the **test loader** (`val_loader = test_loader` in drivers.py). This is official
    behaviour and is kept, although the paper text says "training loss". It exposes 256 test images
    per event to width selection. It does not affect BN recalibration or the training gradients, but
    it does steer which widths Joslim trains. A held-out train split would be a one-line fix in
    drivers.py; it was not made, to stay official.
11. `--slim`: upstream samples an independent random width **per layer** (21 dims), but the paper
    defines Slim as US-Net with one width multiplier for all layers. The new `--slim_uniform` gives
    the paper's Slim. `--tau 1` resamples the two middle widths every step, as US-Net does (with
    tau > 1 upstream reuses them for tau steps). The smallest (0.25) and full widths are trained
    every step. Middle widths are continuous U(0.25,1), sliced with floor.
12. BN recalibration uses augmented training images in train mode with a cumulative average
    (Joslim's procedure), on 20 random batches of 256.
13. Resume is per epoch. RNG states are not checkpointed (upstream). `drop_last=True` drops 80
    training images per epoch.

## Kaggle risks
- botorch/gpytorch: pin `botorch==0.16.1 gpytorch==1.15.2 linear_operator==0.6.1` (needs torch >= 2.0.1
  and Python >= 3.10), and check that the pip install did not replace Kaggle's torch.
- BO overhead runs partly on the CPU (scipy L-BFGS). Kaggle's 4 vCPUs, shared by two parallel runs,
  may stretch the 1-2.4 min per event. Only the Joslim run has this overhead.
- 12 h limit: both runs need a resume. Copy `{name}.pt` from the previous version's output.
- Two runs in parallel with the default 8 workers each on 4 vCPUs: set `JOSLIM_WORKERS=2` if
  host memory runs short.

## Validation split (changed after the first package)

Upstream sets `val_loader = test_loader`, so the per-architecture losses that
drive the GP and the Pareto selection were computed on test batches. The patch
now scores them on a fixed 5,000-image subset of the CIFAR training set (seed 0,
test transform), and the test set is read only by the final evaluation.
Training still uses all 50,000 training images. `JOSLIM_VAL=test` restores the
upstream behaviour. The per-epoch test accuracy printed during training is a
monitor only and feeds no decision.
