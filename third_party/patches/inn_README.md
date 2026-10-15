# INN (official code), converted and fine-tuned, under our protocol: CIFAR-100 / ResNet-50, Tiny ImageNet / ResNet-18, CIFAR-100 / MobileNetV2 (slim_last)

Paper: Solodskikh et al., "Integral Neural Networks", CVPR 2023 (supplementary: Appendix D,
experiment details). Upstream: https://github.com/TheStageAI/TorchIntegral
Pinned commit: `84c5e89047f1d47bbf6c62065aa223d997f9b5e7` (HEAD, 2024-12-02, "Merge pull request
#39"). Our changes: `inn.patch` (= `git diff` against that commit, new files `cifar_resnet.py`,
`cifar_mobilenetv2.py`, `protocol.py`, `train_base.py`, `train_inn.py` included), applied with
`git apply`. Checked: applies cleanly to a fresh clone of that commit and reproduces the tested
tree byte for byte (with `core.autocrlf=false`; on Windows with `autocrlf=true`, `git apply`
writes CRLF files, harmless).

## Settings
`--arch` picks the network, `--dataset` the data (default: the arch's own). Each network is a
plain `torch.nn` copy of our in-repo slimmable model at width 1.0, and can be built at any width
w with exactly that model's channel counts (checked, below).

| `--arch` | default `--dataset` | network = our model at 1.0 | reference config | params at 1.0 (base / INN) | INN at 0.25 | integral groups |
|---|---|---|---|---|---|---|
| `r50` | `cifar100` | CIFAR ResNet-50 = `models/us_resnet.py` depth 50 | `apps/cifar100_bx_a_r50.yml` | 23,705,252 / 23,678,692 | 1,523,716 | 37 |
| `r18_tiny` | `tinyimagenet` | ResNet-18 for 64x64 = `us_resnet` depth 18, `stem_stride: 2` | `apps/cifar100_es_a_r18_tiny.yml` | 11,271,432 / 11,266,632 | 724,776 | 12 |
| `mbv2_slimlast` | `cifar100` | CIFAR MobileNetV2 = `models/us_mobilenet_v2_cifar.py`, `slim_last: True` | `apps/cifar100_ev_a_mbv2_slimlast.yml` | 2,351,972 / 2,334,916 | 187,124 | 25 |

- `r50`: 3x3 stride-1 stem, no max-pool (stages see 32, 16, 8, 4 pixels), bottlenecks [3,4,6,3]
  with the stride on the 3x3, 1x1 conv + BN projection where stride or width changes, ReLU after
  the sum, average pool, linear with bias.
- `r18_tiny`: the same stem at stride 2 for 64x64 inputs (stages again see 32, 16, 8, 4), no
  max-pool, basic blocks [2,2,2,2] (3x3 with the stride, 3x3), projection only where stride or
  width changes (not in `layer1.0`), 200 classes.
- `mbv2_slimlast`: 3x3 stride-1 stem to 32, inverted residuals (t,c,n,s) = (1,16,1,1),
  (6,24,2,1), (6,32,3,2), (6,64,4,2), (6,96,3,1), (6,160,3,2), (6,320,1,1) (ImageNet's second
  stride 2 becomes 1), each [1x1 expansion + BN + ReLU6, absent at t = 1] -> 3x3 depthwise
  (stride) + BN + ReLU6 -> 1x1 projection + BN, identity added where stride 1 and full-width
  in = out (no projection shortcut); last 1x1 to 1280 + BN + ReLU6 **slimmed with the width**,
  average pool, linear with bias. Channels at width w: stem, block outputs, last 1x1
  make_divisible(C*w); expansion and depthwise t * make_divisible(C_in*w) (US-Net's ratio
  rounding, as `us_mobilenet_v2_cifar` and DS-Net's `ds_mobilenetv2_cifar_slimlast`).

Init as ours (convs N(0, 2/(k^2 C_out)) incl. depthwise, BN 1/0, linear N(0, 0.01), bias 0);
`make_divisible` copied from ours. INN parameter counts: BN fused into conv biases (below).

## Method
INN does not train a slimmable network from scratch. It converts a **trained ordinary network**
into an integral one and fine-tunes that (the paper's "INN-init", pipeline A, Algorithm 1). So
there are two phases, each its own command:

1. `train_base.py`: the plain network trained with our recipe. It is also our *individual
   network at width 1.0*: it prints one protocol line at 1.00.
2. `train_inn.py`: loads that checkpoint and converts it exactly as the released examples do,
   `IntegralWrapper(init_from_discrete=True, fuse_bn=True)(model, [1,3,S,S], continuous_dims,
   discrete_dims)` (S = 32, 64 on Tiny ImageNet), plus our tracer class for depthwise convs:
   - **BatchNorm fused** into the preceding conv (their default `fuse_bn=True`, and Appendix D:
     "batch normalization layers were fused into parent convolution"). Every BN follows a conv
     with a single user, so all are fused (R50 53, R18 20, MBV2 52) and the INN has **no
     BatchNorm at any width**. Each conv gets a bias instead (R50: 23,705,252 - 53,120 BN +
     26,560 bias = 23,678,692).
   - **TSP channel permutation**: `NOptPermutation` (2-opt on the L1 total variation, 100 random
     restarts, threshold 0.001, their defaults) per group, over every tensor in the group, so the
     network function is unchanged. On BX's weights it cut the summed variation of the 37 R50
     groups from 20,429 to 6,971 (34 %).
   - **Continuous parametrization**: `GridSampleWeights1D/2D` (bicubic `grid_sample`,
     `align_corners=True`, `scale=1`: the stored tensor has the discrete weight's size, no extra
     parameters; `optimize_iters=0`, their default). Parametrized tensors: R50 107, R18 41,
     MBV2 105 (every conv/linear weight and every fused conv bias; the classifier's bias is
     discrete).
   - **Trapezoidal quadrature** along each weight's input-channel dimension (when it has > 3
     channels), scaled by their `train_volume` so that the full-size weights are the discrete ones
     exactly; with n of C points the sampled weights are multiplied by about (C-1)/(n-1). Not on
     depthwise weights (below).
   - Continuous dims: `standard_continuous_dims(model)` (every Conv2d/Linear weight along out and
     in channels, biases along out), a depthwise weight along dim 0 only, minus
     `discrete_dims = {conv1.weight: [1], fc.weight: [0], fc.bias: [0]}` (RGB input of the stem,
     the classes; `imagenet.py` keeps `fc` discrete the same way; same module names in all three
     networks). Their examples make only part of the network continuous (`stage3` of NIN,
     `layer4` of ResNet-18); here **every layer** is, as the protocol slims every layer.

   Checked: the converted INN at width 1.00 is the original network. R50 on BX's width-1.0
   network (loaded into `cifar_resnet.py`, its calibrated running stats fused): loss 1.4216 ->
   1.4215, top-1 error 0.2204 -> 0.2205, max |logit difference| 0.025 on a logit scale of 70 (fp32
   rounding of the fused BN and of `grid_sample`). All three on CPU with random weights and
   randomised BN statistics (so fusion and permutation are not trivial; 2-opt with 1 restart):
   max |logit difference| 9.5e-4 of 23 (R50), 1.5e-5 of 0.75 (R18), 6.4e-6 of 0.15 (MBV2).

**Depthwise convolutions (MobileNetV2).** The release has no handling for grouped convs (none of
its examples has one). Its conv hook ties the weight's dim 1 to the input channels and dim 0 to
the output channels; a depthwise weight is (C, 1, 3, 3), so the released tracer puts that
size-1 dim into the input's group and makes the output channels a group of their own. Checked on
our MBV2 with the released `IntegralTracer`: 42 groups instead of 25, e.g. block 1 split into
{`layers.1.conv1.weight[0]`, `layers.1.conv2.weight[1]`} and {`layers.1.conv2.weight[0]`,
`layers.1.conv3.weight[1]`}, two independent 96-point grids for one channel axis; resampled to half
size the forward fails (`Given groups=16, weight of size [16, 16, 3, 3], expected input[2, 16, 32,
32] to have 256 channels`). Handled in `train_inn.py`, without touching `torch_integral`:
- `ProtocolTracer` (a subclass of `IntegralTracer`, passed through the wrapper's own
  `integral_tracer_class` argument, used for both of its traces, before and after BN fusion):
  a `Conv2d` with groups > 1 must be depthwise (groups = in = out channels, else
  `NotImplementedError`) and gets a depthwise hook instead of `conv_linear_hook`: bias[0] ~
  weight[0], **weight[0] ~ input channels**, output channels ~ weight[0]. So input channels,
  depthwise filters and output channels are ONE group: in a t = 6 block the expansion conv's
  output, the depthwise conv and the projection conv's input; in the t = 1 block the stem output,
  its depthwise conv and its projection input. Every other module goes through the released
  hooks, so the ResNets are traced exactly as by the release (R50: the same 37 groups as
  before the extension).
- `continuous_dims` of a depthwise weight is `[0]`: its dim 1 (size 1) stays discrete, so it is a
  `GridSampleWeights1D` along the channels (discrete shape [1, 3, 3]) with **no quadrature**
  (`build_parameterization` adds it only along a continuous dim 1; a depthwise conv sums over no
  channel, so its filters are interpolated, not rescaled, when the group has fewer points).
- `groups` follows the channel count: in `cifar_mobilenetv2.py` a forward pre-hook sets
  `conv.groups` to the input's channel count at every call (as our `USConv2d` sets groups =
  in_channels at each width), so the resampled and the unparametrized networks stay depthwise.
- The 2-opt permutation then permutes the depthwise filters together with the expansion conv's
  output and the projection's input (one permutation per group, all its tensors), which keeps the
  function (the 6.4e-6 check above).

**Integral groups.** All without subgroups (no concatenation).
- R50, 37: 32 *inner* groups, two per bottleneck (conv1 out = conv2 in, conv2 out = conv3 in;
  64/128/256/512 by stage); the stem output 64 (stem conv out, `layer1.0` conv1 in and downsample
  in); 4 *residual streams* (they contain the residual `add`) 256 / 512 / 1024 / 2048 (every conv3
  out and the downsample out of the stage, every conv1 in of its later blocks, the next stage's
  first conv1 and downsample in; for 2048 also `fc` in).
- R18, 12: 8 inner (conv1 out = conv2 in of each basic block: 64, 64, 128, 128, 256, 256, 512,
  512); 4 residual streams 64 (stem out and `layer1`, whose first block has an identity shortcut,
  up to `layer2.0` conv1 in and downsample in), 128, 256, 512 (+ `fc` in).
- MBV2, 25: 16 *expanded* groups, one per t = 6 block (expansion out = depthwise = projection in:
  96, 144 x2, 192 x3, 384 x4, 576 x3, 960 x3); 5 residual streams 24, 32, 64, 96, 160 (the block
  outputs of a stage with its adds, and the next block's input); 32 (stem out = block 0's
  depthwise = its projection in); 16 (block 0 out = block 1 in); 320 (last block out = last 1x1
  in); 1280 (last 1x1 out = `fc` in).

**Widths.** The protocol width maps to every group's size by our rounding: a group of C channels
gets **n = make_divisible(C*w / r) * r** points (divisor 8), r = 1 except on MBV2's expanded
groups, r = t = 6 (n = 6 * make_divisible(C_in*w), our ratio rounding). Every conv of
`cifar_mobilenetv2.py` declares its `channel_ratio = (out, in)`; `train_inn.py` reads a group's r
from its tensors (they must agree). Three checks:
1. at start (`check_group_sizes`, before any training): at each of the 16 widths the size given
   to every group equals, for every tensor of the group, that dimension in the plain network
   built at that width (prints `Group sizes = the protocol's channel counts at all 16 widths`);
2. at the evaluation the resampled network (`get_unparametrized_model()`) must have, in every
   conv/linear weight (54 / 21 / 53), the shape of the plain network built at that width (e.g.
   R50 stem 16, pooled 512 at w = 0.25), else it raises;
3. once, on CPU (script not in the patch): the plain network built at each of the 16 widths has,
   layer by layer in forward order, the (out, in, groups, kernel, stride) of our in-repo model
   (built with the reference config's FLAGS, `width_mult` set) at that width: identical in all
   three settings at all 16 widths; parameters at 1.0 identical (23,705,252 / 11,271,432 /
   2,351,972). R50's rewritten `cifar_resnet50` has the previous patch's 320 state-dict keys and
   the same initial weights at a given seed.
With the MBV2 ratio forced to 1 (make_divisible(C*w) on expanded groups) checks 1 and 2 both
raise (`width 0.25: group of 96 gets 24 points, but layers.1.conv1.weight[0] has 48 channels`).

**Width sampling.** Their default draws each group's size independently (`RandomLinspace` with
one `UniformDistribution` per group; `imagenet.py` sets 338-512 on two groups). Here one width
w ~ U(0.25, 1.0) is drawn per step (Python `random`) and every group draws exactly its n at w
(a one-point `UniformDistribution(n, n)` through their `reset_distribution`), so each step trains
one uniform-width network. One forward/backward per step, plain cross-entropy (Algorithm 1; no
sandwich, no distillation).

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

Default here, all three settings: Adam 1e-4, weight decay 1e-8, batch 128, 50 epochs, x 0.33 at
epochs 10/20/30/40 (the code's schedule, deterministic). On Tiny ImageNet too (the paper's
ImageNet recipe, 100 epochs at batch 256, is for 1.28 M images; we keep one recipe across
settings). `--lr_schedule plateau` gives the paper's wording instead (`ReduceLROnPlateau`, factor
0.33, patience 10, on the epoch's mean **training** loss; the paper's loss is presumably the
validation loss, i.e. the test set here, which we do not use). No gradient clipping (none
upstream; `--grad_clip` exists), fp32.

**Evaluation.** 16 widths 0.25, 0.30, ..., 1.00, each resampled from the same INN, on the test
set (CIFAR-100 test, Tiny ImageNet official val, 10k each). BatchNorm does not survive the
conversion, so there is nothing to recalibrate: the run prints `(no BatchNorm layers: evaluated
without recalibration)` and tests directly (`--recal_batches` only acts if BN layers exist).

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
there). Loader workers: env `INN_WORKERS` (default 2). One GPU per run (`CUDA_VISIBLE_DEVICES`).
`train_inn.py` builds networks on the `meta` device for the channel checks (torch >= 2.0).

Data (`--data_dir`):
- CIFAR-100: the folder that *contains* `cifar-100-python/` (read-only Kaggle input works).
- Tiny ImageNet: as our `data/tinyimagenet.py` (loader code copied from the DS-Net patch's
  `protocol.py`), first hit wins: `--data_dir/tiny-imagenet-200`, any folder with the official
  layout under `/kaggle/input` (also a zip of it, unpacked into `--data_dir`), else the Stanford
  zip downloaded into `--data_dir`. Class ids and val labels md5-checked against the official
  release; the JPEGs are decoded once into `--data_dir/tinyimagenet_cache/{train_x,train_y,val_x,
  val_y}.npy` (1.35 GB, the same files our pipeline writes, read as is if present; memory-mapped
  per loader process). So `--data_dir` must be writable unless the cache is already there: on
  Kaggle use e.g. `/kaggle/working/data` with the dataset attached as an input. RandomCrop 64
  pad 8 + HFlip, mean/std (0.4802, 0.4481, 0.3975) / (0.2770, 0.2691, 0.2821), the official val
  split (10k) as the test set; the training loader keeps the last partial batch, as ours.

Windows: upstream `setup.py` refuses Windows, and setuptools' MinGW path is broken
(`customize_compiler` with no CC); the local test built the solver by hand with MinGW-w64 g++ 13.2
(ucrt): `cython -3 --cplus torch_integral/tsp_solver/solver.pyx`, then
`g++ -std=c++11 -O3 -shared -DMS_WIN64 -I<python include> -I<numpy include> solver.cpp
-L<python>/libs -lpython310 -static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive
-lwinpthread -Wl,--no-whole-archive -o torch_integral/tsp_solver/solver.cp310-win_amd64.pyd`.
Not needed on Kaggle.

Tested with Python 3.10.20, torch 2.11.0+cu128, torchvision 0.26.0, numpy 2.2.6, scipy 1.15.3,
Cython 3.3.0 (the only package installed for this, into `fimaq_apb`) on an RTX 5060 Ti 16 GB.
`torch.fx` tracing of all three networks by `IntegralTracer` works unchanged on torch 2.11. The
Linux build (`setup.py`, `-Ofast`) was not run here (no Linux machine); the patched `.pyx`
compiles with Cython 3.3.

For `scripts/build_external_notebooks.py` the `UPSTREAMS` entry would be
`('https://github.com/TheStageAI/TorchIntegral.git', '84c5e89047f1d47bbf6c62065aa223d997f9b5e7',
'TorchIntegral', 'inn.patch', 'cython')` and `WORKER_ENV` should carry `INN_WORKERS`. The two
phases are sequential on one card (chain them, e.g. `bash -c "python -u train_base.py ... &&
python -u train_inn.py ..."`), and give them **different labels or separate `run_pinned` calls**:
`VAL_LINE` ignores the tag, so under one label the INN's 1.00 line overwrites the base's.

## Commands (one T4 per run)
Labels: `inn_base_<setting>` (phase 1) and `inn_<setting>` (phase 2), setting = `r50`,
`r18_tiny`, `mbv2_slimlast`; each run with `SEED=1995`, `2026`, `2006` (same label, the seed tells
the runs apart; one `--ckpt_dir` per label and seed, the resume checkpoint is `{ckpt_dir}/last.pt`).
```bash
C100=/kaggle/input/cifar100              # contains cifar-100-python/
TINY=/kaggle/working/data                # tiny-imagenet-200/ here or under /kaggle/input; cache written here
CK=/kaggle/working/ckpt
SEED=1995                                # 1995, 2026, 2006
export CUDA_VISIBLE_DEVICES=0 INN_WORKERS=2

# r50: CIFAR-100 / CIFAR ResNet-50
python -u train_base.py --arch r50 --data_dir $C100 --seed $SEED \
  --ckpt_dir $CK/inn_base_r50_s$SEED --log_prefix inn_base_r50
python -u train_inn.py --arch r50 --data_dir $C100 --seed $SEED \
  --base_ckpt $CK/inn_base_r50_s$SEED/last.pt --ckpt_dir $CK/inn_r50_s$SEED --log_prefix inn_r50

# r18_tiny: Tiny ImageNet / ResNet-18 (--dataset tinyimagenet is the default of this arch)
python -u train_base.py --arch r18_tiny --dataset tinyimagenet --data_dir $TINY --seed $SEED \
  --ckpt_dir $CK/inn_base_r18_tiny_s$SEED --log_prefix inn_base_r18_tiny
python -u train_inn.py --arch r18_tiny --dataset tinyimagenet --data_dir $TINY --seed $SEED \
  --base_ckpt $CK/inn_base_r18_tiny_s$SEED/last.pt --ckpt_dir $CK/inn_r18_tiny_s$SEED \
  --log_prefix inn_r18_tiny

# mbv2_slimlast: CIFAR-100 / MobileNetV2 with the last 1x1 slimmed
python -u train_base.py --arch mbv2_slimlast --data_dir $C100 --seed $SEED \
  --ckpt_dir $CK/inn_base_mbv2_slimlast_s$SEED --log_prefix inn_base_mbv2_slimlast
python -u train_inn.py --arch mbv2_slimlast --data_dir $C100 --seed $SEED \
  --base_ckpt $CK/inn_base_mbv2_slimlast_s$SEED/last.pt --ckpt_dir $CK/inn_mbv2_slimlast_s$SEED \
  --log_prefix inn_mbv2_slimlast
```
Without `--arch` both scripts default to `r50` on `cifar100` (the earlier R50 commands still
work; a base checkpoint written before `--arch` existed is taken as R50). `train_inn.py` refuses a
base checkpoint of another arch. Its `--log_prefix` defaults to `inn_<arch>`.

Defaults of `train_base.py` = our recipe, identical in the three settings (= the recipe of
`apps/cifar100_es_a_r18_tiny.yml` and `apps/cifar100_ev_a_mbv2_slimlast.yml`): 100 epochs, batch
256, Nesterov SGD momentum 0.9, lr 0.2, per-step linear warm-up over 5 epochs then per-step cosine
to 0 (our `lr_schedule_per_iteration` formula, steps per epoch = training images / batch: 195.3 on
CIFAR-100, 390.6 on Tiny ImageNet), wd 5e-4 on conv (not depthwise) and linear weights only,
grad clip 5, 20-batch BN recalibration before the width-1.00 test (`--recal_batches 0`: trained
running stats). Defaults of `train_inn.py` = the recipe above. Extra flags: `--eval_converted`
(also test the 16 widths right after conversion, before any fine-tuning, tagged
`<log_prefix>_conv`, `-1/0`), `--tsp_iters` (2-opt restarts, default 100),
`--min_width/--max_width`, `--lr_schedule plateau`.

Output lines (tab separated, elapsed = seconds since the process started):
`[inn_r50] 15912.3s	val	0.25	-1/50: loss: 1.2345, top1_error: 0.4321, top5_error: 0.1234`
(`-1/50` = the fine-tuning epochs; the base prints `-1/100`). Before each line `train_inn.py`
prints the resampled network's parameter count and the channel check.

Evaluation only, from the resume checkpoint: the same command + `--eval_only` (raises if there is
no checkpoint). ~40 s for 16 widths on R50 here, ~15 s R18 Tiny, ~12 s MBV2.

Resume. After every epoch `{ckpt_dir}/last.pt` is written atomically (`.tmp` + `os.replace`):
model, optimizer state (SGD momentum / Adam moments), the plateau scheduler's state if used, next
epoch, arch and dataset, and Python/NumPy/torch/CUDA RNG states. The LR schedules of both phases
are pure functions of the epoch (and step), so nothing else is needed. If `last.pt` exists the
identical command resumes (`Resumed from ... at epoch N`). `train_inn.py` also writes `last.pt`
at epoch 0 right after the conversion; on resume it rebuilds the INN structure from the base
checkpoint **without** the permutation search (`init_from_discrete=False`, as `imagenet.py
--checkpoint` does; 0.5 s) and loads the trained, permuted parametrization. `--stop_at_epoch N`
exits cleanly after N epochs (no eval); copy `last.pt` (and for phase 2 the base `last.pt`) into
the new session and rerun. Granularity is one epoch. (Resume was smoke-tested on R50 only; the
path is the same code for all three.)

## Timing (RTX 5060 Ti, fp32, 2 workers, steady state)
| setting / phase | batch | s/step | steps/epoch | per run here | peak mem |
|---|---|---|---|---|---|
| R50 1 base | 256 | 0.290 (mean, steps 4-10 of both epochs) | 196 | ~57 s/epoch, ~1.6 h | 9.37-9.46 GiB |
| R50 conversion (CPU) | | | | 266 s (smoke base), 328 s (BX weights) | |
| R50 2 INN | 128 | 0.182 mean, 0.167-0.174 median | 391 | ~71 s/epoch, ~1.0 h | 2.03-2.04 GiB |
| R18 Tiny 1 base | 256 | 0.081 (3 steps) | 391 | ~32 s/epoch, ~0.9 h | 3.32 GiB |
| R18 Tiny 2 INN | 128 | 0.060 (3 steps) | 782 | ~47 s/epoch, ~0.7 h | 0.50 GiB |
| MBV2 1 base | 256 | 0.105 (3 steps) | 196 | ~21 s/epoch, ~0.6 h | 5.72 GiB |
| MBV2 2 INN | 128 | 0.057 mean, 0.049 median (3 steps) | 391 | ~22 s/epoch, ~0.3 h | 1.42 GiB |

The R18 Tiny and MBV2 rows rest on 3 steady steps of the smoke (steps 4-6), so treat them as
+-20 %; the 16-width evaluation and the conversion are not included. Conversion with 100 2-opt
restarts was not timed for them: their largest groups are 512 (R18) and 1280 (MBV2) against R50's
2048 / 1024, and one restart costs 0.08 s at n = 1024, 0.76 s at 2048, so expect well under a
minute (R18) and about a minute (MBV2).

Phase 1 uses `cudnn.benchmark = True` (fixed shapes; its peak includes cuDNN trial workspaces);
phase 2 sets it False (a new width almost every step; LCS measured ~20 s of autotuning per new
shape). Phase 2 is cheap per step: one pass at one width (mean width 0.625), no BN, but each step
resamples every parametrized tensor through `grid_sample`. Our US-Net R50 is ~0.59 s/step here
and 10.7 h on a T4 (factor ~3.3). By that factor on a T4: R50 ~5 h (phase 1) + ~3.5 h (phase 2)
+ 5-10 min conversion, ~9 h, tight for one 12 h session, so plan a `--stop_at_epoch` (e.g. phase
1 in one session, phase 2 in the next); R18 Tiny ~3 h + ~2.2 h and MBV2 ~2 h + ~1 h, both phases
chained should fit one session (keep `--stop_at_epoch` as a safeguard). On Kaggle's 2-CPU machines
the loaders (2 workers, PIL crop/flip/ToTensor) may limit the faster R18 Tiny and MBV2 phases.
Not measured on a T4.

The 2-opt solver is the release's algorithm with typed memory access (item 8): one restart at
n = 1024 took 3.5 s with the released code and 0.08 s patched (0.76 s at n = 2048); with the
released code 100 restarts on the 2048 / 1024 groups alone would take hours.

## Smoke tests actually run (local GPU)
**R50** (before the extension; the R50 path is unchanged by it, see Widths check 3).
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
trained weights (Method) is the meaningful one.

**R18 Tiny and MBV2** (one run per phase, no resume run). Phase 1 `--epochs 1
--max_iters_per_epoch 6`; phase 2 from that checkpoint `--epochs 1 --max_iters_per_epoch 6
--tsp_iters 1` (6 steps rather than 3 only so that the steady timer, which skips 3 steps, prints a
rate), each with the 16-width evaluation. Tiny ImageNet from our `data/` (its cache read as is).
```
base r18_tiny: ResNet-18 for 64x64 on tinyimagenet: 11,271,432 parameters
               Tiny ImageNet at .../data\tiny-imagenet-200: official classes and val labels
               Epoch 0 train time 8.7s, 6 steps, loss 5.3192, top1_error 0.9980, lr now 0.00061, steady 0.081 s/step mean, 0.083 median over 3 steps, peak mem 3.32 GiB
               (BN recalibrated per width: 20 layers, 20 batches of 256, cumulative)
[smoke_inn_base_r18_tiny] 23.2s	val	1.00	-1/1: loss: 5.3154, top1_error: 0.9951, top5_error: 0.9755
inn r18_tiny:  Conversion took 1.5s / 12 integral groups ... / BatchNorm layers left after conversion: 0
               Group sizes = the protocol's channel counts at all 16 widths
               Epoch 0 train time 4.2s, 6 steps, loss 5.3063, mean width 0.723, lr 0.0001, steady 0.060 s/step mean, 0.059 median over 3 steps, peak mem 0.50 GiB
               width 0.25: 724,776 parameters, channel counts = the protocol's in all 21 conv/linear layers
               (no BatchNorm layers: evaluated without recalibration)
[smoke_inn_r18_tiny] 11.1s	val	0.25	-1/1: loss: 1219.6114, top1_error: 0.9951, top5_error: 0.9740
... (16 lines) ...
[smoke_inn_r18_tiny] 25.7s	val	1.00	-1/1: loss: 5.3035, top1_error: 0.9963, top5_error: 0.9735
base mbv2:     CIFAR MobileNetV2 (slim_last) on cifar100: 2,351,972 parameters
               Epoch 0 train time 14.6s, 6 steps, loss 4.6141, top1_error 0.9915, lr now 0.00123, steady 0.105 s/step mean, 0.105 median over 3 steps, peak mem 5.72 GiB
               (BN recalibrated per width: 52 layers, 20 batches of 256, cumulative)
[smoke_inn_base_mbv2_slimlast] 36.4s	val	1.00	-1/1: loss: 4.6118, top1_error: 0.9921, top5_error: 0.9503
inn mbv2:      Conversion took 2.2s / 25 integral groups ... / BatchNorm layers left after conversion: 0
               Group sizes = the protocol's channel counts at all 16 widths
               Epoch 0 train time 5.9s, 6 steps, loss 4.6175, mean width 0.514, lr 0.0001, steady 0.057 s/step mean, 0.049 median over 3 steps, peak mem 1.42 GiB
               width 0.25: 187,124 parameters, channel counts = the protocol's in all 53 conv/linear layers
               (no BatchNorm layers: evaluated without recalibration)
[smoke_inn_mbv2_slimlast] 15.8s	val	0.25	-1/1: loss: 4.6261, top1_error: 0.9900, top5_error: 0.9500
... (16 lines) ...
[smoke_inn_mbv2_slimlast] 28.1s	val	1.00	-1/1: loss: 4.6073, top1_error: 0.9900, top5_error: 0.9500
```
Not run: a full epoch, a resume or `--eval_only` on R18 Tiny / MBV2, `--eval_converted` on them,
conversion with 100 restarts on them, the plateau schedule, the Linux build.

Conversion only (R50, BX's width-1.0 network, no fine-tuning; a US-Net-trained network, so only
indicative): 1.00 -> 77.95 %, but every narrower width is near chance, 0.95 -> 7.6 %, 0.80 -> 7.3 %,
0.50 -> 1.4 %, 0.25 -> 1.0 % (loss 640). Resampling a converted BN-free ResNet-50 is not
usable by itself; everything below 1.00 must come from phase 2.

## Changes and deviations
Code (all in `inn.patch`):
1. `cifar_resnet.py` (new): plain ResNets = our `us_resnet` at width w: `cifar_resnet50` (depth
   50, stem stride 1) and `resnet18_tiny` (depth 18, basic blocks, stem stride 2, 200 classes);
   projection where stride or full-width channels change; our initialisation; torchvision module
   names. `width` builds the network at width w (make_divisible(C*w), stem input and classes
   kept); at 1.0 the same modules in the same order as before the extension (R50 keys and init
   unchanged). `make_divisible` copied from ours.
2. `cifar_mobilenetv2.py` (new): plain CIFAR MobileNetV2 with slim_last = our
   `us_mobilenet_v2_cifar` (`slim_last: True`) at width w (Settings); module names `conv1/bn1`
   (stem), `layers.i.conv1..3/bn1..3` (expansion, depthwise, projection), `conv_last/bn_last`,
   `fc`; each conv declares `channel_ratio`; the depthwise conv's forward pre-hook sets `groups`
   to its input's channel count.
3. `train_base.py` (new): phase 1, our recipe (above), plain loop; `--arch`, `--dataset`; LR
   steps per epoch from the training set's size.
4. `train_inn.py` (new): phase 2 in a plain loop instead of catalyst's `SupervisedRunner`;
   conversion call as in the examples, through `ProtocolTracer` (depthwise convs, Method) and
   with depthwise weights continuous along dim 0 only; group sizes n = make_divisible(C*w/r)*r;
   the start-up group-size check and the resample-and-check evaluation against the plain network
   built at w (on the `meta` device); the group description, the width sampling, resume,
   `--eval_converted`, auto-build of the solver; `--arch`, `--dataset`, example input 32 or 64.
5. `protocol.py` (new): CIFAR-100 loaders (RandomCrop 32 pad 4 + HFlip, mean/std
   (0.5071,0.4865,0.4409)/(0.2673,0.2564,0.2762)) and Tiny ImageNet loaders (copied from the
   DS-Net patch: locate / md5 check / decode cache / memory-mapped dataset; RandomCrop 64 pad 8 +
   HFlip, Tiny mean/std); both: `INN_WORKERS`, persistent workers, training loader keeps the last
   partial batch, as ours, recalibration and test batches of 256. The `--arch` / `--dataset`
   tables, checkpoint save/load, steady step timer, BN recalibration (cumulative, 1/(i+1)),
   evaluation and the protocol lines (as LCS's/DS-Net's `protocol.py`).
6. `.gitignore`: the in-place build outputs (`solver.cpp`, `*.so`, `*.pyd`).
7. No change to `torch_integral` (tracer, grids, parametrizations, quadrature, wrapper) except
   item 8; the depthwise handling is a subclass passed through the wrapper's
   `integral_tracer_class` argument; no `torch.fx` fix was needed on torch 2.11.
8. `torch_integral/tsp_solver/solver.pyx`, speed only: the distance matrix (the release's
   Python double loop of `np.mean(np.abs(t[i] - t[j]))`) is computed with `torch.cdist(p=1) / D`
   on the tensor's device; `SymmMatrix` reads its packed array through a typed memoryview, and
   `_calculate_path_dist` takes a typed array. Same algorithm, same random restarts: checked
   against the released solver on random tensors (n = 128, 384, 4 tensors, 3 restarts, same seed):
   identical routes, path lengths within 2e-4 (of ~1,700), distance entries within 3e-6. Kept as
   released: 2-opt's gain test (`before` counts the edge (before_start, start) twice instead of
   adding (end, after_end)), the relative threshold, 100 restarts.

Method and recipe (ours vs paper / upstream):
9. Base network: trained by us with our recipe (100 epochs SGD) on our data. The paper converts
   pre-trained networks (pytorchcv on CIFAR-10, torchvision on ImageNet).
10. Whole network continuous (R50 37, R18 12, MBV2 25 groups, stem output, all residual streams
    and MBV2's last 1x1 to 1280 included); the examples make one stage continuous. Classifier
    classes and RGB input discrete.
11. One width per step for all groups, w ~ U(0.25, 1.0), sizes make_divisible(C*w/r)*r (r = t on
    MBV2's expanded channels, else 1); upstream samples each group independently in a per-group
    range (their examples: 66-100 % of two groups). The paper never resamples a whole network to
    25 %.
12. Evaluation at the 16 protocol widths through `get_unparametrized_model()` (an ordinary network
    with exactly our channel counts); upstream evaluates the resized INN directly (same weights).
13. Fine-tuning: Adam 1e-4, wd 1e-8, batch 128, 50 epochs, x 0.33 at 10/20/30/40 (code) rather
    than on a plateau (paper; `--lr_schedule plateau` on training loss). Plain CE, no clip, fp32.
    The same recipe on Tiny ImageNet (the paper's 100-epoch, batch-256 ImageNet recipe is not
    used).
14. Data: CIFAR-100 and Tiny ImageNet with our augmentation and normalisation (their CIFAR-10
    example: HFlip only, CIFAR-10 mean/std; no Tiny ImageNet example).
15. BN fused, so no BN at any width and no recalibration (our other rows recalibrate BN per width).
16. Depthwise convs (MBV2): our tracer tie of input channels, depthwise filters and output
    channels into one group, depthwise weights 1-D continuous along the channels without
    quadrature, `groups` following the channel count. Not from the authors (the release cannot
    convert a depthwise network: Method); it is what an integral operator with a diagonal kernel
    discretises to, and it keeps the converted network's function exactly.
17. `cudnn.benchmark` True in phase 1, False in phase 2 (upstream does not set it).
18. Seeds: `--seed` seeds Python/NumPy/torch (init, data order, the 2-opt restarts through
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
  chance on a well-trained R50 (BX test above), so 0.25-0.95 must be learned in 50 epochs of
  Adam at 1e-4 (then lower), BN-free, from that starting point. The paper itself notes INNs trained
  from scratch lag, "probably caused by absence of batch-normalization layers", and never trains a
  range as wide as 0.25-1.00 over the whole network. I expect the narrow end to be weak; the smokes
  (6-20 steps on 6-20-step bases) say nothing about it. A cheap first look on Kaggle: phase 2 with
  `--stop_at_epoch 5` and `--eval_only` on that checkpoint.
- **MBV2 without BN** may be harder still: ReLU6 clips at 6 after a fused BN, and the depthwise
  filters are resampled without any rescaling; the depthwise treatment is ours (item 16), checked
  for structure and function at 1.00, not for accuracy.
- Tiny ImageNet with the CIFAR-10 fine-tuning recipe (50 epochs, batch 128) is our choice.
- Schedule choice (code's milestones vs paper's plateau) and weight decay 1e-8 (code only; the
  paper names none).
- T4 timing is extrapolated, not measured; the R18 Tiny / MBV2 rates rest on 3 steps; the Linux
  build of the solver was not run here.
- 2-opt restarts are random, so the permutation (and so the INN) depends on the seed.
