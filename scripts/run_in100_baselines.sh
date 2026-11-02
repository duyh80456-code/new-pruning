#!/usr/bin/env bash
# ImageNet-100, ResNet-50: SlimOT with blur 0.1 (KM), AlphaNet (KJ), Scala
# (KL), then DS-Net from the authors' code (dsnet_ib_r50_in100), after SlimOT
# and US-Net. Run from the repo root inside tmux: it waits for
# run_in100_4090.sh to finish, so it can be started now. Smokes first; a
# rerun resumes from the last checkpoint.
set -eo pipefail
DATA=/home/s24gbn1/Documents/vinhpad/data/seed_1993_subset_100_imagenet/data
ROOT=$(pwd)
DSNET=$ROOT/../DS-Net

# DS-Net at the pinned commit with our patch, next to this repo; and timm
if [ ! -d "$DSNET" ]; then
    git clone https://github.com/changlin31/DS-Net.git "$DSNET"
    git -C "$DSNET" checkout 15cd3036970ec27d2c306014344fd50d9e9b888b
    git -C "$DSNET" apply "$ROOT/third_party/patches/dsnet.patch"
fi
# the timm the patch was tested with; it needs torch >= 1.7, so torch stays
python -c "import timm" 2>/dev/null || pip install --no-cache-dir timm==1.0.27
dsnet() {  # dsnet <log prefix> <ckpt dir> [extra flags]
    (cd "$DSNET" && DSNET_WORKERS=16 python train.py -c configs/in100_ds_resnet50_uniform.yml \
        --data_dir "$DATA" --seed 1995 --skip_upstream_test --output out_in100 \
        --log_prefix "$1" --ckpt_dir "$2" "${@:3}")
}

while pgrep -f run_in100_4090.sh > /dev/null; do sleep 600; done
for b in km_k_heads4_r50_in100_eps010 kj_alphanet_r50_in100 kl_scala_r50_in100; do
    rm -rf logs/smoke_$b
    python train.py app:apps/smoke_$b.yml 2>&1 | tee smoke_$b.log || { echo "smoke $b failed"; exit 1; }
done
rm -rf "$DSNET/ck_smoke_in100"
dsnet smoke_dsnet_in100 ck_smoke_in100 --epochs 2 --warmup-epochs 1 --max_iters_per_epoch 5 \
    --recal_batches 2 2>&1 | tee smoke_dsnet_in100.log || { echo "smoke dsnet failed"; exit 1; }

for b in km_k_heads4_r50_in100_eps010 kj_alphanet_r50_in100 kl_scala_r50_in100; do
    python train.py app:apps/cifar100_$b.yml 2>&1 | tee -a run_$b.log
done
dsnet dsnet_ib_r50_in100 ck_dsnet_in100 --recal_batches 40 2>&1 | tee -a run_dsnet_in100.log
