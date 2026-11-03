#!/usr/bin/env bash
# ImageNet-100, ResNet-50, every run in one queue, no smoke: SlimOT eps 0.2
# (KH), US-Net (KI), SlimOT eps 0.1 (KM), AlphaNet (KJ), Scala (KL), then
# DS-Net from the authors' code. Replaces run_in100_4090.sh followed by
# run_in100_baselines.sh. Run from the repo root inside tmux. Each run resumes
# from its last checkpoint, so after a crash run this again. A failed run is
# reported and the queue moves on to the next one.
set -o pipefail
DATA=/home/s24gbn1/Documents/vinhpad/data/seed_1993_subset_100_imagenet/data
ROOT=$(pwd)
DSNET=$ROOT/../DS-Net
FAILED=()

for b in kh_k_heads4_r50_in100 ki_a_r50_in100 km_k_heads4_r50_in100_eps010 \
         kj_alphanet_r50_in100 kl_scala_r50_in100; do
    echo "=== $b $(date)"
    python train.py app:apps/cifar100_$b.yml 2>&1 | tee -a run_$b.log \
        || { echo "=== $b FAILED"; FAILED+=("$b"); }
done

# DS-Net at the pinned commit with our patch, next to this repo; and timm
if [ ! -d "$DSNET" ]; then
    git clone https://github.com/changlin31/DS-Net.git "$DSNET"
    git -C "$DSNET" checkout 15cd3036970ec27d2c306014344fd50d9e9b888b
    git -C "$DSNET" apply "$ROOT/third_party/patches/dsnet.patch"
fi
# the timm the patch was tested with; it needs torch >= 1.7, so torch stays
python -c "import timm" 2>/dev/null || pip install --no-cache-dir timm==1.0.27
echo "=== dsnet $(date)"
(cd "$DSNET" && DSNET_WORKERS=16 python train.py -c configs/in100_ds_resnet50_uniform.yml \
    --data_dir "$DATA" --seed 1995 --skip_upstream_test --output out_in100 \
    --log_prefix dsnet_ib_r50_in100 --ckpt_dir ck_dsnet_in100 --recal_batches 40) \
    2>&1 | tee -a run_dsnet_in100.log || { echo "=== dsnet FAILED"; FAILED+=(dsnet); }

echo "=== done $(date), failed: ${FAILED[*]:-none}"
