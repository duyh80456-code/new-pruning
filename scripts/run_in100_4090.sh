#!/usr/bin/env bash
# ImageNet-100, ResNet-50: SlimOT (KH) then US-Net (KI). Run from the repo
# root inside tmux. A smoke first; a rerun resumes from the last checkpoint.
set -e
for b in kh_k_heads4_r50_in100 ki_a_r50_in100; do
    rm -rf logs/smoke_$b
    python train.py app:apps/smoke_$b.yml > smoke_$b.log 2>&1 || { echo "smoke $b failed"; exit 1; }
done
for b in kh_k_heads4_r50_in100 ki_a_r50_in100; do
    python train.py app:apps/cifar100_$b.yml >> run_$b.log 2>&1
done
