#!/bin/bash

cd ..
export CUDA_VISIBLE_DEVICES=$1,$2,$3,$4
echo $CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=4
dataset_path="${HOME}"

torchrun --rdzv_backend=c10d --nproc_per_node=4 train.py --model resnet50 --batch-size 256 --lr 0.03 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 600 --random-erase 0.1 --weight-decay 0.00002 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 176 --model-ema --val-resize-size 232 --ra-sampler --ra-reps 4 --data-path "${dataset_path}" --use-poly --output-dir imagenet_resnet50/ --range 5.0 --reg_range 4.8 --reg_coef 0.0001 --clip --regularize --dataset imagenet --penalty_exp 10 --degree 4 --seed 1 --crypto-precision 10