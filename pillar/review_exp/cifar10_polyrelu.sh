#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
dataset_path="/scratch/lprfenau/datasets/"
run_numbers=("1" "2" "3" "4" "5")
#run_numbers=("1")


for run_number in "${run_numbers[@]}"
do
  python train.py --epochs 185 --lr 0.013 --reg_range 4.8 --reg_coef 0.00005 --output-dir "review_exp/cifar10_resnet18_polyrelu/run_${run_number}" --model resnet18 --batch-size 128 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --weight-decay 0.0005 --use-poly --clip --range 5.0 --regularize --dataset cifar10 --degree 4 --data-path "${dataset_path}" --seed "${run_number}" --no-overwrite --crypto-precision 10
done

for run_number in "${run_numbers[@]}"
do
    python test_model.py --model-file "review_exp/cifar10_resnet18_polyrelu/run_${run_number}/best_model.pth"
done