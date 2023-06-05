#!/bin/bash


export CUDA_VISIBLE_DEVICES=$1
grid=$2
dataset_path="${HOME}"
run_numbers=("1" "2" "3" "4" "5")

cd ..

if [[ "$grid" -eq 1 ]]; then
  for run_number in "${run_numbers[@]}"
  do
    python train.py --epochs 185 --lr 0.013 --output-dir "cifar100_resnet18_relu/run_${run_number}" --model resnet18 --batch-size 128 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --weight-decay 0.0005 --dataset cifar100 --data-path "${dataset_path}" --seed "${run_number}" --no-overwrite
  done
elif [[ "$grid" -eq 2 ]]; then
  for run_number in "${run_numbers[@]}"
  do
    python train.py --model resnet32 --dataset cifar100 --output-dir "cifar100_resnet32_relu/run_${run_number}" --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --wd 0.0001 --batch-size 128 --lr 0.1 --epochs 150 --data-path "${dataset_path}" --seed "${run_number}" --no-overwrite
  done
elif [[ "$grid" -eq 3 ]]; then
  for run_number in "${run_numbers[@]}"
  do
      python train.py --epochs 300 --lr 0.0025 --output-dir "cifar100_vgg16bnavg_relu/run_${run_number}" --model vgg16bnavg --batch-size 128 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --weight-decay 0.0005 --dataset cifar100  --no-overwrite --data-path "${dataset_path}" --seed "${run_number}"
  done
else
  echo "${grid} is not a proper grid value."
fi
