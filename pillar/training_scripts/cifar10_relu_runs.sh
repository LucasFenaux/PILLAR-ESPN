#!/bin/bash


export CUDA_VISIBLE_DEVICES=$1
grid=$2
dataset_path="${HOME}"

run_numbers=("1" "2" "3" "4" "5")

cd ..
if [[ "$grid" -eq 1 ]]; then
  for run_number in "${run_numbers[@]}"
  do
      python train.py --epochs 185 --lr 0.013 --output-dir "cifar10_resnet18_relu/run_${run_number}" --model resnet18 --batch-size 128 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --weight-decay 0.0005 --dataset cifar10 --data-path "${dataset_path}" --seed "${run_number}" --no-overwrite
  done
elif [[ "$grid" -eq 2 ]]; then
  for run_number in "${run_numbers[@]}"
  do
    python train.py --epochs 185 --lr 0.013 --output-dir "cifar10_resnet110_relu/run_${run_number}" --model resnet110 --batch-size 128 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --weight-decay 0.0005 --dataset cifar10 --data-path "${dataset_path}" --seed "${run_number}" --no-overwrite
  done
elif [[ "$grid" -eq 3 ]]; then
  for run_number in "${run_numbers[@]}"
  do
    python train.py --epochs 180 --lr 0.06 --output-dir "cifar10_minionnbn_relu/run_${run_number}" --model minionnbn --batch-size 128 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --weight-decay 0.0005 --dataset cifar10 --data-path "${dataset_path}" --seed "${run_number}" --no-overwrite
  done
elif [[ "$grid" -eq 4 ]]; then
  for run_number in "${run_numbers[@]}"
  do
    python train.py --epochs 300 --lr 0.0025 --output-dir "cifar10_vgg16bnavg_relu/run_${run_number}" --model vgg16bnavg --batch-size 128 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --weight-decay 0.0005 --dataset cifar10 --no-overwrite --data-path "${dataset_path}" --seed "${run_number}"
  done
else
  echo "${grid} is not a proper grid value."
fi