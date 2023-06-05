#!/bin/bash


export CUDA_VISIBLE_DEVICES=$1
grid=$2
dataset_path="${HOME}"

run_numbers=("1" "2" "3" "4" "5")

cd ..
if [[ "$grid" -eq 1 ]]; then
  for run_number in "${run_numbers[@]}"
  do
    python train.py --epochs 185 --lr 0.013 --reg_range 4.8 --reg_coef 0.00005 --output-dir "cifar10_resnet18_polyrelu/run_${run_number}" --model resnet18 --batch-size 128 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --weight-decay 0.0005 --use-poly --clip --range 5.0 --regularize --dataset cifar10 --degree 4 --data-path "${dataset_path}" --seed "${run_number}" --no-overwrite --crypto-precision 10
  done
elif [[ "$grid" -eq 2 ]]; then
  for run_number in "${run_numbers[@]}"
  do
    python train.py --epochs 185 --lr 0.013 --reg_range 4.8 --reg_coef 0.00005 --output-dir "cifar10_resnet110_polyrelu/run_${run_number}" --model resnet110 --batch-size 128 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --weight-decay 0.0005 --use-poly --clip --range 5.0 --regularize --dataset cifar10 --degree 4 --data-path "${dataset_path}" --seed "${run_number}" --no-overwrite --crypto-precision 10
  done
elif [[ "$grid" -eq 3 ]]; then
  for run_number in "${run_numbers[@]}"
  do
    python train.py --epochs 180 --lr 0.06 --reg_range 4.4 --reg_coef 0.00005 --output-dir "cifar10_minionnbn_polyrelu/run_${run_number}" --model minionnbn --batch-size 128 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --weight-decay 0.0005 --use-poly --clip --range 5.0 --regularize --dataset cifar10 --degree 6 --no-overwrite --data-path "${dataset_path}" --crypto-precision 10 --seed "${run_number}"
  done
elif [[ "$grid" -eq 4 ]]; then
  for run_number in "${run_numbers[@]}"
  do
    python train.py --epochs 300 --lr 0.0025 --reg_range 4.8 --reg_coef 0.000001 --output-dir "cifar10_vgg16bnavg_polyrelu/run_${run_number}" --model vgg16bnavg --batch-size 128 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --weight-decay 0.0005 --use-poly --clip --range 5.0 --regularize --dataset cifar10 --degree 4 --no-overwrite --data-path "${dataset_path}" --crypto-precision 10 --seed "${run_number}"
  done
else
  echo "${grid} is not a proper grid value."
fi