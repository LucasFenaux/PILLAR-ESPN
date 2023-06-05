#!/bin/bash


export CUDA_VISIBLE_DEVICES=$1
grid=$2
run_numbers=("1" "2" "3" "4" "5")

cd ..

if [[ "$grid" -eq 1 ]]; then
  for run_number in "${run_numbers[@]}"
  do
      python test_model.py --model-file "cifar100_resnet18_relu/run_${run_number}/best_model.pth"
  done
elif [[ "$grid" -eq 2 ]]; then
  for run_number in "${run_numbers[@]}"
  do
    python test_model.py --model-file "cifar100_resnet32_relu/run_${run_number}/best_model.pth"
  done
elif [[ "$grid" -eq 3 ]]; then
  for run_number in "${run_numbers[@]}"
  do
    python test_model.py --model-file "cifar100_vgg16bnavg_relu/run_${run_number}/best_model.pth"
  done
else
  echo "${grid} is not a proper grid value."
fi
