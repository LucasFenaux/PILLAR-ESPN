#!/bin/bash

# trains a poly_relu convnet on cifar10 with regularization

cd ..


degree=4
reg_coefs=("0.1" "0.01" "0.001" "0.0001" "0.00001" "0.000001" "0.0000001" "0.00000001")
model_nums=("1" "2" "3" "4" "5")
for reg_coef in "${reg_coefs[@]}"
do
  for model_num in "${model_nums[@]}"
  do
  python train.py --model convnet --batch-size 256 --lr 0.1 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --epochs 150 --weight-decay 0.0001 --data-path ./data/ --output-dir  "./experiments/figure_3/convnet_pr_degree_${degree}_reg_coef_${reg_coef}_${model_num}" --dataset cifar10 --use-poly --clip --regularize --no-coef-quant --no-reg-warmup --reg_coef "${reg_coef}" --reg_range 5.0 --range 5.0 --degree ${degree} --no-overwrite --seed "${model_num}"
  done
done