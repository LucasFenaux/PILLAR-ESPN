#!/bin/bash


# trains a series of poly_relu convnets with varying polynomial degrees

cd ..

python figure_4_train.py --model convnet --batch-size 256 --lr 0.1 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --no-coef-quant --epochs 150 --weight-decay 0.0001 --data-path ./data/ --output-dir testing_figure_2_train/ --use-poly --range 5.0 --degree 4 --dataset cifar10 --save-every-epoch --seed 1
