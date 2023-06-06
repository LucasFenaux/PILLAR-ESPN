#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

python test_model.py --model-file "imagenet_resnet50/best_model.pth"
