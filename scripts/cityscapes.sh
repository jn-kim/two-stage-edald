#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 main_al.py \
  --dataset_name "cityscapes" \
  --pretrained_model "imagenet_256" \
  --keep_global "0.5" \
  --local_budget "50" \
  --data_usage "100" \
  --uncertainty "entropy_dald" \
  --final_budget_factor "0.1" \
  --n_stages "10"