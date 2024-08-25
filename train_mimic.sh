#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python main.py --epochs 50 --lr_backbone 1e-5 --lr 1e-4 --batch_size 32 --image_size 300 --vocab_size 4253 --theta 0.4 --gamma 0.4 --beta 1.0 --delta 0.01 --dataset_name mimic_cxr --t_model_weight_path ./weight_path/mimic_t_model.pth --anno_path ../dataset/mimic_cxr/annotation.json --data_dir ../dataset/mimic_cxr/images300 --mode train
