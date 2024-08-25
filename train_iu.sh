#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
python main.py --epochs 50 --lr_backbone 1e-5 --lr 1e-4 --batch_size 8 --image_size 300 --vocab_size 760 --theta 0.4 --gamma 0.4 --beta 1.0 --delta 0.01 --dataset_name iu_xray --t_model_weight_path ./weight_path/iu_t_model.pth --anno_path ../dataset/iu_xray/annotation.json --data_dir ../dataset/iu_xray/images --mode train
