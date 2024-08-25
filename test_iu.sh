#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python main.py --batch_size 16 --image_size 300 --vocab_size 760 --theta 0.4 --gamma 0.4 --beta 1.0 --delta 0.01 --dataset_name iu_xray --anno_path ../dataset/iu_xray/annotation.json --data_dir ../dataset/iu_xray/images --mode test --knowledge_prompt_path ./knowledge_path/knowledge_prompt_iu.pkl --test_path ./weight_path/iu_weight.pth