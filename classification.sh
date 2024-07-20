#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
lr=2e-5
seed=39  # 将种子值设置为固定的值
output_dir="output"

mkdir -p "$output_dir"
python classification.py  \
        --kmer -1 \
        --run_name "Classification_bacillus_${lr}_seed${seed}" \
        --model_max_length 40 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate "${lr}" \
        --num_train_epochs 10 \
        --seed "${seed}" \
        --fp16 \
        --weight_decay 0.01 \
        --save_steps 1000 \
        --output_dir  "$output_dir" \
        --lr_scheduler_type cosine \
        --warmup_steps 400 \
        --logging_steps 50 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False \
        --eval_steps 200 \
        --evaluation_strategy steps



