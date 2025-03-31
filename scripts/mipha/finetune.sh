#!/bin/bash
model_dir=ckpts/Mipha-3B
outputdir=ckpts/Olympus_2e_lr5e-5

deepspeed --num_nodes=8 --num_gpus=8 --master_addr="$MASTER_ADDR" --master_port=29600 mipha/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $model_dir \
    --version v0 \
    --data_path train_data/Olympus.json \
    --image_folder train_data \
    --tune_mm_mlp_adapter True \
    --freeze_vision_tower False \
    --freeze_backbone False \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --fp16 True \
    --output_dir $outputdir \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 32 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name 'Olympus'

