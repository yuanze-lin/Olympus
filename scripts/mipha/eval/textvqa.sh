#!/bin/bash

model_name=Olympus
DATA_DIR=eval
MODELDIR=ckpts

python -m mipha.eval.model_vqa_loader \
    --model-path $MODELDIR/$model_name \
    --question-file $DATA_DIR/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $DATA_DIR/textvqa/train_images \
    --answers-file $DATA_DIR/textvqa/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode v0

python -m mipha.eval.eval_textvqa \
    --annotation-file $DATA_DIR/textvqa/TextVQA_0.5.1_val.json \
    --result-file $DATA_DIR/textvqa/answers/$model_name.jsonl
