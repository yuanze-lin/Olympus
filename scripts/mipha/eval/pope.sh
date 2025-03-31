#!/bin/bash

model_name=Olympus
DATA_DIR=eval
MODELDIR=ckpts
COCO_DIR=train_data/coco

python -m mipha.eval.model_vqa_loader \
    --model-path  $MODELDIR/$model_name \
    --question-file $DATA_DIR/pope/llava_pope_test.jsonl \
    --image-folder $COCO_DIR/val2014 \
    --answers-file $DATA_DIR/pope/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode v0

python mipha/eval/eval_pope.py \
    --annotation-dir $DATA_DIR/pope/coco \
    --question-file $DATA_DIR/pope/llava_pope_test.jsonl \
    --result-file $DATA_DIR/pope/answers/$model_name.jsonl
~                                                           
