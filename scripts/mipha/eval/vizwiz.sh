#!/bin/bash

model_name=Olympus
DATA_DIR=eval
MODELDIR=ckpts

python -m mipha.eval.model_vqa_loader \
    --model-path $MODELDIR/$model_name \
    --question-file $DATA_DIR/vizwiz/llava_test.jsonl \
    --image-folder $DATA_DIR/vizwiz/test \
    --answers-file $DATA_DIR/vizwiz/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode v0

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file $DATA_DIR/vizwiz/llava_test.jsonl \
    --result-file $DATA_DIR/vizwiz/answers/$model_name.jsonl \
    --result-upload-file $DATA_DIR/vizwiz/answers_upload/$model_name.json
