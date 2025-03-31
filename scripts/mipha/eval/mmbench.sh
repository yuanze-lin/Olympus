#!/bin/bash

SPLIT="mmbench_dev_20230712"
model_name=Olympus
DATA_DIR=eval
MODELDIR=ckpts

python -m mipha.eval.model_vqa_mmbench \
    --model-path $MODELDIR/$model_name \
    --image-folder $DATA_DIR/mmbench/images \
    --question-file $DATA_DIR/mmbench/$SPLIT.tsv \
    --answers-file $DATA_DIR/mmbench/answers/$SPLIT/$model_name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode v0

mkdir -p $DATA_DIR/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file $DATA_DIR/mmbench/$SPLIT.tsv \
    --result-dir $DATA_DIR/mmbench/answers/$SPLIT \
    --upload-dir $DATA_DIR/mmbench/answers_upload/$SPLIT \
    --experiment $model_name
