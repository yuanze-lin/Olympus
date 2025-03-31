#!/bin/bash

model_name=Olympus
DATA_DIR=eval
MODELDIR=ckpts

python -m mipha.eval.model_vqa \
    --model-path $MODELDIR/$model_name \
    --question-file $DATA_DIR/mm-vet/llava-mm-vet.jsonl \
    --image-folder $DATA_DIR/mm-vet/images \
    --answers-file $DATA_DIR/mm-vet/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode phi

mkdir -p $DATA_DIR/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $DATA_DIR/mm-vet/answers/$model_name.jsonl \
    --dst $DATA_DIR/mm-vet/results/$model_name.json

