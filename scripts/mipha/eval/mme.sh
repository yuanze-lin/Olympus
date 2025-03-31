#!/bin/bash

model_name=Olympus
MODELDIR=ckpts
DATA_DIR=eval

python -m mipha.eval.model_vqa_loader \
    --model-path $MODELDIR/$model_name \
    --question-file $DATA_DIR/MME/llava_mme.jsonl \
    --image-folder $DATA_DIR/MME/MME_Benchmark_release_version \
    --answers-file $DATA_DIR/MME/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode v0

cd $DATA_DIR/MME

python convert_answer_to_mme.py --experiment $model_name

cd eval_tool

python calculation.py --results_dir answers/$model_name
