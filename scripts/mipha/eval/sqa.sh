#!/bin/bash

model_name=Olympus
MODELDIR=ckpts/$model_name
DATADIR=eval

python -m mipha.eval.model_vqa_science \
    --model-path $MODELDIR \
    --question-file $DATADIR/scienceqa/llava_test_CQM-A.json \
    --image-folder $DATADIR/scienceqa/images/test \
    --answers-file $DATADIR/scienceqa/answers/$model_name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode v0

python mipha/eval/eval_science_qa.py \
    --base-dir $DATADIR/scienceqa \
    --result-file $DATADIR/scienceqa/answers/$model_name.jsonl \
    --output-file $DATADIR/scienceqa/answers/$model_name-output.jsonl \
    --output-result $DATADIR/scienceqa/answers/$model_name-result.json

