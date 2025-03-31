#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_vqav2_mscoco_test-dev2015"

model_name=Olympus
MODELDIR=ckpts
DATA_DIR=eval

CKPT=$model_name

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m mipha.eval.model_vqa_loader \
        --model-path $MODELDIR/$model_name \
        --question-file $DATA_DIR/vqav2/$SPLIT.jsonl \
        --image-folder $DATA_DIR/vqav2/test2015 \
        --answers-file $DATA_DIR/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode v0 &
done

wait

output_file=$DATA_DIR/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $DATA_DIR/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --dir $DATA_DIR/vqav2 --split $SPLIT --ckpt $CKPT

