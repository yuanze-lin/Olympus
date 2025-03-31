#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

model_name=Olympus
SPLIT="llava_gqa_testdev_balanced"
DATADIR=eval
MODELDIR=ckpts

CKPT=$model_name

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m mipha.eval.model_vqa_loader \
        --model-path $MODELDIR/$model_name \
        --question-file $DATADIR/gqa/$SPLIT.jsonl \
        --image-folder $DATADIR/gqa/images \
        --answers-file $DATADIR/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode phi &
done

wait

output_file=$DATADIR/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $DATADIR/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $DATADIR/gqa/testdev_balanced_predictions.json

cd $DATADIR/gqa/
python eval.py --tier testdev_balanced
