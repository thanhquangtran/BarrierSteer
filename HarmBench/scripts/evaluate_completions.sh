#!/bin/bash

# source command removed - not needed for local execution

cls_path="$1"
behaviors_path=$2
completions_path=$3
save_path=$4

echo "cls_path=$cls_path"
echo "behaviors_path=$behaviors_path"
echo "completions_path=$completions_path"
echo "save_path=$save_path"

# module load command removed - not needed for local execution

# cd command removed - script should be run from HarmBench directory
python -u evaluate_completions.py \
    --cls_path $cls_path \
    --behaviors_path $behaviors_path \
    --completions_path $completions_path \
    --save_path $save_path \
    --save_path $save_path \
    --include_advbench_metric \
    "${@:5}"
