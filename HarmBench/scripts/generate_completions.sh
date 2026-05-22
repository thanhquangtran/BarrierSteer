#!/bin/bash

model_name=$1
behaviors_path=$2
test_cases_path=$3
save_path=$4
max_new_tokens=$5
incremental_update=$6
save_hidden_states_path=${7:-None}  # Default to None if not provided
part_size=${8:-None}  # Default to None if not provided
part_number=${9:-1}   # Default to 1 if not provided
use_defense=${10:-False}  # Default to False if not provided
defense_method=${11:-None}  # Default to None if not provided
starting_batch_size=${12:-16} # Default to 16 if not provided

echo "model_name=$model_name"
echo "behaviors_path=$behaviors_path"
echo "test_cases_path=$test_cases_path"
echo "save_path=$save_path"
echo "max_new_tokens=$max_new_tokens"
echo "incremental_update=$incremental_update"
echo "save_hidden_states_path=$save_hidden_states_path"
echo "part_size=$part_size"
echo "part_number=$part_number"
echo "use_defense=$use_defense"
echo "defense_method=$defense_method"
echo "starting_batch_size=$starting_batch_size"

# Create directories for save_hidden_states_path only if it's not None
if [ "$save_hidden_states_path" != "None" ]; then
    mkdir -p "$(dirname "$save_hidden_states_path")"
fi

# Construct the base command
# cd command removed - script should be run from HarmBench directory
cmd="python -u generate_completions.py \
    --model_name $model_name \
    --behaviors_path $behaviors_path \
    --test_cases_path $test_cases_path \
    --save_path $save_path \
    --max_new_tokens $max_new_tokens \
    --starting_batch_size $starting_batch_size"

# module load command removed - not needed for local execution

# Add save_hidden_states_path to the command only if it's not None
if [ "$save_hidden_states_path" != "None" ]; then
    cmd="$cmd --save_hidden_states_path $save_hidden_states_path"
fi

# Add part_size and part_number to the command only if part_size is not None
if [ "$part_size" != "None" ]; then
    cmd="$cmd --part_size $part_size --part_number $part_number"
fi

# Check if incremental_update is provided and equals "True"
if [[ ! -z "$incremental_update" && "$incremental_update" == "True" ]]; then
    cmd="$cmd --incremental_update"
fi

# Add defense configuration if use_defense is True
if [[ "$use_defense" == "True" ]]; then
    cmd="$cmd --use_defense --defense_method $defense_method"
fi

# Execute the command
eval $cmd
