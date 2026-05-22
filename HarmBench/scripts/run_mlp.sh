#!/bin/bash

# Set variables
MODEL="safe_qwen_1.5b"
MODE="sbatch"
ALGORITHMS="AutoPrompt,DirectRequest,GCG,UAT,HumanJailbreaks,GBDA,PEZ,AutoDAN"
# ALGORITHMS="GCG,GBDA,PEZ"
EXP_IDENT="MLP+5"
PART_SIZE="500"
USE_DEFENSE=""
DEFENSE_METHOD=""

# Base directory containing the weight files
# BASE_DIR="/cluster/project/krause/chexin/crlhf/multirun/harmbench/2025-04-04/mlp_qwen_1epoch-15-02-25"
# BASE_DIR="/cluster/project/krause/chexin/crlhf/multirun/harmbench/2025-04-04/mlp_mistral_1epoch-15-02-39"
# BASE_DIR="/cluster/project/krause/chexin/crlhf/multirun/harmbench/2025-04-04/mlp_llama_1epoch-14-29-31"

BASE_DIR="/cluster/project/krause/chexin/crlhf/multirun/harmbench/2025-04-04/5_layer_mlp_qwen_1epoch-21-04-22"
# BASE_DIR="/cluster/project/krause/chexin/crlhf/multirun/harmbench/2025-04-04/5_layer_mlp_mistral_1epoch-20-56-15"
# BASE_DIR="/cluster/project/krause/chexin/crlhf/multirun/harmbench/2025-04-04/5_layer_mlp_llama_1epoch-21-01-02"

# Print current configuration
echo "Current configuration:"
echo "Model: $MODEL"
echo "Experiment ID: $EXP_IDENT"
echo "Base directory: $BASE_DIR"
echo "----------------------------------------"
sleep $((4 * 60 * 60))

wait_for_jobs_to_run() {
    local all_running=0
    local check_interval=60  # Check every minute

    # Split ALGORITHMS into an array
    IFS=',' read -ra ALGORITHM_ARRAY <<< "$ALGORITHMS"

    while [ $all_running -eq 0 ]; do
        # Get count of jobs with "generate_${ALGORITHM}_${MODEL}*" that are in Running state
        running_count=0
        total_count=0

        for algorithm in "${ALGORITHM_ARRAY[@]}"; do
            running_count=$((running_count + $(squeue -o "%.18i %.60j %.8u %.2t %.10M %.6D %R" -u $USER -h -t R | grep "generate_${algorithm}_${MODEL}" | wc -l)))
            total_count=$((total_count + $(squeue -o "%.18i %.60j %.8u %.2t %.10M %.6D %R" -u $USER -h | grep "generate_${algorithm}_${MODEL}" | wc -l)))
        done

        echo "Current status: $running_count running out of $total_count total generate jobs"

        if [ $running_count -eq $total_count ] && [ $total_count -gt 0 ]; then
            all_running=1
            echo "All generate jobs are now running!"
        else
            echo "Waiting for all jobs to start running... Will check again in 1 minute"
            sleep $check_interval
        fi
    done
}

# Loop through directories 0-4
for i in {0..4}; do
    # Construct the full path to the weights file
    POLYTOPE_PATH="${BASE_DIR}/${i}/weights.pth"

    # Check if the weights file exists
    if [ -f "$POLYTOPE_PATH" ]; then
        echo "Running with polytope weights from: $POLYTOPE_PATH"

        # Construct a unique experiment identifier for each run
        CURRENT_EXP_IDENT="${EXP_IDENT}_seed_${i}"

        # Call the step_2_and_3.sh script
        bash scripts/step_2_and_3.sh \
            "$MODEL" \
            "$MODE" \
            "$ALGORITHMS" \
            "$CURRENT_EXP_IDENT" \
            "$PART_SIZE" \
            "$POLYTOPE_PATH" \
            "$USE_DEFENSE" \
            "$DEFENSE_METHOD"

        echo "Started running step_2_and_3.sh with model: $MODEL, algorithms: $ALGORITHMS, seed: $i"

        # Wait for all jobs from this seed to starr running before moving to the next seed
        wait_for_jobs_to_run

        echo "Finished submitting step_2_and_3.sh with model: $MODEL, algorithms: $ALGORITHMS, seed: $i"
    else
        echo "Warning: Weights file not found at $POLYTOPE_PATH"
    fi
done

echo "All runs completed"
