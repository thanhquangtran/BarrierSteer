#!/bin/bash

# Parameters
# MODEL_NAME="safe_qwen_1.5b"  # Model name parameter
MODEL_NAME="safe_llama2_7b"  # Model name parameter
NUM_SAMPLES=500               # Number of samples parameter

# Paths and methods
BASE_PATH="Your_Polytope_Model_Path"
METHODS=("AutoPrompt" "DirectRequest" "GCG" "UAT" "HumanJailbreaks" "GBDA" "PEZ" "AutoDAN")

# Function to check if all generate jobs are running
wait_for_jobs_to_run() {
    local all_running=0
    local check_interval=60  # Check every minute

    while [ $all_running -eq 0 ]; do
        # Get count of jobs with "generate_${ALGORITHM}_${MODEL}*" that are in Running state
        running_count=0
        total_count=0

        for method in "${METHODS[@]}"; do
            running_count=$((running_count + $(squeue -o "%.18i %.60j %.8u %.2t %.10M %.6D %R" -u $USER -h -t R | grep "generate_${method}_${MODEL_NAME}" | wc -l)))
            total_count=$((total_count + $(squeue -o "%.18i %.60j %.8u %.2t %.10M %.6D %R" -u $USER -h | grep "generate_${method}_${MODEL_NAME}" | wc -l)))
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

# Function to run method
run_method() {
    local edges=$1
    local folder_num=$2
    local exp_ident=$3
    local methods=$(IFS=,; echo "${METHODS[*]}")

    ./scripts/step_2_and_3.sh $MODEL_NAME sbatch $methods "${edges}_edges_${exp_ident}" $NUM_SAMPLES "${BASE_PATH}/${folder_num}/weights.pth"
}

# Loop through all 5 seeds
for SEED in {3..5}; do
    BASE_FOLDER=$((SEED - 1))
    EXP_IDENT="reject_seed_${SEED}"
    echo "Processing seed $SEED"

    # First handle the case of 1 edge
    EDGES=1
    FOLDER_NUM=$BASE_FOLDER
    echo $EDGES

    # Loop through each method for 1 edge
    run_method "$EDGES" "$FOLDER_NUM" "$EXP_IDENT"
    wait_for_jobs_to_run

    # Then handle the cases from 10 to 50
    for i in {0..4}; do
        EDGES=$((10 + i * 10))  # Will generate 10, 20, 30, 40, 50
        FOLDER_NUM=$((BASE_FOLDER + 5 * (i + 1)))

        echo $EDGES

        run_method "$EDGES" "$FOLDER_NUM" "$EXP_IDENT"
        wait_for_jobs_to_run
    done
done
