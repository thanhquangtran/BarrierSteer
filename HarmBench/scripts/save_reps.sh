#!/bin/bash

MODEL=$1
MODE=$2
ALGORITHMS=$3
LAYER_NUM="${4:-"-1"}"
PART_SIZE="${5:-"500"}"  # Default to 500 test cases per part
STARTING_BATCH_SIZE="${6:-"8"}" # Default to 8
PARTITION="cais"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

MEM="128G"
GPUMEM="32g"
TIME="4:00:00"

# GPU Detection for local_parallel
if [ "$MODE" == "local_parallel" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
        IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
        GPU_COUNT=${#GPU_IDS[@]}
    else
        GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        if [ -z "${GPU_COUNT}" ] || [ "${GPU_COUNT}" -lt 1 ]; then
             GPU_COUNT=1
             GPU_IDS=(0)
        else
            for ((g=0; g<GPU_COUNT; g++)); do
                GPU_IDS+=("${g}")
            done
        fi
    fi
    echo "Detected ${GPU_COUNT} GPUs: ${GPU_IDS[*]}"
fi

# Create logs directory and command log file
LOGS_DIR="./slurm_logs/command_logs"
mkdir -p ${LOGS_DIR}
COMMAND_LOG="${LOGS_DIR}/commands_${MODEL}_${TIMESTAMP}.log"

# Log script invocation
echo "=== Command Log Created at $(date) ===" > ${COMMAND_LOG}
echo "Model: ${MODEL}" >> ${COMMAND_LOG}
echo "Mode: ${MODE}" >> ${COMMAND_LOG}
echo "Algorithms: ${ALGORITHMS}" >> ${COMMAND_LOG}
echo "Layer Num: ${LAYER_NUM}" >> ${COMMAND_LOG}
echo "Starting Batch Size: ${STARTING_BATCH_SIZE}" >> ${COMMAND_LOG}
echo "===================================" >> ${COMMAND_LOG}

# Split the algorithms string into an array
IFS=',' read -ra ALGORITHM_ARRAY <<< "$ALGORITHMS"

# Function to get total number of test cases from a JSON file
get_test_case_count() {
    local file=$1
    python3 -c "
import json
with open('$1', 'r') as f:
    data = json.load(f)
total = sum(len(cases) for cases in data.values())
print(total)
"
}

# Loop through each algorithm
for ALGORITHM in "${ALGORITHM_ARRAY[@]}"; do
    # Trim whitespace from algorithm name
    ALGORITHM=$(echo "$ALGORITHM" | xargs)
    
    echo "Processing algorithm: ${ALGORITHM}"
    echo "=== Processing algorithm: ${ALGORITHM} at $(date) ===" >> ${COMMAND_LOG}
    
    # Set OUTPUT_DIR and RESULTS_DIR based on algorithm
    # OUTPUT_DIR and RESULTS_DIR are always model-specific
    RESULTS_BASE_DIR="${HARM_BENCH_RESULTS_DIR:-"./results"}"
    OUTPUT_DIR="./slurm_logs/generate_completions/${ALGORITHM}/${MODEL}"
    
    # Determine RESULTS_DIR based on algorithm
    if [[ "${ALGORITHM}" == "DirectRequest" ]]; then
        RESULTS_DIR="${RESULTS_BASE_DIR}/${ALGORITHM}/default"
        TEST_CASE_DIR="${RESULTS_DIR}"
    else
        RESULTS_DIR="${RESULTS_BASE_DIR}/${ALGORITHM}/${MODEL}"
        
        # TEST_CASE_DIR may be shared for some algorithms
        if [[ "${ALGORITHM}" == "HumanJailbreaks" ]]; then
            TEST_CASE_DIR="${RESULTS_BASE_DIR}/${ALGORITHM}/random_subset_5"
        elif [[ "${ALGORITHM}" == "PAP-top5" ]]; then
            TEST_CASE_DIR="${RESULTS_BASE_DIR}/PAP/top_5"
        else
            TEST_CASE_DIR="${RESULTS_DIR}"
        fi
    fi

    # Create necessary directories
    mkdir -p ${OUTPUT_DIR}
    mkdir -p ${RESULTS_DIR}/test_cases
    mkdir -p ${RESULTS_DIR}/completions
    mkdir -p ${RESULTS_DIR}/hidden_states

    # Get total number of test cases. HARM_BENCH_TEST_CASES_PATH allows
    # pipelines with train/val splits (for example WildGuard) to run Stage 2
    # on a train-only testcase JSON while keeping the default test_cases.json
    # for evaluation.
    TEST_CASES_FILE="${HARM_BENCH_TEST_CASES_PATH:-${TEST_CASE_DIR}/test_cases/test_cases.json}"
    TOTAL_CASES=$(get_test_case_count "$TEST_CASES_FILE")
    
    if [ "$PART_SIZE" == "None" ] || [ -z "$PART_SIZE" ]; then
        if [ "$MODE" == "local_parallel" ] && [ "$GPU_COUNT" -gt 1 ]; then
            # Auto-calculate part size to split evenly across GPUs
            PART_SIZE=$(( ($TOTAL_CASES + $GPU_COUNT - 1) / $GPU_COUNT ))
            NUM_PARTS=$(( ($TOTAL_CASES + $PART_SIZE - 1) / $PART_SIZE ))
            echo "Auto-calculated part size: $PART_SIZE for $GPU_COUNT GPUs"
        else
            NUM_PARTS=1
            PART_SIZE="None"
        fi
    else
        NUM_PARTS=$(( ($TOTAL_CASES + $PART_SIZE - 1) / $PART_SIZE ))
    fi
    
    echo "Total test cases: $TOTAL_CASES"
    echo "Number of parts: $NUM_PARTS"
    echo "Total test cases: $TOTAL_CASES" >> ${COMMAND_LOG}
    echo "Number of parts: $NUM_PARTS" >> ${COMMAND_LOG}

    # Track PIDs and slot state for local_parallel
    PIDS=()
    PARTS=()
    PARALLEL_FAIL=0
    FAILED_PARTS=()
    if [ "$MODE" == "local_parallel" ]; then
        GPU_ACTIVE_PIDS=()
        GPU_ACTIVE_PARTS=()
    fi

    # Loop through parts
    for ((PART=1; PART<=$NUM_PARTS; PART++)); do
        JOB_NAME="generate_save_${ALGORITHM}_part${PART}"

        # Logic to determine JOB_IDX for parallel scheduling (0-based)
        JOB_IDX=$((PART - 1))

        # Construct generation command with part information and timestamp
        BEHAVIORS_PATH="${HARM_BENCH_BEHAVIORS_PATH:-"./data/behavior_datasets/harmbench_behaviors_text_all.csv"}"
        
        GENERATE_COMMAND="scripts/generate_completions_save.sh ${MODEL} \
            ${BEHAVIORS_PATH} \
            ${TEST_CASES_FILE} \
            ${RESULTS_DIR}/completions/${MODEL}_${TIMESTAMP}.json \
            512 False \
            ${RESULTS_DIR}/hidden_states/hidden_states_${MODEL}_layer${LAYER_NUM}_${TIMESTAMP}.pth \
            ${PART_SIZE} ${PART} ${LAYER_NUM} ${STARTING_BATCH_SIZE}"

        echo "Processing part: ${PART}/${NUM_PARTS}"
        echo -e "\n=== Processing part ${PART}/${NUM_PARTS} at $(date) ===" >> ${COMMAND_LOG}
        echo "Generate command:" >> ${COMMAND_LOG}
        echo "${GENERATE_COMMAND}" >> ${COMMAND_LOG}

        if [ "$MODE" == "sbatch" ]; then
            # Construct the full sbatch command
            FULL_SBATCH_COMMAND="sbatch --partition=${PARTITION} \
                --job-name=${JOB_NAME} \
                --nodes=1 \
                --gpus-per-node=1 \
                --time=${TIME} \
                --output=${OUTPUT_DIR}/${MODEL}_${TIMESTAMP}_part${PART}.log \
                --mem-per-cpu=${MEM} \
                --gres=gpumem:${GPUMEM} \
                --parsable \
                ${GENERATE_COMMAND}"

            echo "Full sbatch command:" >> ${COMMAND_LOG}
            echo "${FULL_SBATCH_COMMAND}" >> ${COMMAND_LOG}
            
            GENERATE_JOB_ID=$(eval "${FULL_SBATCH_COMMAND}")
            echo "Generation Job ID: ${GENERATE_JOB_ID}" >> ${COMMAND_LOG}

            # Only run merging and evaluation after the last part is complete
            if [ "$PART" == "$NUM_PARTS" ]; then
                # Create a script to merge parts and run evaluation
                EVALUATE_OUTPUT_DIR="./slurm_logs/evaluate_completions/${ALGORITHM}/${MODEL}"
                mkdir -p ${EVALUATE_OUTPUT_DIR}
                
                MERGE_AND_EVALUATE_SCRIPT="${OUTPUT_DIR}/merge_and_evaluate_${TIMESTAMP}.sh"
                echo "#!/bin/bash" > ${MERGE_AND_EVALUATE_SCRIPT}
                if [ "$NUM_PARTS" == "1" ]; then
                    echo "# Run evaluation directly (no merge)" >> ${MERGE_AND_EVALUATE_SCRIPT}
                    echo "scripts/evaluate_completions.sh cais/HarmBench-Llama-2-13b-cls \
                        ${BEHAVIORS_PATH} \
                        ${RESULTS_DIR}/completions/${MODEL}_${TIMESTAMP}.json \
                        ${RESULTS_DIR}/results/${MODEL}_${TIMESTAMP}.json" >> ${MERGE_AND_EVALUATE_SCRIPT}
                else
                    echo "# Merge completion parts" >> ${MERGE_AND_EVALUATE_SCRIPT}
                    echo "python merge_completion_parts.py \
                        --input_pattern '${RESULTS_DIR}/completions/${MODEL}_${TIMESTAMP}_part*.json' \
                        --output_file '${RESULTS_DIR}/completions/${MODEL}_${TIMESTAMP}_merged.json'" >> ${MERGE_AND_EVALUATE_SCRIPT}
                    echo "# Merge hidden states parts" >> ${MERGE_AND_EVALUATE_SCRIPT}
                    echo "python merge_completion_parts.py \
                        --input_pattern '${RESULTS_DIR}/hidden_states/hidden_states_${MODEL}_layer${LAYER_NUM}_${TIMESTAMP}_part*.pth' \
                        --output_file '${RESULTS_DIR}/hidden_states/hidden_states_${MODEL}_layer${LAYER_NUM}_${TIMESTAMP}.pth'" >> ${MERGE_AND_EVALUATE_SCRIPT}
                    echo "# Run evaluation" >> ${MERGE_AND_EVALUATE_SCRIPT}
                    echo "scripts/evaluate_completions.sh cais/HarmBench-Llama-2-13b-cls \
                        ${BEHAVIORS_PATH} \
                        ${RESULTS_DIR}/completions/${MODEL}_${TIMESTAMP}_merged.json \
                        ${RESULTS_DIR}/results/${MODEL}_${TIMESTAMP}.json" >> ${MERGE_AND_EVALUATE_SCRIPT}
                fi
                chmod +x ${MERGE_AND_EVALUATE_SCRIPT}

                # Log the merge and evaluate script
                echo -e "\n=== Created merge and evaluate script at $(date) ===" >> ${COMMAND_LOG}
                echo "Script path: ${MERGE_AND_EVALUATE_SCRIPT}" >> ${COMMAND_LOG}
                cat ${MERGE_AND_EVALUATE_SCRIPT} >> ${COMMAND_LOG}

                # Submit the merge and evaluate job
                echo -e "\n=== Submitting merge and evaluate job at $(date) ===" >> ${COMMAND_LOG}
                
                # Construct the full evaluation sbatch command
                FULL_EVAL_COMMAND="sbatch --partition=${PARTITION} \
                    --job-name=evaluate_${ALGORITHM} \
                    --nodes=1 \
                    --gpus-per-node=1 \
                    --time=4:00:00 \
                    --output=${EVALUATE_OUTPUT_DIR}/${MODEL}_${TIMESTAMP}.log \
                    --mem-per-cpu=${MEM} \
                    --gres=gpumem:40g \
                    --dependency=afterok:${GENERATE_JOB_ID} \
                    --parsable \
                    ${MERGE_AND_EVALUATE_SCRIPT}"
                
                echo "Full evaluation sbatch command:" >> ${COMMAND_LOG}
                echo "${FULL_EVAL_COMMAND}" >> ${COMMAND_LOG}
                
                EVAL_JOB_ID=$(eval "${FULL_EVAL_COMMAND}")
                
                echo "Merge and Evaluate Job ID: ${EVAL_JOB_ID}" >> ${COMMAND_LOG}
            fi

        elif [ "$MODE" == "local_parallel" ]; then
            # Assign GPU Round Robin
            GPU_PTR=$(( JOB_IDX % GPU_COUNT ))
            GPU_ID="${GPU_IDS[${GPU_PTR}]}"

            # Enforce one process per GPU slot:
            # before scheduling a new part on this GPU, wait for the previous
            # part assigned to the same slot to finish.
            PREV_PID="${GPU_ACTIVE_PIDS[${GPU_PTR}]}"
            PREV_PART="${GPU_ACTIVE_PARTS[${GPU_PTR}]}"
            if [ -n "${PREV_PID}" ] && kill -0 "${PREV_PID}" 2>/dev/null; then
                echo "GPU ${GPU_ID} busy with part ${PREV_PART} (PID ${PREV_PID}), waiting..."
                wait "${PREV_PID}" || {
                    PARALLEL_FAIL=1
                    FAILED_PARTS+=("${PREV_PART}")
                }
            fi
            
            LOG_FILE="${OUTPUT_DIR}/${MODEL}_${TIMESTAMP}_part${PART}.log"
            echo "Running part ${PART} on GPU ${GPU_ID} > ${LOG_FILE}"

            # Run in background
            CUDA_VISIBLE_DEVICES=${GPU_ID} bash ${GENERATE_COMMAND} > "${LOG_FILE}" 2>&1 &
            PID=$!
            PIDS+=($PID)
            PARTS+=($PART)
            GPU_ACTIVE_PIDS[${GPU_PTR}]="${PID}"
            GPU_ACTIVE_PARTS[${GPU_PTR}]="${PART}"
            echo "Started job with PID ${PID}"

        elif [ "$MODE" == "local" ]; then
            bash ${GENERATE_COMMAND}
            
            # Check exit code
            if [ $? -ne 0 ]; then
                echo "Error: Generation failed for part ${PART}"
                exit 1
            fi
            
            # If this is the last part
            if [ "$PART" == "$NUM_PARTS" ]; then
                if [ "$NUM_PARTS" == "1" ]; then
                     # No merge needed, just run evaluation
                    bash scripts/evaluate_completions.sh cais/HarmBench-Llama-2-13b-cls \
                        ${BEHAVIORS_PATH} \
                        ${RESULTS_DIR}/completions/${MODEL}_${TIMESTAMP}.json \
                        ${RESULTS_DIR}/results/${MODEL}_${TIMESTAMP}.json
                else
                    # Merge completion parts
                    python merge_completion_parts.py \
                        --input_pattern "${RESULTS_DIR}/completions/${MODEL}_${TIMESTAMP}_part*.json" \
                        --output_file "${RESULTS_DIR}/completions/${MODEL}_${TIMESTAMP}_merged.json"
                    
                    # Merge hidden states parts
                    python merge_completion_parts.py \
                        --input_pattern "${RESULTS_DIR}/hidden_states/hidden_states_${MODEL}_layer${LAYER_NUM}_${TIMESTAMP}_part*.pth" \
                        --output_file "${RESULTS_DIR}/hidden_states/hidden_states_${MODEL}_layer${LAYER_NUM}_${TIMESTAMP}.pth"

                    # Run evaluation
                    bash scripts/evaluate_completions.sh cais/HarmBench-Llama-2-13b-cls \
                        ${BEHAVIORS_PATH} \
                        ${RESULTS_DIR}/completions/${MODEL}_${TIMESTAMP}_merged.json \
                        ${RESULTS_DIR}/results/${MODEL}_${TIMESTAMP}.json
                fi
            fi
        else
            echo "Invalid mode: $MODE. Use 'sbatch', 'local_parallel', or 'local'."
            exit 1
        fi
    done

    # Final cleanup / merge for local_parallel
    if [ "$MODE" == "local_parallel" ]; then
        echo "Waiting for all parts to complete..."
        FAIL=${PARALLEL_FAIL}
        for gpu_idx in "${!GPU_IDS[@]}"; do
            pid="${GPU_ACTIVE_PIDS[${gpu_idx}]}"
            part="${GPU_ACTIVE_PARTS[${gpu_idx}]}"
            if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
                wait "${pid}" || {
                    FAIL=1
                    FAILED_PARTS+=("${part}")
                }
            fi
        done

        if [ "$FAIL" == "1" ]; then
            echo "Error: One or more parts failed."
            if [ "${#FAILED_PARTS[@]}" -gt 0 ]; then
                echo "Failed parts for ${ALGORITHM}: ${FAILED_PARTS[*]}"
            fi
            exit 1
        fi

        echo "All parts completed. Merging..."
        # Merge completion parts
        python merge_completion_parts.py \
            --input_pattern "${RESULTS_DIR}/completions/${MODEL}_${TIMESTAMP}_part*.json" \
            --output_file "${RESULTS_DIR}/completions/${MODEL}_${TIMESTAMP}_merged.json"
        
        # Merge hidden states parts
        python merge_completion_parts.py \
            --input_pattern "${RESULTS_DIR}/hidden_states/hidden_states_${MODEL}_layer${LAYER_NUM}_${TIMESTAMP}_part*.pth" \
            --output_file "${RESULTS_DIR}/hidden_states/hidden_states_${MODEL}_layer${LAYER_NUM}_${TIMESTAMP}.pth"

        # Run evaluation
        EVAL_LOG="${LOGS_DIR}/eval_${MODEL}_${TIMESTAMP}.log"
        echo "Running evaluation. Logging to ${EVAL_LOG}"
        bash scripts/evaluate_completions.sh cais/HarmBench-Llama-2-13b-cls \
            ${BEHAVIORS_PATH} \
            ${RESULTS_DIR}/completions/${MODEL}_${TIMESTAMP}_merged.json \
            ${RESULTS_DIR}/results/${MODEL}_${TIMESTAMP}.json > "${EVAL_LOG}" 2>&1
    fi


    echo "Completed processing algorithm: ${ALGORITHM}"
    echo "----------------------------------------"
    echo -e "\n=== Completed processing algorithm: ${ALGORITHM} at $(date) ===" >> ${COMMAND_LOG}
    echo "----------------------------------------" >> ${COMMAND_LOG}
done
