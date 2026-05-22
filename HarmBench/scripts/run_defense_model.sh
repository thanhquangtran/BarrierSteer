#!/bin/bash

# Parameters
MODEL_NAME="defense_mistral_8b"  # Model name parameter
# MODEL_NAME="defense_llama2_7b"  # Model name parameter
# MODEL_NAME="defense_qwen_1.5b"  # Model name parameter
NUM_SAMPLES=500            # Number of samples parameter
MODE="sbatch"              # Mode parameter (sbatch or local)
USE_DEFENSE="True"        # Whether to use defense

# Methods
# METHODS="AutoPrompt,DirectRequest,GCG,UAT,HumanJailbreaks,GBDA,PEZ"
# METHODS="DirectRequest,GCG,UAT,HumanJailbreaks,GBDA,PEZ"
METHODS="AutoDAN"

# Define array of defense methods
DEFENSE_METHODS=("response_check" "smoothllm" "ICL" "self_reminder")

# Loop through each defense method
for DEFENSE_METHOD in "${DEFENSE_METHODS[@]}"; do
    echo "Starting defense method: ${DEFENSE_METHOD}"

    # Loop through seeds 1-5
    for SEED in {1..5}; do
        EXP_IDENT="${DEFENSE_METHOD}_seed_${SEED}"

        # Run step_2_and_3.sh with all methods at once
        COMMAND="./scripts/step_2_and_3.sh \
            ${MODEL_NAME} \
            ${MODE} \
            ${METHODS} \
            ${EXP_IDENT} \
            ${NUM_SAMPLES} \
            \"\" \
            ${USE_DEFENSE} \
            ${DEFENSE_METHOD}"

        echo "Executing ${DEFENSE_METHOD} with seed ${SEED}: $COMMAND"
        eval $COMMAND
    done
done
