#!/bin/bash

MODEL=$1
MODE=$2
ALGORITHMS=$3
echo "DEBUG: ALL ARGS: $@"
echo "DEBUG: Arg 20 (Dataset): '${20}'"
EXP_IDENT=${4:-""}  # New argument, defaults to empty string
PART_SIZE=${5:-"500"}
POLYTOPE_PATH=${6:-""}  # New argument for polytope weight path
USE_DEFENSE=${7:-""}  # New argument for using defense
DEFENSE_METHOD=${8:-""}  # New argument for defense method
# Optional CBF steering arguments (for SafetyPolytope polytope model)
USE_CBF=${9:-""}
CBF_MODE=${10:-"estimated"}
CBF_DT=${11:-"1.0"}
CBF_K=${12:-"1.0"}
CBF_W=${13:-"1.0"}
CBF_P=${14:-"10.0"}
CBF_MAX=${15:-"2"}
CBF_CONSTRAINT_MODE=${16:-"topk"}
# Optional kappa controlling LogSumExp-based constraint merging sharpness
CBF_KAPPA=${17:-"10.0"}
# Control radius (limits steering magnitude ||u|| <= R) - applies to all CBF modes
CBF_CONTROL_RADIUS=${18:-"1.0"}
# Number of repeated CBF updates per generated token
CBF_NUM_STEPS=${19:-"1"}
# Optional behavior dataset for evaluation
BEHAVIOR_DATASET=${20:-"./data/behavior_datasets/harmbench_behaviors_text_all.csv"}
# Starting batch size
STARTING_BATCH_SIZE=${21:-"16"}
# Multi-CBF arguments
MULTI_CBF_ENABLED=${22:-"False"}
MULTI_CBF_MODELS_DIR=${23:-""}
MULTI_CBF_LOAD_ATTACKS=${24:-""}
# Generation seed for reproducibility
GENERATION_SEED=${25:-""}
CUSTOM_MODEL_PATH=${26:-""}
CUSTOM_ADAPTER_PATH=${27:-""}
REFT_R1_VECTOR_PATH=${28:-""}
FILENAME_SUFFIX=${EXP_IDENT}

PARTITION="cais"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

CONFIG_FILE="configs/model_configs/models.yaml"

CONFIG_FILE="configs/model_configs/models.yaml"
PYTHON="${PYTHON_EXECUTABLE:-python3}"
MERGED_TMP_PATH_FILE="./slurm_logs/merged_tmp_${MODEL}_${EXP_IDENT:-default}.path"
mkdir -p ./slurm_logs
rm -f "${MERGED_TMP_PATH_FILE}"
cleanup_merged_tmp() {
    if [ -f "${MERGED_TMP_PATH_FILE}" ]; then
        MERGED_TMP_DIR=$(cat "${MERGED_TMP_PATH_FILE}" 2>/dev/null)
        if [ -n "${MERGED_TMP_DIR}" ] && [ -d "${MERGED_TMP_DIR}" ]; then
            rm -rf "${MERGED_TMP_DIR}"
        fi
        rm -f "${MERGED_TMP_PATH_FILE}"
    fi
}
trap cleanup_merged_tmp EXIT
flock "${CONFIG_FILE}.lock" $PYTHON -c "
import yaml
import sys
import os
import tempfile

with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)

model_cfg = config['${MODEL}']['model']

base_model_path = model_cfg.get('source_model_name_or_path') or model_cfg.get('pretrained_model_name_or_path') or model_cfg['model_name_or_path']
model_cfg['model_name_or_path'] = base_model_path

if '${CUSTOM_MODEL_PATH}':
    base_model_path = '${CUSTOM_MODEL_PATH}'
    model_cfg['model_name_or_path'] = '${CUSTOM_MODEL_PATH}'

if '${CUSTOM_ADAPTER_PATH}':
    import tempfile
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = tempfile.mkdtemp(prefix='harmbench_cb_merged_')
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype='auto', device_map='auto')
    model = PeftModel.from_pretrained(model, '${CUSTOM_ADAPTER_PATH}')
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(merged_dir)
    model_cfg['model_name_or_path'] = merged_dir
    with open('${MERGED_TMP_PATH_FILE}', 'w', encoding='utf-8') as tmp_f:
        tmp_f.write(merged_dir)
    print(f'Merged CircuitBreaker adapter into temporary model dir: {merged_dir}')
for transient_key in ('source_model_name_or_path', 'api_base', 'token'):
    if transient_key in model_cfg:
        del model_cfg[transient_key]
if '${GENERATION_SEED}':
    model_cfg['seed'] = int('${GENERATION_SEED}')
# Update Basic Polytope Settings
if '${POLYTOPE_PATH}':
    model_cfg['polytope_weight_path'] = '${POLYTOPE_PATH}'
else:
    # Remove polytope path and all polytope-specific parameters for base model
    if 'polytope_weight_path' in model_cfg:
        del model_cfg['polytope_weight_path']
    # Remove other polytope-specific parameters that aren't valid for base models
    polytope_params = ['lambda_weight', 'safe_violation_weight', 'steer_layer', 'use_backup_response']
    for param in polytope_params:
        if param in model_cfg:
            del model_cfg[param]

# Update Standard CBF Settings
if '${USE_CBF}' == 'True':
    model_cfg['use_cbf'] = True
    model_cfg['cbf_mode'] = '${CBF_MODE}'
    model_cfg['cbf_dt'] = float('${CBF_DT}')
    model_cfg['cbf_k'] = float('${CBF_K}')
    model_cfg['cbf_w'] = float('${CBF_W}')
    model_cfg['cbf_p'] = float('${CBF_P}')
    model_cfg['cbf_kappa'] = float('${CBF_KAPPA}')
    if '${CBF_MAX}':
        cbf_max_constraints = int('${CBF_MAX}')
        if cbf_max_constraints > 0:
            model_cfg['cbf_max_constraints'] = cbf_max_constraints
        elif 'cbf_max_constraints' in model_cfg:
            del model_cfg['cbf_max_constraints']
    model_cfg['cbf_constraint_mode'] = '${CBF_CONSTRAINT_MODE}'
    model_cfg['cbf_num_steps'] = max(1, int('${CBF_NUM_STEPS}' or '1'))
else:
    model_cfg['use_cbf'] = False
    for key in ('cbf_mode','cbf_dt','cbf_k','cbf_w','cbf_p','cbf_kappa','cbf_max_constraints','cbf_constraint_mode','cbf_num_steps'):
        if key in model_cfg:
            del model_cfg[key]

# Control Radius (applies to all modes)
if '${CBF_CONTROL_RADIUS}':
    model_cfg['cbf_control_radius'] = float('${CBF_CONTROL_RADIUS}')
elif 'cbf_control_radius' in model_cfg:
    del model_cfg['cbf_control_radius']

# Update ReFT-r1 activation intervention settings
if '${REFT_R1_VECTOR_PATH}':
    model_cfg['use_reft_r1'] = True
    model_cfg['reft_r1_vector_path'] = '${REFT_R1_VECTOR_PATH}'
else:
    for key in ('use_reft_r1', 'reft_r1_vector_path', 'reft_r1_target_layer', 'reft_r1_top_k', 'reft_r1_beta'):
        if key in model_cfg:
            del model_cfg[key]

# Update Multi-CBF Settings
if '${MULTI_CBF_ENABLED}' == 'True':
    model_cfg['multi_cbf_enabled'] = True
    model_cfg['multi_cbf_models_dir'] = '${MULTI_CBF_MODELS_DIR}'
    if '${MULTI_CBF_LOAD_ATTACKS}':
        model_cfg['multi_cbf_load_attacks'] = '${MULTI_CBF_LOAD_ATTACKS}'
    # If multi-CBF is enabled, unset single polytope path to avoid confusion
    if 'polytope_weight_path' in model_cfg:
        del model_cfg['polytope_weight_path']
elif 'multi_cbf_enabled' in model_cfg:
    # Ensure it's disabled if not requested
    model_cfg['multi_cbf_enabled'] = False
    for key in ('multi_cbf_models_dir','multi_cbf_load_attacks'):
        if key in model_cfg:
            del model_cfg[key]

fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename('${CONFIG_FILE}') + '.', suffix='.tmp', dir=os.path.dirname('${CONFIG_FILE}') or '.')
with os.fdopen(fd, 'w') as f:
    yaml.safe_dump(config, f)
os.replace(tmp_path, '${CONFIG_FILE}')
"

# Split the algorithms string into an array
IFS=',' read -ra ALGORITHM_ARRAY <<< "$ALGORITHMS"

# --- Determine Parallelism / N_JOBS ---
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    IFS=',' read -ra GPU_IDS_ARR <<< "${CUDA_VISIBLE_DEVICES}"
    GPU_COUNT=${#GPU_IDS_ARR[@]}
else
    GPU_COUNT=$($PYTHON -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    if [ -z "${GPU_COUNT}" ] || [ "${GPU_COUNT}" -lt 1 ]; then
         GPU_COUNT=1
    fi
fi

N_JOBS=8  # Default for Slurm
if [ "$MODE" == "local_parallel" ]; then
    N_JOBS=${GPU_COUNT}
    echo "Detected ${GPU_COUNT} GPUs. Will distribute tasks into ${N_JOBS} jobs."
elif [ "$MODE" == "slurm" ]; then
    N_JOBS=10
    echo "Running in Slurm mode. Will distribute tasks into ${N_JOBS} jobs."
else
    # Local sequential
    N_JOBS=1
fi

export N_JOBS

# 2. Generate Tasks and Distribute to Batch Configs
# This python script calculates all chunks (Algo, Part) and writes K batch config files.

$PYTHON -c "
import json, csv, sys, os, math

algorithms = '${ALGORITHMS}'.split(',')
behaviors_path = '${BEHAVIOR_DATASET}'
part_size = int('${PART_SIZE}')
model_name = '${MODEL}'
n_jobs = int(os.environ['N_JOBS'])
filename_suffix = '${FILENAME_SUFFIX}'
results_dir = os.environ.get('HARMBENCH_RESULTS_DIR', './results')

# 1. Load behavior count cache logic (simplified)
valid_behaviors = set()
try:
    with open(behaviors_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'BehaviorID' in row:
                valid_behaviors.add(row['BehaviorID'])
except Exception as e:
    print(f'Error reading behaviors: {e}')

# 2. Build list of all tasks
all_tasks = []

print(f'{n_jobs} Jobs Planned.')

for algo in algorithms:
    algo = algo.strip()
    if not algo: continue
    
    # Resolve paths
    test_case_model = model_name
    if test_case_model.startswith('safe_'): test_case_model = test_case_model[5:]
    if test_case_model.startswith('defense_'): test_case_model = test_case_model[8:]
    
    if algo == 'DirectRequest':
        save_dir = f'{results_dir}/{algo}/default'
        tc_dir = f'{results_dir}/{algo}/default'
    elif algo == 'HumanJailbreaks':
        save_dir = f'{results_dir}/{algo}/random_subset_5'
        tc_dir = f'{results_dir}/{algo}/random_subset_5'
    elif algo == 'PAP-top5':
        save_dir = f'{results_dir}/PAP/top_5'
        tc_dir = f'{results_dir}/PAP/top_5'
    else:
        save_dir = f'{results_dir}/{algo}/{model_name}'
        tc_dir = f'{results_dir}/{algo}/{test_case_model}'
    
    test_cases_file = f'{tc_dir}/test_cases/test_cases.json'
    
    print(f\"DEBUG: Executable: {sys.executable}\")
    print(f\"DEBUG: Checking {test_cases_file}\")
    print(f\"DEBUG: Repr path: {repr(test_cases_file)}\")
    print(f\"DEBUG: Exists: {os.path.exists(test_cases_file)}\")

    # Count cases
    total_cases = 0
    try:
        with open(test_cases_file, 'r') as f:
            data = json.load(f)
        for bid, cases in data.items():
            if not valid_behaviors or bid in valid_behaviors:
                total_cases += len(cases)
    except Exception as e:
        print(f'Error counting cases for {algo}: {e} (File: {test_cases_file})')
        # Fallback if file missing? 
        continue

    num_parts = math.ceil(total_cases / part_size) if total_cases > 0 else 0
    print(f'Algorithm: {algo}, Total Cases: {total_cases}, Parts: {num_parts}')
    
    # Create tasks for each part
    for p in range(1, num_parts + 1):
        task = {
            'behaviors_path': behaviors_path,
            'test_cases_path': test_cases_file,
            'save_path': f'{save_dir}/completions/{model_name}_{filename_suffix}.json',
            'max_new_tokens': 64,
            'incremental_update': False,
            'use_defense': '${USE_DEFENSE}' == 'True',
            'defense_method': '${DEFENSE_METHOD}',
            'part_size': part_size,
            'part_number': p,
            'algorithm': algo
        }
        all_tasks.append(task)

print(f'Total Tasks (Chunks): {len(all_tasks)}')

# 3. Cap jobs to the actual number of tasks
if all_tasks:
    n_jobs = min(n_jobs, len(all_tasks))
else:
    n_jobs = 0
print(f'Using {n_jobs} job(s) after task count adjustment.')

# 4. Distribute tasks into N_JOBS buckets (Round Robin)
batches = [[] for _ in range(n_jobs)]
for i, task in enumerate(all_tasks):
    bucket = i % n_jobs
    batches[bucket].append(task)

# 5. Write batch config files
for job_idx, tasks in enumerate(batches):
    if not tasks: continue
    config_file = f'./slurm_logs/batch_config_{model_name}_{filename_suffix}_job{job_idx}.json'
    with open(config_file, 'w') as f:
        json.dump(tasks, f, indent=4)
    print(f'Created batch config for Job {job_idx}: {len(tasks)} tasks -> {config_file}')
"

# 3. Submit Jobs
GEN_JOB_IDS=()
PIDS=()

if [ "$MODE" == "local_parallel" ]; then
    # We already detected GPU_COUNT inside the python logic? 
    # Wait, we need it here for the loop
    # Re-detect or reuse logic (it was detected above!)
    GPU_IDS=()
    if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
        IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    else
        for ((g=0; g<GPU_COUNT; g++)); do
            GPU_IDS+=("${g}")
        done
    fi
fi

for BATCH_CONFIG_FILE in ./slurm_logs/batch_config_${MODEL}_${FILENAME_SUFFIX}_job*.json; do
    if [ ! -f "$BATCH_CONFIG_FILE" ]; then
        continue
    fi

    JOB_IDX=${BATCH_CONFIG_FILE##*_job}
    JOB_IDX=${JOB_IDX%.json}

    GENERATE_COMMAND="$PYTHON -u generate_completions.py \
        --model_name ${MODEL} \
        --starting_batch_size ${STARTING_BATCH_SIZE} \
        --batch_config ${BATCH_CONFIG_FILE}"

    # Use vLLM only for plain model generation paths.
    # Polytope/CAST/CBF modes must go through SafeRepModel (HF path), otherwise steering is bypassed.
    if [ -z "${POLYTOPE_PATH}" ] && [ "${USE_DEFENSE}" != "True" ] && [ -z "${REFT_R1_VECTOR_PATH}" ] && [ -z "${CUSTOM_ADAPTER_PATH}" ]; then
        GENERATE_COMMAND="${GENERATE_COMMAND} --generate_with_vllm"
    fi
    
    # Add seed if provided
    if [ -n "${GENERATION_SEED}" ]; then
        GENERATE_COMMAND="${GENERATE_COMMAND} --seed ${GENERATION_SEED}"
    fi

    JOB_NAME="gen_batch_${MODEL}_job${JOB_IDX}"
    
    if [ "$MODE" == "slurm" ]; then
        OUTPUT_LOG="./slurm_logs/batch_gen_${MODEL}_${FILENAME_SUFFIX}_job${JOB_IDX}.log"
        FULL_SBATCH_COMMAND="sbatch --partition=${PARTITION} \
            --job-name=${JOB_NAME} \
            --nodes=1 \
            --gpus-per-node=1 \
            --time=${TIME} \
            --output=${OUTPUT_LOG} \
            --mem-per-cpu=${MEM} \
            --gres=gpumem:${GPUMEM} \
            --parsable \
            --wrap=\"${GENERATE_COMMAND}\""
        
        echo "Submitting: ${FULL_SBATCH_COMMAND}" >> ${COMMAND_LOG}
        GENERATE_JOB_ID=$(eval "${FULL_SBATCH_COMMAND}")
        echo "Submitted generation job ${GENERATE_JOB_ID}"
        GEN_JOB_IDS+=("${GENERATE_JOB_ID}")

    elif [ "$MODE" == "local" ]; then
        echo "Running locally: ${GENERATE_COMMAND}"
        eval ${GENERATE_COMMAND}
        
    elif [ "$MODE" == "local_parallel" ]; then
        # Throttling
        while [ "$(jobs -r | wc -l)" -ge "${GPU_COUNT}" ]; do
            sleep 1
        done

        # Assign GPU (Round Robin on jobs is redundant since JOB_IDX < GPU_COUNT usually, but safe)
        GPU_PTR=$(( JOB_IDX % GPU_COUNT ))
        GPU_ID="${GPU_IDS[${GPU_PTR}]}"
        
        echo "Running local_parallel Job ${JOB_IDX} on GPU ${GPU_ID}"
        CUDA_VISIBLE_DEVICES=${GPU_ID} eval ${GENERATE_COMMAND} > "./slurm_logs/generate_${MODEL}_job${JOB_IDX}.log" 2>&1 &
        PIDS+=($!)
    fi
done

# --- Post-Generation: Merge and Evaluate ---
# We need to wait for all parts to finish.

# Construct dependency string (only needed for SLURM)
if [ "$MODE" == "slurm" ]; then
    DEP_STRING=$(IFS=:; echo "${GEN_JOB_IDS[*]}")
fi

# Create Batch Eval Config (Common for both)
# We only need ONE eval config, which lists all algorithms.
EVAL_BATCH_CONFIG="./slurm_logs/batch_eval_config_${MODEL}_${FILENAME_SUFFIX}.json"

$PYTHON -c "
import json
import os
tasks = []
algorithms = '${ALGORITHMS}'.split(',')
for algo in algorithms:
    algo = algo.strip()
    if not algo: continue
    
    print(f'Generating eval config for: {algo}')
    print(f'Generating eval config for: {algo}')
    results_dir = os.environ.get('HARMBENCH_RESULTS_DIR', './results')
    if algo == 'DirectRequest':
        save_dir = f'{results_dir}/{algo}/default'
    elif algo == 'HumanJailbreaks':
        save_dir = f'{results_dir}/{algo}/random_subset_5'
    elif algo == 'PAP-top5':
        save_dir = f'{results_dir}/PAP/top_5'
    else:
        save_dir = f'{results_dir}/{algo}/${MODEL}'
        
    tasks.append({
        'behaviors_path': '${BEHAVIOR_DATASET}',
        'completions_path': f'{save_dir}/completions/${MODEL}_${FILENAME_SUFFIX}_merged.json',
        'save_path': f'{save_dir}/results/${MODEL}_${FILENAME_SUFFIX}.json',
        'include_advbench_metric': False, # simplified
        'num_tokens': 512
    })
    
with open('${EVAL_BATCH_CONFIG}', 'w') as f:
    json.dump(tasks, f, indent=4)
"

# Create Merge Script that merges parts for ALL algorithms (Common for both)
MERGE_SCRIPT="./slurm_logs/merge_batch_${MODEL}_${FILENAME_SUFFIX}.sh"
cat > "${MERGE_SCRIPT}" <<EOF
#!/bin/bash
ALGORITHMS="${ALGORITHMS}"
IFS=',' read -ra ALGOS <<< "\$ALGORITHMS"
for algo in "\${ALGOS[@]}"; do
    algo=\$(echo "\$algo" | xargs)
    RESULTS_BASE="\${HARMBENCH_RESULTS_DIR:-"./results"}"
    if [ "\$algo" == "DirectRequest" ]; then
        RESULTS_DIR="\${RESULTS_BASE}/\$algo/default"
    elif [ "\$algo" == "HumanJailbreaks" ]; then
        RESULTS_DIR="\${RESULTS_BASE}/\$algo/random_subset_5"
    elif [ "\$algo" == "PAP-top5" ]; then
        RESULTS_DIR="\${RESULTS_BASE}/PAP/top_5"
    else
        RESULTS_DIR="\${RESULTS_BASE}/\$algo/${MODEL}"
    fi

    echo "Merging for \$algo..."
    echo "Merging for \$algo..."
    ${PYTHON_EXECUTABLE:-python3} merge_completion_parts.py \\
        --input_pattern "\${RESULTS_DIR}/completions/${MODEL}_${FILENAME_SUFFIX}_part*.json" \\
        --output_file "\${RESULTS_DIR}/completions/${MODEL}_${FILENAME_SUFFIX}_merged.json"
        
    echo "Merging timing for \$algo..."
    ${PYTHON_EXECUTABLE:-python3} merge_completion_parts.py \\
        --input_pattern "\${RESULTS_DIR}/completions/${MODEL}_${FILENAME_SUFFIX}_part*_timing.json" \\
        --output_file "\${RESULTS_DIR}/completions/${MODEL}_${FILENAME_SUFFIX}_merged_timing.json"
done

# Run Batch Evaluation
${PYTHON_EXECUTABLE:-python3} evaluate_completions.py \\
    --cls_path cais/HarmBench-Llama-2-13b-cls \\
    --batch_config ${EVAL_BATCH_CONFIG}
EOF
chmod +x ${MERGE_SCRIPT}

if [ "$MODE" == "slurm" ]; then
    FULL_EVAL_COMMAND="sbatch --partition=${PARTITION} \
        --job-name=eval_batch_${MODEL} \
        --nodes=1 \
        --gpus-per-node=1 \
        --time=4:00:00 \
        --output=./slurm_logs/eval_batch_${MODEL}_${FILENAME_SUFFIX}.log \
        --mem-per-cpu=${MEM} \
        --gres=gpumem:40g \
        --dependency=afterok:${DEP_STRING} \
        --parsable \
        ${MERGE_SCRIPT}"

    echo "Submitting Eval: ${FULL_EVAL_COMMAND}" >> ${COMMAND_LOG}
    EVAL_JOB_ID=$(eval "${FULL_EVAL_COMMAND}")
    echo "Submitted Merge & Eval Job ${EVAL_JOB_ID}"
    
elif [ "$MODE" == "local_parallel" ]; then
    failed=0
    for pid in "${PIDS[@]}"; do
        if ! wait "${pid}"; then
            echo "ERROR: Generation job with PID ${pid} failed."
            failed=1
        fi
    done
    
    if [ "${failed}" -ne 0 ]; then
        echo "ERROR: One or more generation parts failed. Aborting merge/eval."
        exit 1
    fi

    # Run merge and eval locally
    echo "Running merge and eval locally..."
    bash ${MERGE_SCRIPT}

elif [ "$MODE" == "local" ]; then
    # Run merge and eval locally (sequential execution already finished)
    echo "Running merge and eval locally..."
    bash ${MERGE_SCRIPT}
fi
