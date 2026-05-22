#!/bin/bash
# Script to run LM Harness evaluations with configurable parameters.
# Usage: ./scripts/eval_lm_harness_batch.sh <MODEL_NAME> <STEER_TYPE> <SEED> <K> <BATCH_SIZE> [KAPPA]

set -euo pipefail

MODEL_NAME=${1:-}
STEER_TYPE=${2:-}
SEED=${3:-}
K=${4:-}
BATCH_SIZE=${5:-}
KAPPA=${6:-1.0}

if [ -z "$MODEL_NAME" ] || [ -z "$STEER_TYPE" ] || [ -z "$SEED" ] || [ -z "$BATCH_SIZE" ]; then
  echo "Usage: $0 <MODEL_NAME> <STEER_TYPE> <SEED> <K> <BATCH_SIZE> [KAPPA]"
  echo "MODEL_NAME: llama2_7b, mistral_8b, qwen_1.5b, gemma_2_9b"
  echo "STEER_TYPE: base_model, sap, cbf_topk, cbf_qp, cbf_merge, actadd, dirablate"
  echo "SEED: generation seed (e.g., 42)"
  echo "K: CBF k value (ignored for base_model/sap/actadd/dirablate)"
  echo "BATCH_SIZE: batch size (e.g., 128)"
  echo "KAPPA: merge kappa (default: 1.0, used only for cbf_merge)"
  exit 1
fi

case "$MODEL_NAME" in
  "llama2_7b") MODEL_PATH="meta-llama/Llama-2-7b-chat-hf" ;;
  "mistral_8b") MODEL_PATH="mistralai/Ministral-8B-Instruct-2410" ;;
  "qwen_1.5b") MODEL_PATH="Qwen/Qwen2-1.5B-Instruct" ;;
  "gemma_2_9b") MODEL_PATH="google/gemma-2-9b-it" ;;
  *)
    echo "Error: Unknown model name '$MODEL_NAME'"
    exit 1
    ;;
esac

WORKSPACE_ROOT=$(pwd)
OUTPUT_ROOT="${WORKSPACE_ROOT}/outputs/harmbench/${MODEL_NAME}"
LM_HARNESS_OUT_DIR="${WORKSPACE_ROOT}/outputs/lm_harness/${MODEL_NAME}"
mkdir -p "$LM_HARNESS_OUT_DIR"

TASKS="[mmlu,gsm8k]"
NUM_FEWSHOT=0

find_weight_file() {
  local seed="$1"
  shift
  local prefixes=("$@")

  for prefix in "${prefixes[@]}"; do
    local matches=()
    while IFS= read -r d; do
      matches+=("$d")
    done < <(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "${prefix}*" | sort)

    for d in "${matches[@]}"; do
      if [ -f "${d}/weight_${seed}.pth" ]; then
        echo "${d}/weight_${seed}.pth"
        return 0
      fi
      if [ -f "${d}/weights.pth" ]; then
        echo "${d}/weights.pth"
        return 0
      fi
    done
  done
  return 1
}

find_weight_prefer_dirs() {
  local seed="$1"
  shift
  local dirs=("$@")

  for rel in "${dirs[@]}"; do
    local d="${OUTPUT_ROOT}/${rel}"
    if [ -f "${d}/weight_${seed}.pth" ]; then
      echo "${d}/weight_${seed}.pth"
      return 0
    fi
    if [ -f "${d}/weights.pth" ]; then
      echo "${d}/weights.pth"
      return 0
    fi
  done
  return 1
}

USE_SAFE_REP_MODEL="true"
USE_CBF="false"
PROJECTION="false"
CBF_ARGS=""
STEERING_ARGS=""
OUTPUT_SUFFIX=""
POLYTOPE_WEIGHT_PATH=""

case "$STEER_TYPE" in
  "base_model")
    USE_SAFE_REP_MODEL="false"
    OUTPUT_SUFFIX="base_model"
    ;;

  "sap")
    PROJECTION="true"
    OUTPUT_SUFFIX="sap_seed${SEED}"
    POLYTOPE_WEIGHT_PATH=$(find_weight_prefer_dirs "$SEED" "sap_phi030") || \
    POLYTOPE_WEIGHT_PATH=$(find_weight_file "$SEED" "sap") || {
      echo "Error: could not find SAP weights for seed ${SEED} under ${OUTPUT_ROOT}"
      exit 1
    }
    ;;

  "cbf_topk")
    USE_CBF="true"
    CBF_ARGS="cbf_mode=estimated cbf_dt=1.0 cbf_max_constraints=2 cbf_constraint_mode=topk cbf_k=${K}"
    OUTPUT_SUFFIX="cbf_topk_k${K}_seed${SEED}"
    POLYTOPE_WEIGHT_PATH=$(find_weight_prefer_dirs "$SEED" "cbf_topk_nonlinear_phi004") || \
    POLYTOPE_WEIGHT_PATH=$(find_weight_file "$SEED" "cbf_topk_nonlinear") || {
      echo "Error: could not find CBF topk weights for seed ${SEED} under ${OUTPUT_ROOT}"
      exit 1
    }
    ;;

  "cbf_qp")
    USE_CBF="true"
    CBF_ARGS="cbf_mode=estimated cbf_dt=1.0 cbf_max_constraints=4 cbf_constraint_mode=qp cbf_k=${K}"
    OUTPUT_SUFFIX="cbf_qp_k${K}_seed${SEED}"
    # QP must reuse shared topk nonlinear weights only.
    POLYTOPE_WEIGHT_PATH=$(find_weight_prefer_dirs "$SEED" "cbf_topk_nonlinear_phi004") || {
      echo "Error: missing required shared topk weights for cbf_qp at ${OUTPUT_ROOT}/cbf_topk_nonlinear_phi004 (seed ${SEED})"
      exit 1
    }
    ;;

  "cbf_merge")
    USE_CBF="true"
    CBF_ARGS="cbf_mode=estimated cbf_dt=1.0 cbf_max_constraints=4 cbf_constraint_mode=merge cbf_k=${K} cbf_kappa=${KAPPA}"
    OUTPUT_SUFFIX="cbf_merge_k${K}_kappa${KAPPA}_seed${SEED}"
    # Merge must reuse shared topk nonlinear weights only.
    POLYTOPE_WEIGHT_PATH=$(find_weight_prefer_dirs "$SEED" "cbf_topk_nonlinear_phi004") || {
      echo "Error: missing required shared topk weights for cbf_merge at ${OUTPUT_ROOT}/cbf_topk_nonlinear_phi004 (seed ${SEED})"
      exit 1
    }
    ;;

  "actadd")
    STEERING_ARGS="steering_method=actadd"
    OUTPUT_SUFFIX="actadd"
    POLYTOPE_WEIGHT_PATH=$(find_weight_prefer_dirs "$SEED" "actadd_phi000") || \
    POLYTOPE_WEIGHT_PATH=$(find_weight_file "$SEED" "actadd") || {
      echo "Error: could not find ActAdd weights for seed ${SEED} under ${OUTPUT_ROOT}"
      exit 1
    }
    ;;

  "dirablate")
    STEERING_ARGS="steering_method=dirablate"
    OUTPUT_SUFFIX="dirablate"
    POLYTOPE_WEIGHT_PATH=$(find_weight_prefer_dirs "$SEED" "dirablate_phi000") || \
    POLYTOPE_WEIGHT_PATH=$(find_weight_file "$SEED" "dirablate") || {
      echo "Error: could not find DirAblate weights for seed ${SEED} under ${OUTPUT_ROOT}"
      exit 1
    }
    ;;

  *)
    echo "Error: Unknown steer type '$STEER_TYPE'"
    exit 1
    ;;
esac

OUTPUT_FILE="${LM_HARNESS_OUT_DIR}/${MODEL_NAME}_${OUTPUT_SUFFIX}.json"

CMD="HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 accelerate launch -m barriersteer_pipeline.evaluation.lm_harness_eval \
  model_path=\"$MODEL_PATH\" \
  use_safe_rep_model=$USE_SAFE_REP_MODEL \
  use_defense=false \
  steer_layer=20 \
  use_cbf=$USE_CBF \
  projection=$PROJECTION \
  lm_harness_tasks=$TASKS \
  num_fewshot=$NUM_FEWSHOT \
  batch_size=$BATCH_SIZE \
  lm_harness_output_path=\"$OUTPUT_FILE\""

if [ "$USE_SAFE_REP_MODEL" = "true" ]; then
  CMD+=" polytope_weight_path=\"$POLYTOPE_WEIGHT_PATH\""
fi

if [ -n "$CBF_ARGS" ]; then
  CMD+=" $CBF_ARGS"
fi

if [ -n "$STEERING_ARGS" ]; then
  CMD+=" $STEERING_ARGS"
fi

echo "Running Evaluation for: $MODEL_NAME"
echo "Steer Type: $STEER_TYPE"
echo "Seed: $SEED"
[ -n "$POLYTOPE_WEIGHT_PATH" ] && echo "Weight Path: $POLYTOPE_WEIGHT_PATH"
echo "Output File: $OUTPUT_FILE"

echo "Executing command..."
eval "$CMD"
