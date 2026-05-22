#!/bin/bash
# Benchmark ablation: CBF inference time vs network size and num_phi
# Model: qwen_1.5b, trained with seed=666
# Two ablation axes:
#   1. Network size (S/P1024/P2048/H1536/H1792/L) with num_phi=4
#   2. K (num_phi: 4/12/20/28/36/44/52/60) with S network
# Constraint modes tested: topk, qp, merge
# Usage: bash scripts/benchmark_latency_ablation.sh
# Multi-GPU usage:
#   bash scripts/benchmark_latency_ablation.sh
#     -> auto-detect and use all available GPUs
#   BENCH_GPUS=0,1,2,3 bash scripts/benchmark_latency_ablation.sh
#     -> explicit GPU override
# The scheduler runs one job per GPU and rotates automatically when jobs finish.


CONFIG="src/barriersteer_pipeline/harmbench/config/harmbench.yaml"
MODEL="qwen_1.5b"
SEED=666
BATCH_SIZE="${BATCH_SIZE:-1}"
WARMUP="${WARMUP:-100}"
# Number of benchmark batches (with BATCH_SIZE=1 this equals sample count).
LIMIT="${LIMIT:-10000}"

BASE_WEIGHTS="outputs/harmbench/qwen_1.5b"
OUT_DIR="outputs/benchmark_latency_ablation/qwen_1.5b"
mkdir -p "$OUT_DIR"

CONSTRAINT_MODES=("topk" "qp" "merge")
declare -a JOB_LABELS=()
declare -a JOB_STEER_MODES=()
declare -a JOB_WEIGHTS=()
declare -a JOB_OUTPUTS=()
declare -a JOB_MAXK=()

declare -a GPUS=()
if [ -n "${BENCH_GPUS:-}" ]; then
  IFS=',' read -r -a GPUS <<< "$BENCH_GPUS"
  for i in "${!GPUS[@]}"; do
    GPUS[$i]="$(echo "${GPUS[$i]}" | xargs)"
  done
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')
  fi
fi

if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "No GPUs detected. Set BENCH_GPUS manually, e.g., BENCH_GPUS=0,1"
  exit 1
fi

STATUS_FILE="$(mktemp /tmp/benchmark_latency_ablation_status.XXXXXX)"
trap 'rm -f "$STATUS_FILE"' EXIT

enqueue_job() {
  local label="$1"
  local steer_mode="$2"
  local weight_path="$3"
  local out_json="$4"
  local max_k="${5:-}"
  JOB_LABELS+=("$label")
  JOB_STEER_MODES+=("$steer_mode")
  JOB_WEIGHTS+=("$weight_path")
  JOB_OUTPUTS+=("$out_json")
  JOB_MAXK+=("$max_k")
}

run_job_on_gpu() {
  local gpu="$1"
  local idx="$2"
  local label="${JOB_LABELS[$idx]}"
  local steer_mode="${JOB_STEER_MODES[$idx]}"
  local weight_path="${JOB_WEIGHTS[$idx]}"
  local out_json="${JOB_OUTPUTS[$idx]}"
  local max_k="${JOB_MAXK[$idx]}"

  if [ ! -f "$weight_path" ]; then
    echo "[GPU $gpu] [SKIP] $label: weight not found at $weight_path"
    echo "SKIP|$label|$gpu" >> "$STATUS_FILE"
    return 0
  fi

  echo ""
  echo "[GPU $gpu] --- $label ---"

  local cmd=(
    python scripts/benchmark_latency.py
    --config "$CONFIG"
    --model "$MODEL"
    --mode "$steer_mode"
    --polytope_model_path "$weight_path"
    --batch_size "$BATCH_SIZE"
    --warmup "$WARMUP"
    --limit_batches "$LIMIT"
    --output_json "$out_json"
  )
  if [ -n "$max_k" ]; then
    cmd+=(--cbf_max_constraints "$max_k")
  fi

  if CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"; then
    echo "OK|$label|$gpu" >> "$STATUS_FILE"
  else
    echo "[GPU $gpu] [FAIL] $label exited with error"
    echo "FAIL|$label|$gpu" >> "$STATUS_FILE"
  fi
}

worker_loop() {
  local gpu="$1"
  shift
  local job_idx
  for job_idx in "$@"; do
    run_job_on_gpu "$gpu" "$job_idx"
  done
}

echo "=========================================="
echo " Ablation 1: Network Size (num_phi=4)"
echo "=========================================="

# Network size variants:
#   S=[128], P1024=[1024], P2048=[2048], H1536=[1536,768,384,192],
#   H1792=[1792,896,448,224], L=[2048,1024,512,256]
declare -A SIZE_MODES=(
  ["S"]="cbf_topk_nonlinear_s"
  ["P1024"]="cbf_topk_nonlinear_p1024"
  ["P2048"]="cbf_topk_nonlinear_p2048"
  ["H1536"]="cbf_topk_nonlinear_h1536"
  ["H1792"]="cbf_topk_nonlinear_h1792"
  ["L"]="cbf_topk_nonlinear"
)

for size_label in "S" "P1024" "P2048" "H1536" "H1792" "L"; do
  training_mode="${SIZE_MODES[$size_label]}"
  weight_path="${BASE_WEIGHTS}/${training_mode}_phi004/weight_${SEED}.pth"

  for cmode in "${CONSTRAINT_MODES[@]}"; do
    if [ "$cmode" = "topk" ]; then
      steer_mode="${training_mode}"
    elif [ "$cmode" = "qp" ]; then
      steer_mode="${training_mode/topk/qp}"
    elif [ "$cmode" = "merge" ]; then
      steer_mode="${training_mode/topk/merge}"
    fi

    enqueue_job \
      "Size=$size_label, Mode=$cmode" \
      "$steer_mode" \
      "$weight_path" \
      "${OUT_DIR}/size_${size_label}_${cmode}.json" \
      ""
  done
done

echo ""
echo "=========================================="
echo " Ablation 2: num_phi (S network)"
echo "=========================================="

NUM_PHIS=(4 12 20 28 36 44 52 60)

for np in "${NUM_PHIS[@]}"; do
  np_padded=$(printf "%03d" "$np")
  weight_path="${BASE_WEIGHTS}/cbf_topk_nonlinear_s_phi${np_padded}/weight_${SEED}.pth"

  for cmode in "${CONSTRAINT_MODES[@]}"; do
    if [ "$cmode" = "topk" ]; then
      steer_mode="cbf_topk_nonlinear_s"
      max_k=2
    elif [ "$cmode" = "qp" ]; then
      steer_mode="cbf_qp_nonlinear_s"
      max_k=$np
    elif [ "$cmode" = "merge" ]; then
      steer_mode="cbf_merge_nonlinear_s"
      max_k=$np
    fi

    enqueue_job \
      "num_phi=$np, Mode=$cmode" \
      "$steer_mode" \
      "$weight_path" \
      "${OUT_DIR}/numphi_${np}_${cmode}.json" \
      "$max_k"
  done
done

echo ""
echo "=========================================="
echo " Running Jobs"
echo "=========================================="
echo "Total jobs queued: ${#JOB_LABELS[@]}"
echo "GPUs: ${GPUS[*]}"

declare -a GPU_QUEUES=()
for gidx in "${!GPUS[@]}"; do
  GPU_QUEUES[$gidx]=""
done

for idx in "${!JOB_LABELS[@]}"; do
  gidx=$((idx % ${#GPUS[@]}))
  GPU_QUEUES[$gidx]="${GPU_QUEUES[$gidx]} $idx"
done

declare -a WORKER_PIDS=()
for gidx in "${!GPUS[@]}"; do
  queue="${GPU_QUEUES[$gidx]}"
  if [ -n "$queue" ]; then
    # shellcheck disable=SC2086
    worker_loop "${GPUS[$gidx]}" $queue &
    WORKER_PIDS+=("$!")
  fi
done

for pid in "${WORKER_PIDS[@]}"; do
  wait "$pid"
done

COMPLETED=$(grep -c '^OK|' "$STATUS_FILE" || true)
SKIPPED=$(grep -c '^SKIP|' "$STATUS_FILE" || true)
FAILED=$(grep -c '^FAIL|' "$STATUS_FILE" || true)

echo ""
echo "=========================================="
echo " Summary"
echo "=========================================="
echo "Completed: $COMPLETED | Skipped: $SKIPPED | Failed: $FAILED"
echo "Per-job JSON results in: $OUT_DIR"

echo "All done."
