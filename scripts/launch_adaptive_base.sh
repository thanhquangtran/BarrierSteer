#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/launch_adaptive_base.sh [options]

Launch adaptive attacks for undefended/base models.

Options:
  --models CSV       Base target aliases (default: gemma-2-9b,qwen-1.5b,mistral-8b,llama2-7b)
  --gpus CSV         GPU ids to use (default: 0,1,2,3,4,5,6,7)
  --slots-per-gpu N  Pack N independent runs onto each GPU (default: 4)
  --workers N        Workers per run_adaptive_attack process (default: 1)
  --judge MODEL      Judge model (default: gpt-4o-mini)
  --n-behaviors N    Forward to run_adaptive_attack for smoke/partial runs
  --start-idx N      Forward to run_adaptive_attack
  --dry-run          Print commands; do not launch
  --force            Launch even if output results.json already exists
USAGE
}

MODELS="gemma-2-9b,qwen-1.5b,mistral-8b,llama2-7b"
GPUS="0,1,2,3,4,5,6,7"
SLOTS_PER_GPU=4
WORKERS=1
JUDGE_MODEL="gpt-4o-mini"
N_BEHAVIORS=""
START_IDX=""
DRY_RUN=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --slots-per-gpu) SLOTS_PER_GPU="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --judge) JUDGE_MODEL="$2"; shift 2 ;;
    --n-behaviors) N_BEHAVIORS="$2"; shift 2 ;;
    --start-idx) START_IDX="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set -a
[[ -f .env ]] && . ./.env
[[ -f ../.env ]] && . ../.env
set +a
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set in .env for gpt judge}"
: "${HF_TOKEN:?HF_TOKEN must be set in ../.env or environment for gated HF models}"

export WORKSPACE_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src"
RUN_ROOT="$ROOT/third_party/llm-adaptive-attacks"
LOG_DIR="$ROOT/logs/adaptive_attack_openai/base_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a MODEL_ARR <<< "$MODELS"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
if [[ "$SLOTS_PER_GPU" -lt 1 ]]; then echo "--slots-per-gpu must be >= 1" >&2; exit 2; fi
if [[ ${#GPU_ARR[@]} -eq 0 ]]; then echo "No GPUs specified" >&2; exit 2; fi

printf 'pid\tmodel\tgpu\tlog\n' > "$LOG_DIR/pids.tsv"
printf 'model\tgpu\toutput\tlog\n' > "$LOG_DIR/manifest.tsv"

selected=0
for model in "${MODEL_ARR[@]}"; do
  gpu="${GPU_ARR[$(((selected / SLOTS_PER_GPU) % ${#GPU_ARR[@]}))]}"
  out_model="${model//-/_}"
  out="$RUN_ROOT/results/adaptive_attack_base/$out_model/default"
  log="$LOG_DIR/${selected}_${out_model}_gpu${gpu}.log"
  if [[ -f "$out/results.json" && "$FORCE" -ne 1 ]]; then
    echo "Skipping existing result: $out/results.json (use --force to rerun)" >&2
    selected=$((selected + 1))
    continue
  fi
  extra_args=()
  [[ -n "$N_BEHAVIORS" ]] && extra_args+=(--n-behaviors "$N_BEHAVIORS")
  [[ -n "$START_IDX" ]] && extra_args+=(--start-idx "$START_IDX")
  printf '%s\t%s\t%s\t%s\n' "$model" "$gpu" "$out" "$log" >> "$LOG_DIR/manifest.tsv"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'DRY-RUN model=%s gpu=%s out=%s\n' "$model" "$gpu" "$out"
  else
    quoted_extra=""
    if [[ ${#extra_args[@]} -gt 0 ]]; then printf -v quoted_extra " %q" "${extra_args[@]}"; fi
    launch_cmd="cd $(printf %q "$RUN_ROOT") && export CUDA_VISIBLE_DEVICES=$(printf %q "$gpu") && exec python -u run_adaptive_attack.py --target-model $(printf %q "$model") --judge-model $(printf %q "$JUDGE_MODEL") --workers $(printf %q "$WORKERS") --gpus $(printf %q "$gpu") --output-dir $(printf %q "$out")$quoted_extra"
    setsid bash -lc "$launch_cmd" > "$log" 2>&1 < /dev/null &
    pid=$!
    printf '%s\t%s\t%s\t%s\n' "$pid" "$model" "$gpu" "$log" | tee -a "$LOG_DIR/pids.tsv"
  fi
  selected=$((selected + 1))
done

echo "log_dir=$LOG_DIR"
if [[ "$DRY_RUN" -eq 0 ]]; then echo "Launched jobs. Monitor with: tail -f $LOG_DIR/*.log"; fi
