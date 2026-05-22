#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/launch_adaptive_method_seed.sh <seed> [options]

Common seeded adaptive-attack launcher for HarmBench-configured methods
(BarrierSteer modes, Circuit Breaker, and other modes resolvable from
harmbench.yaml). Each launched model/mode/seed run can span multiple GPUs;
run_adaptive_attack.py distributes behaviors across workers and aggregates into
one results.json for that run.

Options:
  --models CSV             Models to run (default: gemma_2_9b,qwen_1.5b,mistral_8b,llama2_7b)
  --modes CSV              Modes to run (default: cbf_topk_nonlinear,cbf_qp_nonlinear,cbf_merge_nonlinear)
  --gpus CSV               GPU ids used by each launched run (default: 0,1,2,3,4,5,6,7)
  --workers-per-gpu N      Workers per listed GPU (default: 6)
  --workers N              Total workers per launched run; overrides --workers-per-gpu
  --judge MODEL            Judge model (default: gpt-4o-mini)
  --num-phi N              Force --num-phi for all modes (default: use harmbench.yaml mode default)
  --cbf-k FLOAT            Override CBF steering strength alpha/k for resolved methods
  --cbf-kappa FLOAT        Override CBF LSE smoothing kappa for resolved methods
  --lambda-weight FLOAT    Override SaP positive-violation lambda_weight
  --result-family NAME     Force results/<NAME>/... family (default: auto by mode)
  --n-iterations N         Max attack iterations per behavior (default: 500)
  --n-behaviors N          Forward to run_adaptive_attack for smoke/partial runs
  --start-idx N            Forward to run_adaptive_attack
  --output-suffix SUFFIX   Append suffix to output dir, useful for non-canonical reruns
  --offset N               Skip first N expanded model/mode jobs (default: 0)
  --limit N                Launch/print at most N jobs after offset (default: all)
  --dry-run                Print commands and validate config/artifacts; do not launch
  --resume-existing        Reuse completed rows in output results.json and run only missing rows
  --force                  Launch even if output results.json already exists

Examples:
  # BarrierSteer QP seed43 across four GPUs => 24 workers, one results.json.
  bash scripts/launch_adaptive_method_seed.sh 43 \
    --models gemma_2_9b \
    --modes cbf_qp_nonlinear \
    --gpus 0,1,2,3 \
    --dry-run

  # Circuit Breaker seed43 across four GPUs.
  bash scripts/launch_adaptive_method_seed.sh 43 \
    --models gemma_2_9b \
    --modes circuit_breaker \
    --gpus 4,5,6,7

  # Two seeds in parallel on disjoint GPU groups.
  bash scripts/launch_adaptive_method_seed.sh 42 --models gemma_2_9b --modes circuit_breaker --gpus 0,1,2,3 --force &
  bash scripts/launch_adaptive_method_seed.sh 43 --models gemma_2_9b --modes circuit_breaker --gpus 4,5,6,7 &
  wait
USAGE
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SEED="$1"; shift
MODELS="gemma_2_9b,qwen_1.5b,mistral_8b,llama2_7b"
MODES="cbf_topk_nonlinear,cbf_qp_nonlinear,cbf_merge_nonlinear"
GPUS="0,1,2,3,4,5,6,7"
WORKERS_PER_GPU=6
WORKERS=""
JUDGE_MODEL="gpt-4o-mini"
NUM_PHI=""
CBF_K=""
CBF_KAPPA=""
LAMBDA_WEIGHT=""
RESULT_FAMILY=""
N_ITERATIONS=500
N_BEHAVIORS=""
START_IDX=""
OUTPUT_SUFFIX=""
OFFSET=0
LIMIT=""
DRY_RUN=0
RESUME_EXISTING=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --workers-per-gpu) WORKERS_PER_GPU="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --judge) JUDGE_MODEL="$2"; shift 2 ;;
    --num-phi) NUM_PHI="$2"; shift 2 ;;
    --cbf-k) CBF_K="$2"; shift 2 ;;
    --cbf-kappa) CBF_KAPPA="$2"; shift 2 ;;
    --lambda-weight) LAMBDA_WEIGHT="$2"; shift 2 ;;
    --result-family) RESULT_FAMILY="$2"; shift 2 ;;
    --n-iterations) N_ITERATIONS="$2"; shift 2 ;;
    --n-behaviors) N_BEHAVIORS="$2"; shift 2 ;;
    --start-idx) START_IDX="$2"; shift 2 ;;
    --output-suffix) OUTPUT_SUFFIX="$2"; shift 2 ;;
    --offset) OFFSET="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --resume-existing) RESUME_EXISTING=1; shift ;;
    --force) FORCE=1; shift ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$SEED" in
  42|43|44) ;;
  *) echo "Expected seed 42, 43, or 44; got '$SEED'" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

set -a
[[ -f .env ]] && . ./.env
[[ -f ../.env ]] && . ../.env
set +a

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set in .env for gpt judge}"
: "${HF_TOKEN:?HF_TOKEN must be set in ../.env or environment for gated HF models}"

export WORKSPACE_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

RUN_ROOT="$ROOT/third_party/llm-adaptive-attacks"
CONFIG="$ROOT/src/barriersteer_pipeline/harmbench/config/harmbench.yaml"
LOG_DIR="$ROOT/logs/adaptive_attack_openai/method_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a MODEL_ARR <<< "$MODELS"
IFS=',' read -r -a MODE_ARR <<< "$MODES"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

if [[ ${#GPU_ARR[@]} -eq 0 ]]; then
  echo "No GPUs specified" >&2
  exit 2
fi
if [[ "$WORKERS_PER_GPU" -lt 1 ]]; then
  echo "--workers-per-gpu must be >= 1" >&2
  exit 2
fi
if [[ -z "$WORKERS" ]]; then
  WORKERS=$(( ${#GPU_ARR[@]} * WORKERS_PER_GPU ))
fi

printf 'pid\tseed\tmodel\tmode\tgpus\tworkers\toutput\tlog\n' > "$LOG_DIR/pids.tsv"
printf 'seed\tmodel\tmode\tgpus\tworkers\toutput\tlog\n' > "$LOG_DIR/manifest.tsv"

mode_result_family() {
  local mode="$1"
  if [[ -n "$RESULT_FAMILY" ]]; then
    printf '%s' "$RESULT_FAMILY"
  elif [[ "$mode" == "circuit_breaker" ]]; then
    printf 'adaptive_attack_circuit_breaker'
  else
    printf 'adaptive_attack'
  fi
}

sanitize_gpus() { echo "$1" | tr ',' '-'; }

idx=0
selected=0
for model in "${MODEL_ARR[@]}"; do
  for mode in "${MODE_ARR[@]}"; do
    if (( idx < OFFSET )); then
      idx=$((idx + 1))
      continue
    fi
    if [[ -n "$LIMIT" && "$selected" -ge "$LIMIT" ]]; then
      idx=$((idx + 1))
      continue
    fi

    family="$(mode_result_family "$mode")"
    suffix="${mode}_steerseed_${SEED}${OUTPUT_SUFFIX}"
    out="$RUN_ROOT/results/$family/$model/$suffix"
    log="$LOG_DIR/${idx}_${model}_${mode}_seed${SEED}_gpus$(sanitize_gpus "$GPUS")_w${WORKERS}.log"

    resolver_args=(--config "$CONFIG" --model "$model" --mode "$mode" --seed "$SEED" --format json)
    [[ -n "$NUM_PHI" ]] && resolver_args+=(--num-phi "$NUM_PHI")
    [[ -n "$CBF_K" ]] && resolver_args+=(--cbf-k "$CBF_K")
    [[ -n "$CBF_KAPPA" ]] && resolver_args+=(--cbf-kappa "$CBF_KAPPA")
    resolved_json="$(python -m barriersteer_pipeline.harmbench.utils.steering_config_resolver "${resolver_args[@]}")"
    resolved_meta="$(RESOLVED_JSON="$resolved_json" python - <<'PY'
import json, os
data=json.loads(os.environ['RESOLVED_JSON'])
cfg=data['harmbench_model_cfg']
print(str(data.get('num_phi', '')) + '\t' + (cfg.get('polytope_weight_path') or cfg.get('custom_adapter_path') or ''))
PY
)"
    resolved_num_phi="${resolved_meta%%$'\t'*}"
    artifact_path="${resolved_meta#*$'\t'}"
    if [[ -n "$artifact_path" && "$artifact_path" == /* && ! -e "$artifact_path" ]]; then
      echo "Missing resolved artifact for $model $mode seed $SEED: $artifact_path" >&2
      exit 1
    fi
    if [[ -f "$out/results.json" && "$FORCE" -ne 1 && "$RESUME_EXISTING" -ne 1 ]]; then
      echo "Skipping existing result: $out/results.json (use --force to rerun or --resume-existing to continue)" >&2
      idx=$((idx + 1))
      continue
    fi

    extra_args=(--n-iterations "$N_ITERATIONS")
    [[ "$RESUME_EXISTING" -eq 1 ]] && extra_args+=(--resume-existing)
    [[ -n "$N_BEHAVIORS" ]] && extra_args+=(--n-behaviors "$N_BEHAVIORS")
    [[ -n "$START_IDX" ]] && extra_args+=(--start-idx "$START_IDX")

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$SEED" "$model" "$mode" "$GPUS" "$WORKERS" "$out" "$log" >> "$LOG_DIR/manifest.tsv"

    if [[ "$DRY_RUN" -eq 1 ]]; then
      selected=$((selected + 1))
      printf 'DRY-RUN seed=%s model=%s mode=%s num_phi=%s gpus=%s workers=%s artifact=%s out=%s\n' \
        "$SEED" "$model" "$mode" "$resolved_num_phi" "$GPUS" "$WORKERS" "$artifact_path" "$out"
    else
      quoted_extra=""
      if [[ ${#extra_args[@]} -gt 0 ]]; then
        printf -v quoted_extra " %q" "${extra_args[@]}"
      fi
      num_phi_arg=""
      if [[ -n "$NUM_PHI" ]]; then
        num_phi_arg=" --num-phi $(printf %q "$NUM_PHI")"
      fi
      cbf_k_arg=""
      if [[ -n "$CBF_K" ]]; then
        cbf_k_arg=" --cbf-k $(printf %q "$CBF_K")"
      fi
      cbf_kappa_arg=""
      if [[ -n "$CBF_KAPPA" ]]; then
        cbf_kappa_arg=" --cbf-kappa $(printf %q "$CBF_KAPPA")"
      fi
      lambda_weight_arg=""
      if [[ -n "$LAMBDA_WEIGHT" ]]; then
        lambda_weight_arg=" --lambda-weight $(printf %q "$LAMBDA_WEIGHT")"
      fi
      launch_cmd="cd $(printf %q "$RUN_ROOT") && exec python -u run_adaptive_attack.py --target-model $(printf %q "safe_${model}") --harmbench-config $(printf %q "$CONFIG") --steering-mode $(printf %q "$mode")$num_phi_arg$cbf_k_arg$cbf_kappa_arg$lambda_weight_arg --steer-seed $(printf %q "$SEED") --judge-model $(printf %q "$JUDGE_MODEL") --workers $(printf %q "$WORKERS") --gpus $(printf %q "$GPUS") --output-dir $(printf %q "$out")$quoted_extra"
      setsid bash -lc "$launch_cmd" > "$log" 2>&1 < /dev/null &
      pid=$!
      selected=$((selected + 1))
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$pid" "$SEED" "$model" "$mode" "$GPUS" "$WORKERS" "$out" "$log" | tee -a "$LOG_DIR/pids.tsv"
    fi

    idx=$((idx + 1))
  done
done

echo "log_dir=$LOG_DIR"
if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "Launched jobs. Monitor with: tail -f $LOG_DIR/*.log"
fi
