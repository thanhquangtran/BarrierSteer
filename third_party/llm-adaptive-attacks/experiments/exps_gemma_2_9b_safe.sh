#!/bin/bash
#
# Experiment script: Run adaptive attacks on Gemma 2 9B (safe vs base)
#
# Usage:
#   # Quick smoke test (1 behavior, debug mode):
#   bash experiments/exps_gemma_2_9b_safe.sh test
#
#   # Run on 5 behaviors:  
#   bash experiments/exps_gemma_2_9b_safe.sh small
#
#   # Full run (all 50 behaviors):
#   bash experiments/exps_gemma_2_9b_safe.sh full
#
# Prerequisites:
#   export OPENROUTER_API_KEY=sk-or-...
#   export CUDA_VISIBLE_DEVICES=0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Check API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set. Export it first."
    echo "  export OPENROUTER_API_KEY=sk-or-..."
    exit 1
fi

MODE=${1:-"test"}
JUDGE_MODEL="openrouter/openai/gpt-4o-mini"

case "$MODE" in
    test)
        N_BEHAVIORS=1
        N_ITERATIONS=100
        DEBUG="--debug"
        echo "=== SMOKE TEST MODE: 1 behavior, 100 iterations ==="
        ;;
    small)
        N_BEHAVIORS=5
        N_ITERATIONS=2000
        DEBUG=""
        echo "=== SMALL MODE: 5 behaviors, 2000 iterations ==="
        ;;
    full)
        N_BEHAVIORS=""  # all 50
        N_ITERATIONS=5000
        DEBUG=""
        echo "=== FULL MODE: all 50 behaviors, 5000 iterations ==="
        ;;
    *)
        echo "Unknown mode: $MODE. Use: test, small, or full"
        exit 1
        ;;
esac

# ============ STEERED MODEL (CBF) ============
echo ""
echo ">>> Starting attack on STEERED model (safe_gemma_2_9b) ..."
echo ""

CMD="python run_adaptive_attack.py \
    --target-model safe_gemma_2_9b \
    --judge-model $JUDGE_MODEL \
    --prompt-template refined_best \
    --n-iterations $N_ITERATIONS \
    --determinstic-jailbreak \
    --schedule-n-to-change \
    $DEBUG"

if [ -n "$N_BEHAVIORS" ]; then
    CMD="$CMD --n-behaviors $N_BEHAVIORS"
fi

eval $CMD

# ============ BASE MODEL (no defense) ============
echo ""
echo ">>> Starting attack on BASE model (gemma-2-9b) ..."
echo ""

CMD="python run_adaptive_attack.py \
    --target-model gemma-2-9b \
    --judge-model $JUDGE_MODEL \
    --prompt-template refined_best \
    --n-iterations $N_ITERATIONS \
    --determinstic-jailbreak \
    --schedule-n-to-change \
    $DEBUG"

if [ -n "$N_BEHAVIORS" ]; then
    CMD="$CMD --n-behaviors $N_BEHAVIORS"
fi

eval $CMD

echo ""
echo "=== DONE ==="
echo "Results in: results/adaptive_attack/"
