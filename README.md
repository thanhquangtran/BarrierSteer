# BarrierSteer: LLM Safety via Learning Barrier Steering

BarrierSteer is an inference-time framework for LLM safety that treats hidden-state safety classifiers as Control Barrier Functions (CBFs), enabling principled, modular, and computationally efficient steering of unsafe latent trajectories through learned nonlinear safety constraints while preserving model utility.

## Install with `uv`

Install `uv` first: <https://docs.astral.sh/uv/getting-started/installation/>

Then install the full project from the repository root:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv sync --all-extras
```

Quick check:

```bash
uv run python - <<'PY'
import barriersteer_pipeline
print('BarrierSteer install OK')
PY
```

For GPU runs, make sure CUDA is visible:

```bash
uv run python - <<'PY'
import torch
print('cuda available:', torch.cuda.is_available())
print('gpu count:', torch.cuda.device_count())
PY
```

## Run HarmBench

Set the repository root for configs that use `${WORKSPACE_ROOT}`:

```bash
export WORKSPACE_ROOT=$PWD
```

Pipeline stages:

| Stage | Purpose |
| --- | --- |
| 1 | Generate HarmBench attack/test-case artifacts. |
| 2 | Extract hidden states for steering/training. |
| 3 | Train or fit steering/baseline artifacts. |
| 4 | Generate completions and compute HarmBench ASR. |

Full run from scratch:

```bash
PYTHONPATH=src uv run python -m barriersteer_pipeline.harmbench.run_harmbench_pipeline \
  --config src/barriersteer_pipeline/harmbench/config/harmbench.yaml \
  --model gemma_2_9b \
  --stages 1,2,3,4
```

If attacks and hidden states already exist, train/evaluate only:

```bash
PYTHONPATH=src uv run python -m barriersteer_pipeline.harmbench.run_harmbench_pipeline \
  --config src/barriersteer_pipeline/harmbench/config/harmbench.yaml \
  --model gemma_2_9b \
  --stages 3,4
```

## Run WildGuard

For WildGuard, we sample train/validation behaviors from the original WildGuard dataset and pair them with DirectRequest test-case files:

```text
HarmBench/data/behavior_datasets/harmbench_behaviors_wildguard_train.csv
HarmBench/data/behavior_datasets/harmbench_behaviors_wildguard_val.csv
HarmBench/results_wildguard/DirectRequest/default/test_cases/test_cases_train.json  # Stage 2 train hidden states
HarmBench/results_wildguard/DirectRequest/default/test_cases/test_cases.json        # Stage 4 validation eval
```

To reuse these files, start from Stage 2:

```bash
PYTHONPATH=src uv run python -m barriersteer_pipeline.harmbench.run_harmbench_pipeline \
  --config src/barriersteer_pipeline/harmbench/config/wildguard.yaml \
  --model qwen_1.5b \
  --stages 2,3,4
```

If hidden states or trained WildGuard steering artifacts already exist, use `--stages 3,4` or `--stages 4` as above.

## Run utility benchmarks

The utility helper runs LM Evaluation Harness tasks such as MMLU and GSM8K.

```bash
bash scripts/eval_lm_harness_batch.sh <model_name> <steer_type> <seed> <k> <batch_size> [kappa]
```

Example base-model run:

```bash
bash scripts/eval_lm_harness_batch.sh gemma_2_9b base_model 42 1.0 64
```

Example BarrierSteer QP run:

```bash
bash scripts/eval_lm_harness_batch.sh gemma_2_9b cbf_qp 42 1.0 64
```

Supported model names:

```text
gemma_2_9b, qwen_1.5b, mistral_8b, llama2_7b
```

Supported steering types:

```text
base_model, sap, cbf_topk, cbf_qp, cbf_merge, actadd, dirablate
```

Outputs are written under:

```text
outputs/lm_harness/<model_name>/
```

Additional utility targets include IFEval, TruthfulQA, MT-Bench, and GPQA-Diamond. TruthfulQA can be run through the LM Harness config by overriding `lm_harness_tasks`; MT-Bench has its own helper.

Example TruthfulQA run:

```bash
PYTHONPATH=src uv run python -m barriersteer_pipeline.evaluation.lm_harness_eval \
  model_path=google/gemma-2-9b-it \
  use_safe_rep_model=false \
  lm_harness_tasks='[truthfulqa_mc2]' \
  num_fewshot=0 \
  batch_size=64 \
  lm_harness_output_path=outputs/lm_harness/gemma_2_9b/gemma_2_9b_truthfulqa_base.json
```

Example MT-Bench run:

```bash
export OPENROUTER_API_KEY=<your_key>
bash scripts/eval_mtbench_batch.sh gemma_2_9b base_model 42 1.0 1
```

## Run adaptive attacks

Adaptive-attack launchers are provided for base models and HarmBench-configured defense modes. They expect the adaptive-attack dependency under `third_party/llm-adaptive-attacks`, a judge API key, and access to gated Hugging Face models.

```bash
export OPENAI_API_KEY=<your_key>
export HF_TOKEN=<your_token>
```

Launch a base-model adaptive attack:

```bash
bash scripts/launch_adaptive_base.sh \
  --models gemma-2-9b \
  --gpus 0 \
  --slots-per-gpu 1
```

Launch a BarrierSteer mode for a fixed seed:

```bash
bash scripts/launch_adaptive_method_seed.sh 43 \
  --models gemma_2_9b \
  --modes cbf_qp_nonlinear \
  --gpus 0
```

Use `--resume-existing` or `--force` to control reruns for method/seed launches.

## Benchmark steering latency

Use `scripts/benchmark_latency.py` to measure BarrierSteer steering latency for a single configuration:

```bash
python scripts/benchmark_latency.py \
  --config src/barriersteer_pipeline/harmbench/config/harmbench.yaml \
  --model qwen_1.5b \
  --mode cbf_topk_nonlinear_s \
  --polytope_model_path outputs/harmbench/qwen_1.5b/cbf_topk_nonlinear_s_phi004/weight_666.pth \
  --output_json outputs/benchmark_latency/qwen_1.5b/single.json
```

To run the full ablation sweep (network size and `num_phi`) across GPUs:

```bash
bash scripts/benchmark_latency_ablation.sh
```

To choose specific GPUs, set `BENCH_GPUS`:

```bash
BENCH_GPUS=0,1 bash scripts/benchmark_latency_ablation.sh
```

## Run over-refusal with XSTest

XSTest safe-only is the over-refusal setting used in the paper. For OpenRouter judging, set `OPENROUTER_API_KEY` first:

```bash
export OPENROUTER_API_KEY=<your_key>
```

Run one XSTest over-refusal job:

```bash
bash scripts/eval_xstest_batch.sh <model_name> <steer_type> <seed> <k> <batch_size> [kappa]
```

Example base-model over-refusal run:

```bash
bash scripts/eval_xstest_batch.sh gemma_2_9b base_model 42 1.0 8
```

Example BarrierSteer LSE / merge over-refusal run:

```bash
bash scripts/eval_xstest_batch.sh gemma_2_9b cbf_merge 42 1.0 8 10.0
```

Outputs are written under:

```text
outputs/xstest/<model_name>/
```

Each run writes a completions CSV and a summary JSON containing `over_refusal_rate`.

## Citation

If you find BarrierSteer useful please cite:

```bibtex
@misc{tran2026barriersteerllmsafetylearning,
      title={BarrierSteer: LLM Safety via Learning Barrier Steering},
      author={Thanh Q. Tran and Arun Verma and Kiwan Wong and Bryan Kian Hsiang Low and Daniela Rus and Wei Xiao},
      year={2026},
      eprint={2602.20102},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.20102},
}
```
