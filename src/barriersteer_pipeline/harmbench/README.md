# HarmBench Pipeline

Minimal instructions for running the BarrierSteer HarmBench experiments pipeline.

## Prerequisites

1. **Download and Install HarmBench**
   ```bash
   cd HarmBench
   pip install -r requirements.txt
   ```
   This repository contains the code and scripts used for generating the paper's experiments.

2. **Configure HarmBench Path**
   Update `harmbench_path` in `config/pipeline_config.yaml` to point to your HarmBench installation.

## Pipeline Stages

The pipeline consists of 4 sequential stages:

### Stage 1: Attack Generation
Generates adversarial test cases using HarmBench's attack methods.
```bash
python src/barriersteer_pipeline/harmbench/run_harmbench_pipeline.py \
    --config src/barriersteer_pipeline/harmbench/config/pipeline_config.yaml \
    --model llama2_7b \
    --stages 1
```

### Stage 2: Hidden State Extraction & Processing
Extracts and processes hidden states from the model on generated attacks.
```bash
python src/barriersteer_pipeline/harmbench/run_harmbench_pipeline.py \
    --config src/barriersteer_pipeline/harmbench/config/pipeline_config.yaml \
    --model llama2_7b \
    --stages 2
```

### Stage 3: Polytope Training
Trains safety polytope constraints from extracted hidden states.
```bash
python src/barriersteer_pipeline/harmbench/run_harmbench_pipeline.py \
    --config src/barriersteer_pipeline/harmbench/config/pipeline_config.yaml \
    --model llama2_7b \
    --stages 3
```

### Stage 4: Steering Evaluation
Evaluates polytope effectiveness for safety steering.
```bash
python src/barriersteer_pipeline/harmbench/run_harmbench_pipeline.py \
    --config src/barriersteer_pipeline/harmbench/config/pipeline_config.yaml \
    --model llama2_7b \
    --stages 4
```

### New Baselines

`CAST` is available through `polytope_training.modes.<mode>.model_type=conditional_steering_vector`. It trains the usual neural CBF/polytope unsafe detector, then computes a steering vector from the same hidden-state dataset and applies that vector only when the detector marks a token state unsafe at inference time.

`CircuitBreaker` is available through `circuit_breaker_training` plus a `steering.modes.<mode>.circuit_breaker.use=true` evaluation mode. Stage 3 builds HarmBench prompt/completion pairs from labeled completions and trains a LoRA circuit-breaker checkpoint; Stage 4 can then evaluate that checkpoint directly by overriding HarmBench's `model_name_or_path`.

`ReFT-r1` is available as a rank-1 activation intervention baseline. Stage 3 can build a HarmBench-derived prompt/target dataset from labeled completions and train one vector at `target_layer`; Stage 4 loads `reft_r1_vector.pt`, disables vLLM, and runs normal HF generation with a forward hook that applies `h <- h + beta * mean(topk(ReLU(h @ w))) * w`.

Example config sketch:
```yaml
models:
  - name: "llama2_7b"
    path: "meta-llama/Llama-2-7b-chat-hf"
    steering:
      modes:
        cast_actadd:
          base_exp_ident_suffix: "cast_actadd"
          num_phi: [4]
          cbf:
            use: false
        circuit_breaker:
          base_exp_ident_suffix: "circuit_breaker"
          num_phi: [0]
          circuit_breaker:
            use: true
            training_mode: "circuit_breaker"
        reft_r1:
          base_exp_ident_suffix: "reft_r1"
          num_phi: [0]
          reft_r1:
            use: true
            training_mode: "reft_r1"
    polytope_training:
      modes:
        cast_actadd:
          model_type: "conditional_steering_vector"
          base_exp_ident_suffix: "cast_actadd"
          num_phi: [4]
          use_neural_phi: true
          steering_method: "actadd"
          steering_alpha: 1.0
        circuit_breaker:
          model_type: "circuit_breaker"
          base_exp_ident_suffix: "circuit_breaker"
          base_output_dir: "${WORKSPACE_ROOT}/outputs/harmbench_cb/llama2_7b"
          target_layers: "20"
          transform_layers: "-1"
          max_steps: 150
        reft_r1:
          model_type: "reft_r1"
          base_exp_ident_suffix: "reft_r1"
          base_output_dir: "${WORKSPACE_ROOT}/outputs/harmbench_reft_r1/llama2_7b"
          behavior_dataset: "data/behavior_datasets/harmbench_behaviors_text_val.csv"
          target_layer: 20
          top_k: 5
          beta: 1.0
          max_steps: 500
```

**Run outputs (easy-to-browse):** after Stage 4 completes, the pipeline will create a per-run folder under
`outputs/harmbench/<YYYY-MM-DD>/harmbench_eval_<model>_<exp_ident>/` containing **only**:
- `*/merged_completions.json`
- `*/merged_timing.json`
- `*/results.json`

**Multi-GPU tip (single machine):** set `steering_evaluation.execution_mode: "local_parallel"` in
`src/barriersteer_pipeline/harmbench/config/pipeline_config.yaml`. This will shard evaluation by parts across GPUs
using `CUDA_VISIBLE_DEVICES` (one generation process per GPU).

### Stage 5: MMLU Evaluation (Optional)
Evaluate model capability preservation using MMLU (Massive Multitask Language Understanding) benchmark.
This helps assess whether polytope steering maintains model performance on general tasks.

**Prerequisites:**
1. Download MMLU data:
   ```bash
   # Download MMLU dataset to a directory (e.g., ./data/mmlu)
   # The dataset should have dev/ and test/ subdirectories with CSV files
   ```

2. Configure evaluation in `exp_configs/eval_config.yaml`:
   ```yaml
   model_path: "meta-llama/Llama-2-7b-chat-hf"  # Your model path
   use_safe_rep_model: true  # Use SafeRepModel with polytope steering
   polytope_weight_path: "/path/to/weights.pth"  # Path to trained polytope weights
   data_dir: "./data/mmlu"  # Path to MMLU dataset
   steer_layer: 20  # Same layer as used in training
   ```

**Run MMLU Evaluation:**
```bash
cd /path/to/SafeLLM
python src/barriersteer_pipeline/evaluation/mmlu.py \
    model_path=meta-llama/Llama-2-7b-chat-hf \
    use_safe_rep_model=true \
    polytope_weight_path=outputs/harmbench/llama2_7b/cbf_topk_linear/weights.pth \
    data_dir=./data/mmlu \
    steer_layer=20
```

**Output:** Results are saved in `outputs/mmlu/<date>/eval-<timestamp>/results_<model_name>/` with per-subject CSV files and overall accuracy logged.

**Note:** MMLU evaluation is separate from the HarmBench pipeline stages. Run it after Stage 4 to evaluate capability preservation for your trained polytope models.

### Stage 6: LM Evaluation Harness
```bash
python -m barriersteer_pipeline.evaluation.lm_harness_eval \
  model_path="meta-llama/Llama-2-7b-chat-hf" \
  use_safe_rep_model=false \
  lm_harness_tasks=[hellaswag,arc_easy,arc_challenge,winogrande,mmlu,gsm8k,truthfulqa_mc2,wikitext] \
  num_fewshot=0 \
  batch_size=64 \
  lm_harness_output_path="outputs/lm_harness/llama2_7b_base.json"
```

### Running Multiple Stages
Run stages sequentially:
```bash
python src/barriersteer_pipeline/harmbench/run_harmbench_pipeline.py \
    --config src/barriersteer_pipeline/harmbench/config/pipeline_config.yaml \
    --model qwen_1.5b \
    --stages 1 2 3 4
```

## Available Models
- `qwen_1.5b` - Qwen2-1.5B-Instruct
- `llama2_7b` - Llama-2-7b
- `mistral_8b` - Ministral-8B-Instruct-2410

## Important Caveats

**Current Experimental Setting:** The hyperparameters used in these experiments are optimized for **inhibiting harmful outputs**, not for generating cohesive answers. This may result in:
- Some steering results outputting nonsensical sentences
- Over-conservative safety responses

**Improving Output Quality:** Standard approaches to improve sentence quality include:
1. Training the polytope with both adversarial datasets AND standard language generation datasets
2. Using the polytope as a rejection classifier rather than for direct steering
3. Adjusting the `margin` and `unsafe_weight` parameters in the config for your specific use case
4. Reducing the `lambda_weight` and `safe_violation_weight` in Harmbench's `models.yaml` configuration

## Configuration
Edit `config/pipeline_config.yaml` to:
- Adjust model-specific hyperparameters
- Change Slurm settings for cluster execution
- Modify attack methods per model
- Configure output directories
