"""
Adaptive Attack Evaluation for BarrierSteer.

Runs the random-search adaptive jailbreak from llm-adaptive-attacks
against a steered (or base) model and reports Attack Success Rate (ASR).

Usage (standalone):
    python -m barriersteer_pipeline.evaluation.adaptive_attack_eval \
        model_path=google/gemma-2-9b-it \
        use_safe_rep_model=true \
        polytope_weight_path=outputs/harmbench/gemma_2_9b/cbf_topk_nonlinear_phi004/weight_44.pth \
        n_behaviors=5

Usage (via wrapper):
    cd third_party/llm-adaptive-attacks
    python run_adaptive_attack.py --target-model safe_gemma_2_9b --n-behaviors 5 --debug
"""

import csv
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger("polytope")

# Resolve the root of SafeLLM and the llm-adaptive-attacks directory
_SAFELLM_ROOT = (
    Path(__file__).resolve().parents[3]
)  # src/barriersteer_pipeline/evaluation -> SafeLLM/
_ADAPTIVE_ATTACKS_DIR = _SAFELLM_ROOT / "third_party" / "llm-adaptive-attacks"
_BEHAVIORS_CSV = (
    _ADAPTIVE_ATTACKS_DIR / "harmful_behaviors" / "harmful_behaviors_pair.csv"
)


def _check_prerequisites():
    """Verify that the adaptive attacks directory exists at the expected location."""
    if not _ADAPTIVE_ATTACKS_DIR.exists():
        raise FileNotFoundError(
            f"llm-adaptive-attacks not found at {_ADAPTIVE_ATTACKS_DIR}. "
            f"Expected the internalized copy under third_party/llm-adaptive-attacks"
        )
    if not (_ADAPTIVE_ATTACKS_DIR / "main.py").exists():
        raise FileNotFoundError(
            f"main.py not found in {_ADAPTIVE_ATTACKS_DIR}. "
            f"The adaptive attacks directory may be incomplete."
        )


def load_behaviors(
    csv_path: str = None,
    n_behaviors: Optional[int] = None,
    start_idx: int = 0,
) -> List[Dict]:
    """Load (goal, target) pairs from the AdvBench behaviors CSV."""
    csv_path = csv_path or str(_BEHAVIORS_CSV)
    behaviors = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            behaviors.append(
                {
                    "index": int(row[""]),
                    "goal": row["goal"],
                    "target": row["target"],
                    "category": row["category"],
                }
            )
    behaviors = behaviors[start_idx:]
    if n_behaviors is not None:
        behaviors = behaviors[:n_behaviors]
    return behaviors


def _build_model_arg(cfg: DictConfig) -> str:
    """
    Determine the --target-model argument for main.py from the eval config.

    If ``harmbench_model_name`` is set (e.g. ``safe_gemma_2_9b``), use it
    directly — the adaptive-attack conversers.py will load the model via
    HarmBench's models.yaml.

    Otherwise fall back to a simple HF model name.
    """
    if cfg.get("harmbench_model_name"):
        return str(cfg.harmbench_model_name)
    # Fallback: derive from model_path
    model_path = str(cfg.model_path)
    if "gemma-2-9b" in model_path.lower():
        return "gemma-2-9b"
    return model_path


def run_attack(
    cfg: DictConfig,
    behaviors: List[Dict],
    output_dir: str,
) -> List[Dict]:
    """
    Launch the adaptive attack for each behavior by calling main.py as
    a subprocess.  Returns a list of per-behavior result dicts.
    """
    _check_prerequisites()
    target_model = _build_model_arg(cfg)
    judge_model = str(cfg.get("judge_model", "gpt-4o-mini"))
    prompt_template = str(cfg.get("prompt_template", "refined_best"))
    n_iterations = int(cfg.get("n_iterations", 5000))
    n_tokens_adv = int(cfg.get("n_tokens_adv", 25))
    n_tokens_change_max = int(cfg.get("n_tokens_change_max", 4))
    target_max_n_tokens = int(cfg.get("target_max_n_tokens", 150))
    seed = int(cfg.get("seed", 1))
    debug = bool(cfg.get("debug", False))

    results = []
    for i, behavior in enumerate(behaviors):
        log.info(
            "[%d/%d] Attacking behavior #%d: %s",
            i + 1,
            len(behaviors),
            behavior["index"],
            behavior["goal"][:80],
        )

        cmd = [
            sys.executable,
            str(_ADAPTIVE_ATTACKS_DIR / "main.py"),
            "--target-model",
            target_model,
            "--judge-model",
            judge_model,
            "--goal",
            behavior["goal"],
            "--goal-modified",
            behavior["goal"],
            "--target-str",
            behavior["target"],
            "--prompt-template",
            prompt_template,
            "--n-iterations",
            str(n_iterations),
            "--n-tokens-adv",
            str(n_tokens_adv),
            "--n-tokens-change-max",
            str(n_tokens_change_max),
            "--target-max-n-tokens",
            str(target_max_n_tokens),
            "--seed",
            str(seed),
            "--index",
            str(behavior["index"]),
            "--category",
            behavior["category"].split(",")[0].strip(),
            "--schedule_n_to_change",
            "--determinstic-jailbreak",
        ]
        if debug:
            cmd.append("--debug")

        t0 = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_ADAPTIVE_ATTACKS_DIR),
        )
        elapsed = time.time() - t0

        # Parse stdout for the last logprob / jailbreak status
        stdout_tail = ""
        if proc.stdout:
            lines = proc.stdout.strip().split("\n")
            stdout_tail = "\n".join(lines[-20:])

        result = {
            "index": behavior["index"],
            "goal": behavior["goal"],
            "category": behavior["category"],
            "returncode": proc.returncode,
            "elapsed_s": round(elapsed, 2),
            "stdout_tail": stdout_tail,
        }
        results.append(result)

        if proc.returncode != 0 and proc.stderr:
            log.warning(
                "stderr (last 5 lines):\n%s",
                "\n".join(proc.stderr.strip().split("\n")[-5:]),
            )

        # Incremental save
        _save_results(results, cfg, output_dir)

    return results


def _save_results(results: List[Dict], cfg: DictConfig, output_dir: str):
    """Write results JSON incrementally."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "adaptive_attack_results.json")
    payload = {
        "config": {
            "target_model": _build_model_arg(cfg),
            "judge_model": str(cfg.get("judge_model", "gpt-4o-mini")),
            "n_iterations": int(cfg.get("n_iterations", 5000)),
            "seed": int(cfg.get("seed", 1)),
        },
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved results to %s", path)


def summarize_results(results: List[Dict]) -> Dict:
    """Compute summary metrics from attack results."""
    total = len(results)
    success = sum(1 for r in results if r["returncode"] == 0)
    failed = total - success
    return {
        "total_behaviors": total,
        "successful_runs": success,
        "failed_runs": failed,
    }


@hydra.main(
    config_path=f"{os.getcwd()}/exp_configs",
    config_name="eval_adaptive_attack_config",
    version_base="1.1",
)
def main(cfg: DictConfig):
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    n_behaviors = cfg.get("n_behaviors")
    n_behaviors = int(n_behaviors) if n_behaviors is not None else None
    start_idx = int(cfg.get("start_idx", 0))
    behaviors_csv = cfg.get("behaviors_csv")

    behaviors = load_behaviors(
        csv_path=str(behaviors_csv) if behaviors_csv else None,
        n_behaviors=n_behaviors,
        start_idx=start_idx,
    )
    log.info("Loaded %d behaviors", len(behaviors))

    target_model = _build_model_arg(cfg)
    output_dir = str(
        cfg.get(
            "output_dir",
            f"outputs/adaptive_attack/{target_model}",
        )
    )

    results = run_attack(cfg, behaviors, output_dir)
    summary = summarize_results(results)

    summary["target_model"] = target_model
    summary["output_dir"] = output_dir
    log.info("Summary: %s", summary)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()
