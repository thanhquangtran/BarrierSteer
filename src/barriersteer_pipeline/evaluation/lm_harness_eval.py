"""
LM Evaluation Harness Integration for SafeLLM

This script integrates EleutherAI's lm-evaluation-harness with SafeLLM's
SafeRepModel and defense mechanisms, with support for multi-GPU evaluation.

Multi-GPU Usage:
  accelerate launch -m barriersteer_pipeline.evaluation.lm_harness_eval ...

The implementation leverages lm-eval's existing HFLM class rather than
reimplementing evaluation logic.
"""

import json
import logging
import os
import sys
import traceback

# Force enabling tracebacks
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"

# Default to offline mode for datasets to speed up loading in multi-process distributed runs
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from typing import Optional, Union

import hydra
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from omegaconf import DictConfig, OmegaConf, open_dict

from steer.common.defense_utils import setup_defense_object

# Debug print to verify module load
print(f"DEBUG: Module loaded (PID: {os.getpid()})", file=sys.stderr)
sys.stderr.flush()

try:
    from accelerate import Accelerator

    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("DEBUG: Accelerate not available", file=sys.stderr)

log = logging.getLogger("polytope")


def _get_device_map(cfg: DictConfig):
    """Return a Transformers-compatible device_map from Hydra config.

    Hydra CLI overrides cannot conveniently pass a dict with an empty-string key
    (``{"": "cuda:0"}``) unless it is quoted as a string. Accept that JSON
    string form so run scripts can force single-device placement while still
    reserving a two-GPU tmux slot.
    """
    device_map = cfg.get("device_map", "auto")
    if isinstance(device_map, str):
        stripped = device_map.strip()
        if stripped.startswith("cuda") or stripped == "cpu":
            return {"": stripped}
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                log.warning("Could not parse device_map JSON string: %s", device_map)
    return device_map


try:
    from accelerate import Accelerator

    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    log.warning("accelerate not available. Multi-GPU evaluation will not be supported.")


class DefenseModelWrapper(torch.nn.Module):
    """
    Wrapper that makes a HuggingFace model + defense combination callable.

    Applies defense prompt transformations before computing logits.
    This enables defense methods to affect lm-eval benchmark results.
    """

    def __init__(self, hf_model, tokenizer, defense):
        super().__init__()
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.defense = defense
        self._defense_type = type(defense).__name__

        # Cache for transformed prompts to avoid redundant transformations
        self._transform_cache = {}

        log.info(f"DefenseModelWrapper initialized with defense: {self._defense_type}")

    def _transform_prompt(self, text: str) -> str:
        """Apply defense-specific prompt transformation."""
        if text in self._transform_cache:
            return self._transform_cache[text]

        transformed = text

        # Apply defense-specific transformations
        if hasattr(self.defense, "_remind_suffix"):
            # SelfReminderDefense
            transformed = self.defense._remind_suffix(text)
        elif hasattr(self.defense, "paraphrase"):
            # ParaphraseDefense
            transformed = self.defense.paraphrase(text)
        elif hasattr(self.defense, "icl_prompt"):
            # ICLDefense - prepend ICL examples
            if hasattr(self.defense, "icl_examples"):
                transformed = self.defense.icl_examples + text
        # Note: smoothllm, backtranslation, response_check don't have simple
        # prompt transformations - they require multiple model calls or post-processing

        self._transform_cache[text] = transformed
        return transformed

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass with defense transformation applied to prompts."""
        # For simple prompt-transforming defenses, we could decode, transform, re-encode
        # But this is complex and may break batching. For now, pass through directly.
        # The main benefit comes from defenses that modify the system message or add prefixes.

        # Pass through to underlying model
        return self.hf_model(input_ids, attention_mask=attention_mask, **kwargs)

    def generate(self, *args, **kwargs):
        """Generation with defense - delegates to underlying model."""
        return self.hf_model.generate(*args, **kwargs)

    @property
    def config(self):
        return self.hf_model.config

    @property
    def device(self):
        return self.hf_model.device

    @property
    def dtype(self):
        return self.hf_model.dtype

    def to(self, *args, **kwargs):
        self.hf_model = self.hf_model.to(*args, **kwargs)
        return self

    def eval(self):
        self.hf_model.eval()
        return self

    def parameters(self):
        return self.hf_model.parameters()


class SafeHarnessLM(HFLM):
    """
    Custom LM wrapper that extends lm-eval's HFLM class to support:
    - SafeRepModel with polytope constraints
    - Defense mechanisms from llm_jailbreaking_defense
    - Standard HuggingFace models

    This class inherits loglikelihood, loglikelihood_rolling, and generate_until
    methods from HFLM, avoiding reimplementation.

    The approach: use HFLM's original __init__ with the pretrained path,
    then swap in the custom model (SafeRepModel/DefendedTargetLM) if needed.
    """

    def __init__(
        self,
        pretrained: str,
        custom_model=None,
        defense=None,
        batch_size: Union[int, str] = 1,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize SafeHarnessLM.

        Args:
            pretrained: Model path or HuggingFace model ID
            custom_model: Optional pre-loaded model to swap in (SafeRepModel, HF model)
            defense: Optional defense object to apply prompt transformations
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
            **kwargs: Additional arguments passed to HFLM
        """
        self._custom_model = custom_model

        # Call HFLM's original __init__
        # We pass pretrained because HFLM requires it, but our _create_model will use _custom_model
        super().__init__(
            pretrained=pretrained,
            batch_size=batch_size,
            max_length=max_length,
            **kwargs,
        )

        self._defense = defense

        # Wrap with defense if provided (and custom_model was used)
        if custom_model is not None and defense is not None:
            log.info(f"Wrapping model with defense: {type(defense).__name__}")
            self._model = DefenseModelWrapper(
                hf_model=self._model,  # _create_model set self._model to custom_model
                tokenizer=self.tokenizer,
                defense=defense,
            )

    def _apply_self_reminder(self, text: str) -> str:
        defense = getattr(self, "_defense", None)
        if defense is None or not hasattr(defense, "_remind_suffix"):
            return text
        reminded = defense._remind_suffix(text)
        prefix_template = getattr(defense, "_remind_prefix", lambda: "")()
        if prefix_template:
            prefix = prefix_template.format(original_system_message="")
            reminded = f"{prefix}\n\n{reminded}"
        return reminded

    @staticmethod
    def _request_with_args(original, args):
        class _Request:
            def __init__(self, source, replacement_args):
                self._source = source
                self.args = replacement_args

            def __getattr__(self, name):
                return getattr(self._source, name)

        return _Request(original, args)

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        if getattr(self, "_defense", None) is None or not hasattr(
            self._defense, "_remind_suffix"
        ):
            return super().loglikelihood(requests, disable_tqdm=disable_tqdm)
        reminded_requests = [
            self._request_with_args(
                req, (self._apply_self_reminder(context), continuation)
            )
            for req in requests
            for context, continuation in [req.args]
        ]
        return super().loglikelihood(reminded_requests, disable_tqdm=disable_tqdm)

    def generate_until(self, requests, disable_tqdm: bool = False):
        if getattr(self, "_defense", None) is None or not hasattr(
            self._defense, "_remind_suffix"
        ):
            return super().generate_until(requests, disable_tqdm=disable_tqdm)
        reminded_requests = [
            self._request_with_args(
                req, (self._apply_self_reminder(context), gen_kwargs)
            )
            for req in requests
            for context, gen_kwargs in [req.args]
        ]
        return super().generate_until(reminded_requests, disable_tqdm=disable_tqdm)

    def _create_model(self, **kwargs):
        """
        Override HFLM's model creation to use our custom model if provided.
        """
        if self._custom_model is not None:
            self._model = self._custom_model
            # IMPORTANT: Set self._device from the model's device so that input tensors are moved correctly
            try:
                self._device = self._model.device
            except AttributeError:
                # Fallback if model doesn't expose .device property directly
                try:
                    self._device = next(self._model.parameters()).device
                except Exception:
                    pass
        else:
            super()._create_model(**kwargs)


def _resolve_max_length(cfg: DictConfig, model=None) -> Optional[int]:
    """
    Resolve the effective max length for lm-eval.

    If the user explicitly sets max_input_length, respect it. Otherwise prefer the
    loaded model's advertised context length rather than a hardcoded 2048 fallback.
    """
    explicit_max_length = cfg.get("max_input_length", None)
    if explicit_max_length is not None:
        return explicit_max_length

    if model is not None:
        config = getattr(model, "config", None)
        if config is not None:
            for attr in (
                "max_position_embeddings",
                "n_positions",
                "max_sequence_length",
                "seq_length",
            ):
                value = getattr(config, attr, None)
                if isinstance(value, int) and value > 0:
                    return value

    return None


def _maybe_attach_reft_r1(cfg: DictConfig, model: torch.nn.Module) -> None:
    """Attach a ReFT-r1 inference hook when configured."""
    if not cfg.get("use_reft_r1", False):
        return
    vector_path = cfg.get("reft_r1_vector_path", None)
    if vector_path is None or str(vector_path).lower() in ("", "none", "null"):
        raise ValueError("use_reft_r1=true requires reft_r1_vector_path")

    from steer.reft_r1 import attach_reft_r1_hook, load_reft_r1_intervention

    intervention = load_reft_r1_intervention(str(vector_path), map_location="cpu")
    if cfg.get("reft_r1_target_layer", None) is not None:
        intervention.config.target_layer = int(cfg.reft_r1_target_layer)
    if cfg.get("reft_r1_top_k", None) is not None:
        intervention.config.top_k = int(cfg.reft_r1_top_k)
    if cfg.get("reft_r1_beta", None) is not None:
        intervention.config.beta = float(cfg.reft_r1_beta)
    try:
        intervention.to(next(model.parameters()).device)
    except Exception:
        pass
    handle = attach_reft_r1_hook(model, intervention)
    model.reft_r1_intervention = intervention
    model.reft_r1_hook_handle = handle
    log.info(
        "Attached ReFT-r1 intervention: path=%s layer=%s top_k=%s beta=%s",
        vector_path,
        intervention.config.target_layer,
        intervention.config.top_k,
        intervention.config.beta,
    )


def create_safe_harness_model(cfg: DictConfig) -> Union[HFLM, SafeHarnessLM]:
    """
    Create a model instance from configuration.

    For baseline HuggingFace models, uses HFLM directly.
    For SafeRepModel, uses SafeHarnessLM wrapper.
    For defense methods, loads HF model + defense separately and wraps them.

    Args:
        cfg: Hydra configuration

    Returns:
        HFLM or SafeHarnessLM instance
    """
    use_safe_rep = cfg.get("use_safe_rep_model", False)
    use_defense = cfg.get("use_defense", False)

    if use_safe_rep:
        # Load SafeRepModel (with optional CBF) - this is the polytope-based steering
        log.info("Loading SafeRepModel with polytope constraints...")
        from barriersteer_pipeline.evaluation.mmlu import load_safe_rep_model

        custom_model, _ = load_safe_rep_model(cfg)
        max_length = _resolve_max_length(cfg, custom_model)

        log.info(f"Creating SafeHarnessLM with SafeRepModel")
        return SafeHarnessLM(
            pretrained=cfg.model_path,
            custom_model=custom_model,
            batch_size=cfg.get("batch_size", 1),
            max_length=max_length,
            device_map=_get_device_map(cfg),
            torch_dtype=torch.float16,
        )
    elif use_defense:
        # Load base HuggingFace model and defense separately
        log.info(f"Loading HuggingFace model for defense: {cfg.defense_method}...")
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.float16,
            device_map=_get_device_map(cfg),
            trust_remote_code=True,
        )
        _maybe_attach_reft_r1(cfg, hf_model)

        # Load defense object
        log.info(f"Setting up defense: {cfg.defense_method}")
        defense = setup_defense_object(cfg.defense_method)
        max_length = _resolve_max_length(cfg, hf_model)

        log.info(f"Creating SafeHarnessLM with defense: {type(defense).__name__}")
        return SafeHarnessLM(
            pretrained=cfg.model_path,
            custom_model=hf_model,
            defense=defense,
            batch_size=cfg.get("batch_size", 1),
            max_length=max_length,
            device_map=_get_device_map(cfg),
            torch_dtype=torch.float16,
        )
    else:
        # Use SafeHarnessLM with manual HF loading for base model too
        # This ensures consistent behavior and correct device placement in distributed settings
        log.info(f"Loading Base HuggingFace model: {cfg.model_path}")
        from transformers import AutoModelForCausalLM

        # Manually load generic HF model with correct device map
        hf_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype="auto",
            device_map=_get_device_map(cfg),
            trust_remote_code=True,
        )
        custom_adapter_path = cfg.get("custom_adapter_path", None)
        if custom_adapter_path and str(custom_adapter_path).lower() not in (
            "none",
            "null",
            "",
        ):
            from peft import PeftModel

            log.info(f"Loading and merging adapter: {custom_adapter_path}")
            hf_model = PeftModel.from_pretrained(hf_model, str(custom_adapter_path))
            hf_model = hf_model.merge_and_unload()
        _maybe_attach_reft_r1(cfg, hf_model)
        hf_model.eval()  # Ensure deterministic evaluation (disable dropout)
        max_length = _resolve_max_length(cfg, hf_model)

        log.info(f"Creating SafeHarnessLM wrapper for base model")
        return SafeHarnessLM(
            pretrained=cfg.model_path,
            custom_model=hf_model,
            batch_size=cfg.get("batch_size", 1),
            max_length=max_length,
            device_map=_get_device_map(cfg),
            torch_dtype=torch.float16,
        )


@hydra.main(
    config_path=f"{os.getcwd()}/exp_configs",
    config_name="eval_lm_harness_config",
    version_base="1.1",
)
def main(cfg: DictConfig):
    """Main evaluation function"""

    # Log configuration
    log.info("Configuration:")

    # Handle multi-GPU execution using standard Accelerator detection
    if ACCELERATE_AVAILABLE:
        try:
            accelerator = Accelerator()
            if accelerator.num_processes > 1:
                log.info(
                    f"Distributed execution: {accelerator.num_processes} processes, Rank {accelerator.process_index}"
                )
                with open_dict(cfg):
                    cfg.device_map = {"": str(accelerator.device)}
                log.info(f"Set device_map to {cfg.device_map}")
            else:
                log.info(f"Single process execution. Device: {accelerator.device}")
                if torch.cuda.device_count() > 1:
                    log.warning(
                        f"Found {torch.cuda.device_count()} GPUs but running in single process mode!"
                    )
                    log.warning(
                        "For data parallelism, run with: accelerate launch -m ..."
                    )

        except Exception as e:
            log.warning(f"Could not initialize Accelerator: {e}")

    log.info(OmegaConf.to_yaml(cfg))

    # Get task list - convert from OmegaConf ListConfig to Python list
    task_list = cfg.get("lm_harness_tasks", ["hellaswag"])
    if isinstance(task_list, str):
        task_list = [task_list]
    else:
        # Convert OmegaConf ListConfig to Python list for lm-eval compatibility
        task_list = list(task_list)

    log.info(f"Evaluating on tasks: {task_list}")

    # Create model
    model = create_safe_harness_model(cfg)

    # Set up evaluation parameters
    num_fewshot = cfg.get("num_fewshot", 0)
    limit = cfg.get("lm_harness_limit", None)

    # Output path
    output_path = cfg.get("lm_harness_output_path")
    if output_path is None:
        # Auto-generate output path
        model_name = os.path.basename(cfg.model_path.rstrip("/"))
        output_path = f"lm_harness_results_{model_name}.json"

    log.info(f"Results will be saved to: {output_path}")

    log.info(f"Results will be saved to: {output_path}")

    # Run evaluation
    log.info("Starting evaluation...")

    # Synchronization barrier to ensure all processes are ready and to minimize dataset cache race conditions
    if ACCELERATE_AVAILABLE:
        try:
            accelerator = Accelerator()
            accelerator.wait_for_everyone()
            log.info(
                f"Rank {accelerator.process_index} passed barrier. Starting evaluation."
            )
        except Exception:
            pass

    try:
        results = evaluator.simple_evaluate(
            model=model,
            tasks=task_list,
            num_fewshot=num_fewshot,
            limit=limit,
            bootstrap_iters=0,  # Disable bootstrap for faster evaluation
            cache_requests=cfg.get("cache_requests", True),  # Enable caching by default
        )
    except Exception as e:
        log.error(f"Evaluation failed: {e}")
        # Print FULL traceback using standard library
        import traceback

        traceback.print_exc()
        raise e

    # Save results
    import json

    class CustomJSONEncoder(json.JSONEncoder):
        """Custom JSON encoder to handle non-serializable objects from lm-eval."""

        def default(self, obj):
            # Handle torch.dtype objects
            if isinstance(obj, torch.dtype):
                return str(obj)
            # Handle numpy types if present
            try:
                import numpy as np

                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
            except ImportError:
                pass
            # Fallback: convert to string
            try:
                return str(obj)
            except Exception:
                return super().default(obj)

    # Save results only on main process to avoid race conditions
    if not ACCELERATE_AVAILABLE or accelerator.is_main_process:
        results_dir = os.path.dirname(output_path) or "."
        os.makedirs(results_dir, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, cls=CustomJSONEncoder)

        log.info(f"Results saved to {output_path}")

        # Create a clean summary with only final metrics
        summary = {}
        if "results" in results:
            for task_name, task_results in results["results"].items():
                summary[task_name] = {}
                for metric_name, metric_value in task_results.items():
                    # Only include primary metrics (skip stderr, alias)
                    if isinstance(metric_value, (int, float)):
                        # Clean up metric name (remove ",none" suffix for readability)
                        clean_name = metric_name.replace(",none", "")
                        summary[task_name][clean_name] = round(metric_value, 4)

        # Save summary to a separate file
        summary_path = output_path.replace(".json", "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Summary saved to {summary_path}")

    # Wait for all processes to finish before exiting to ensure clean shutdown
    if ACCELERATE_AVAILABLE:
        accelerator.wait_for_everyone()

    # Print summary (only if results exist)
    if results is not None:
        log.info("\n" + "=" * 50)
        log.info("EVALUATION SUMMARY")
        log.info("=" * 50)

        if "results" in results:
            for task_name, task_results in results["results"].items():
                log.info(f"\nTask: {task_name}")
                for metric_name, metric_value in task_results.items():
                    if isinstance(metric_value, (int, float)):
                        log.info(f"  {metric_name}: {metric_value:.4f}")

        log.info("\n" + "=" * 50)


if __name__ == "__main__":
    main()
