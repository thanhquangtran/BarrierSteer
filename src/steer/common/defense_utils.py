import logging
import os
from typing import Any, Dict, Optional

import torch
from llm_jailbreaking_defense import (
    BacktranslationConfig,
    DefendedTargetLM,
    ICLDefenseConfig,
    ParaphraseDefenseConfig,
    ResponseCheckConfig,
    SelfReminderConfig,
    SemanticSmoothConfig,
    SmoothLLMConfig,
    TargetLM,
    load_defense,
)
from llm_jailbreaking_defense.defenses import (
    BackTranslationDefense,
    ICLDefense,
    ParaphraseDefense,
    ResponseCheckDefense,
    SelfReminderDefense,
    SemanticSmoothDefense,
    SmoothLLMDefense,
)

log = logging.getLogger("polytope")


class BatchedSelfReminderDefendedTargetLM:
    """Batch-preserving wrapper for Self-Reminder.

    The upstream DefendedTargetLM applies defenses via a Python list
    comprehension. That is fine for defenses that need per-sample extra model
    calls, but Self-Reminder is only a deterministic prompt/system-message
    transform before one target-model generation. This wrapper applies the same
    transform to the whole prompt list and then calls TargetLM.get_response once,
    preserving TargetLM's internal batched generation.
    """

    def __init__(self, target_model, defense):
        self.target_model = target_model
        self.defense = defense

    def get_response(self, prompts_list, responses_list=None, verbose=False):
        only_one_prompt = isinstance(prompts_list, str)
        if only_one_prompt:
            prompts_list = [prompts_list]

        if responses_list is not None:
            raise ValueError("Self-Reminder does not consume precomputed responses")

        reminded_prompts = [
            self.defense._remind_suffix(prompt) for prompt in prompts_list
        ]
        system_message_template = self.defense._remind_prefix()

        import copy

        template = copy.deepcopy(self.target_model.template)
        if system_message_template is not None:
            template.system_message = system_message_template.format(
                original_system_message=template.system_message
            )

        responses = self.target_model.get_response(
            reminded_prompts, template=template, verbose=verbose
        )
        if only_one_prompt:
            return responses[0]
        return responses

    def evaluate_log_likelihood(self, prompt, response):
        return self.target_model.evaluate_log_likelihood(prompt, response)


def setup_defense(model, defense_method):
    """Setup defense wrapper for the model.

    Note: We instantiate defense classes directly rather than using load_defense()
    because some defense classes (e.g., ResponseCheckDefense) have different
    __init__ signatures that don't match what load_defense expects.
    """
    if defense_method == "backtranslation":
        config = BacktranslationConfig()
        defense = BackTranslationDefense(config, preloaded_model=None)
    elif defense_method == "semantic_smoothing":
        config = SemanticSmoothConfig()
        defense = SemanticSmoothDefense(config, preloaded_model=None)
    elif defense_method == "response_check":
        config = ResponseCheckConfig()
        # ResponseCheckConfig.threshold is set by load_from_args, so set it manually
        # (assuming default if not provided, or handle override logic if needed)
        # In mmlu.py, it was set from config.response_check_threshold.
        # Here we use default or allow config attribute access if passed in future.
        if hasattr(config, "response_check_threshold"):
            config.threshold = config.response_check_threshold

        # ResponseCheckDefense only takes config (no preloaded_model)
        defense = ResponseCheckDefense(config)
    elif defense_method == "smoothllm":
        config = SmoothLLMConfig()
        defense = SmoothLLMDefense(config, preloaded_model=None)
    elif defense_method == "paraphrase_prompt":
        config = ParaphraseDefenseConfig()
        defense = ParaphraseDefense(config, preloaded_model=None)
    elif defense_method == "icl":
        config = ICLDefenseConfig()
        defense = ICLDefense(config, preloaded_model=None)
    elif defense_method == "self_reminder":
        config = SelfReminderConfig()
        defense = SelfReminderDefense(config, preloaded_model=None)
        return BatchedSelfReminderDefendedTargetLM(model, defense)
    else:
        raise ValueError(f"Unknown defense method: {defense_method}")

    defended_model = DefendedTargetLM(model, defense)
    return defended_model


def setup_defense_object(defense_method):
    """Create a defense object without wrapping it in DefendedTargetLM.

    This is useful for lm-eval harness integration where we need the defense
    object separately from the model.

    Args:
        defense_method: Name of the defense method

    Returns:
        Defense object instance
    """
    if defense_method == "backtranslation":
        config = BacktranslationConfig()
        defense = BackTranslationDefense(config, preloaded_model=None)
    elif defense_method == "semantic_smoothing":
        config = SemanticSmoothConfig()
        defense = SemanticSmoothDefense(config, preloaded_model=None)
    elif defense_method == "response_check":
        config = ResponseCheckConfig()
        # config.threshold = config.response_check_threshold # Keep consistency with prior impl logic
        defense = ResponseCheckDefense(config)
    elif defense_method == "smoothllm":
        config = SmoothLLMConfig()
        defense = SmoothLLMDefense(config, preloaded_model=None)
    elif defense_method == "paraphrase_prompt":
        config = ParaphraseDefenseConfig()
        defense = ParaphraseDefense(config, preloaded_model=None)
    elif defense_method == "icl":
        config = ICLDefenseConfig()
        defense = ICLDefense(config, preloaded_model=None)
    elif defense_method == "self_reminder":
        config = SelfReminderConfig()
        defense = SelfReminderDefense(config, preloaded_model=None)
    else:
        raise ValueError(f"Unknown defense method: {defense_method}")

    return defense


def detect_model_name_from_path(model_path: str) -> str:
    """
    Detect the model name from the model path.
    Currently supports detecting Llama-2 models.
    """
    model_path = model_path.lower()

    # Llama 2 detection
    if "llama-2" in model_path or "llama2" in model_path:
        if "7b" in model_path:
            return "llama-2-7b"

    if "ministral-8b" in model_path:
        return "ministral-8b"

    if "qwen2-1.5b" in model_path:
        return "qwen2-1.5b"

    return os.path.basename(model_path.rstrip("/"))
