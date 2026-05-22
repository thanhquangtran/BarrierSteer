"""
Configuration utilities for HarmBench pipeline
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _expand_env_vars(obj: Any) -> Any:
    """
    Recursively expand environment variables in configuration values.

    Args:
        obj: Configuration object (dict, list, or str)

    Returns:
        Object with environment variables expanded
    """
    if isinstance(obj, dict):
        return {key: _expand_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        return os.path.expandvars(obj)
    else:
        return obj


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file and expand environment variables.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary with environment variables expanded
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Expand environment variables in all string values
    config = _expand_env_vars(config)

    return config


def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model

    Args:
        config: Full pipeline configuration
        model_name: Name of the model

    Returns:
        Model-specific configuration
    """
    for model in config.get("models", []):
        if model["name"] == model_name:
            return model

    raise ValueError(f"Model configuration not found for: {model_name}")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate pipeline configuration

    Args:
        config: Configuration dictionary to validate
    """
    required_sections = ["pipeline", "models", "slurm"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate pipeline section
    pipeline_config = config["pipeline"]
    required_pipeline_keys = ["harmbench_path", "barriersteer_path"]
    for key in required_pipeline_keys:
        if key not in pipeline_config:
            raise ValueError(f"Missing required pipeline configuration: {key}")

    # Validate that paths exist
    harmbench_path = pipeline_config["harmbench_path"]
    barriersteer_path = pipeline_config["barriersteer_path"]

    if not os.path.exists(harmbench_path):
        raise ValueError(f"HarmBench path does not exist: {harmbench_path}")

    if not os.path.exists(barriersteer_path):
        raise ValueError(f"BarrierSteer path does not exist: {barriersteer_path}")

    # Validate models
    if not config.get("models"):
        raise ValueError("No models configured")

    for model in config["models"]:
        required_model_keys = ["name", "path"]
        # specific check for attack_methods being optional if use_category_data is true
        use_category_data = (
            config.get("polytope_training", {}).get("use_category_data", False)
            or config.get("polytope_training", {})
            .get("dataset", {})
            .get("use_category_data", False)
            or model.get("polytope_training", {}).get("use_category_data", False)
            or model.get("polytope_training", {})
            .get("dataset", {})
            .get("use_category_data", False)
        )
        if "attack_methods" not in model and not use_category_data:
            raise ValueError(f"Missing required model configuration: attack_methods")

        if (
            "attack_methods" in model
            and not model["attack_methods"]
            and not use_category_data
        ):
            raise ValueError(f"No attack methods configured for model: {model['name']}")


def get_harmbench_model_name(model_name: str) -> str:
    """
    Transform model name to HarmBench convention.

    Args:
        model_name: Original model name

    Returns:
        HarmBench-formatted model name
    """
    name_lower = model_name.lower()

    if "llama" in name_lower:
        return "safe_llama2_7b"
    elif "mistral" in name_lower:
        return "safe_mistral_8b"
    elif "qwen" in name_lower:
        return "safe_qwen_1.5b"
    else:
        return f"safe_{model_name}"


def get_results_directory(
    harmbench_path: Path,
    method: str,
    model_name: str,
    transform_model_name: bool = True,
) -> Path:
    """
    Get results directory path for HarmBench method and model.

    Args:
        harmbench_path: Base HarmBench path
        method: Attack method name
        model_name: Model name
        transform_model_name: Whether to transform model name to HarmBench convention

    Returns:
        Results directory path
    """
    # Check for HARMBENCH_RESULTS_DIR environment variable to override default results path
    base_results_dir = os.environ.get("HARMBENCH_RESULTS_DIR")
    if base_results_dir:
        base_path = Path(base_results_dir)
    else:
        base_path = harmbench_path / "results"

    # Methods that use /default/ instead of /{model_name}/
    # HumanJailbreaks uses random_subset_5 for both test cases and results
    if method == "DirectRequest":
        return base_path / method / "default"
    elif method == "HumanJailbreaks":
        return base_path / method / "random_subset_5"
    elif method == "PAP-top5":
        return base_path / "PAP" / "top_5"
    else:
        if transform_model_name:
            model_name = get_harmbench_model_name(model_name)
        return base_path / method / model_name


def get_method_path(
    harmbench_path: Path,
    method: str,
    model_name: str,
    subdirectory: str,
    transform_model_name: bool = True,
) -> Path:
    """
    Get the complete path for a given attack method and subdirectory.

    Args:
        harmbench_path: Base HarmBench path
        method: Attack method name
        model_name: Model name
        subdirectory: Subdirectory (e.g., 'test_cases', 'hidden_states', 'completions')
        transform_model_name: Whether to transform model name to HarmBench convention

    Returns:
        Complete path
    """
    results_dir = get_results_directory(
        harmbench_path, method, model_name, transform_model_name
    )
    return results_dir / subdirectory


def expand_modes_with_num_phi(
    modes: Dict[str, Any],
) -> List[Tuple[str, int, Dict[str, Any]]]:
    """
    Expand modes that have num_phi as a list into separate mode entries.

    Args:
        modes: Dictionary of mode configurations, where num_phi can be a single value or list

    Returns:
        List of (mode_name, num_phi, mode_config) tuples
    """
    expanded = []
    for mode_name, mode_cfg in modes.items():
        # Default to [0] if num_phi is missing (e.g., for defense modes)
        num_phi = mode_cfg.get("num_phi", [0])

        # Convert single value to list for uniform handling
        if not isinstance(num_phi, list):
            num_phi = [num_phi]

        # Create a copy of mode_cfg for each num_phi value
        # Use fixed 3-digit padding for consistency (supports up to 999, ensures alphabetical = numerical sort)
        for phi_val in num_phi:
            mode_cfg_copy = mode_cfg.copy()
            mode_cfg_copy["num_phi"] = phi_val

            # Generate exp_ident_suffix with num_phi (zero-padded for consistent sorting)
            base_suffix = mode_cfg_copy.get("base_exp_ident_suffix", mode_name)

            # Only append phi suffix if num_phi was explicitly configured (not default 0)
            if "num_phi" in mode_cfg:
                mode_cfg_copy["exp_ident_suffix"] = f"{base_suffix}_phi{phi_val:03d}"
            else:
                # For defense modes or modes without num_phi, just use base suffix
                mode_cfg_copy["exp_ident_suffix"] = base_suffix

            expanded.append((mode_name, phi_val, mode_cfg_copy))

    return expanded


def generate_polytope_model_path(
    base_output_dir: str, base_exp_ident_suffix: str, num_phi: int, seed: int = None
) -> str:
    """
    Generate polytope model path from base directory and mode info.
    Checks both zero-padded and non-padded formats for backward compatibility.

    Args:
        base_output_dir: Base output directory
        base_exp_ident_suffix: Base experiment identifier suffix
        num_phi: Number of phi functions
        seed: Optional seed for the model filename

    Returns:
        Full path to weights file
    """
    # Determine filename based on seed
    filename = "weights.pth"
    if seed is not None:
        filename = f"weight_{seed}.pth"

    # Try zero-padded format first (new format)
    mode_dir_padded = f"{base_exp_ident_suffix}_phi{num_phi:03d}"
    path_padded = os.path.join(base_output_dir, mode_dir_padded, filename)

    # Try non-padded format (old format for backward compatibility)
    mode_dir_old = f"{base_exp_ident_suffix}_phi{num_phi}"
    path_old = os.path.join(base_output_dir, mode_dir_old, filename)

    # Check which one exists
    if os.path.exists(path_padded):
        return path_padded
    elif os.path.exists(path_old):
        return path_old

    # Fallback to weights.pth if a seeded weight_X.pth was not found (e.g. ActAdd vectors)
    if seed is not None:
        fallback_padded = os.path.join(base_output_dir, mode_dir_padded, "weights.pth")
        fallback_old = os.path.join(base_output_dir, mode_dir_old, "weights.pth")
        if os.path.exists(fallback_padded):
            return fallback_padded
        if os.path.exists(fallback_old):
            return fallback_old

    # Return zero-padded format by default (for new training runs)
    return path_padded


def generate_circuit_breaker_model_path(
    base_output_dir: str,
    base_exp_ident_suffix: str,
    seed: int = None,
) -> str:
    """Generate a deterministic output directory for a CircuitBreaker checkpoint."""
    base_dir = Path(base_output_dir)
    dirname = (
        base_exp_ident_suffix if seed is None else f"{base_exp_ident_suffix}_seed{seed}"
    )
    return str(base_dir / dirname)


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Setup logging configuration

    Args:
        config: Configuration dictionary
    """
    import logging

    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO")
    log_dir = log_config.get("log_dir", "./logs")

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "pipeline.log")),
            logging.StreamHandler(),
        ],
    )
