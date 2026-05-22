"""Resolve HarmBench steering mode configs into model loader kwargs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from typing import Any, Dict, List, Optional

from .config_utils import (
    expand_modes_with_num_phi,
    generate_polytope_model_path,
    get_model_config,
    load_config,
)

_BASE_MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "gemma_2_9b": {
        "chat_template": "gemma",
        "dtype": "bfloat16",
        "use_fast_tokenizer": True,
    },
    "llama2_7b": {
        "chat_template": "llama-2",
        "dtype": "float16",
        "use_fast_tokenizer": False,
    },
    "mistral_8b": {
        "chat_template": "mistral",
        "dtype": "bfloat16",
        "use_fast_tokenizer": False,
    },
    "qwen_1.5b": {
        "chat_template": "qwen",
        "dtype": "bfloat16",
        "use_fast_tokenizer": False,
    },
}


def _normalize_model_candidates(model_name: str) -> List[str]:
    raw = model_name.strip()
    stripped = raw[5:] if raw.startswith("safe_") else raw
    candidates: List[str] = []
    for candidate in (raw, stripped, raw.replace("-", "_"), stripped.replace("-", "_")):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _resolve_model_name(config: Dict[str, Any], model_name: str) -> str:
    for candidate in _normalize_model_candidates(model_name):
        try:
            get_model_config(config, candidate)
            return candidate
        except ValueError:
            continue
    raise ValueError(f"Model configuration not found for: {model_name}")


def _normalize_seed(raw_seed: Any) -> Optional[int]:
    if raw_seed is None:
        return None
    if isinstance(raw_seed, list):
        return _normalize_seed(raw_seed[0] if raw_seed else None)
    return int(raw_seed)


def _resolve_effective_seed(
    steering_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    mode_training_cfg: Dict[str, Any],
    seed: Optional[int],
) -> int:
    if seed is not None:
        return int(seed)
    return int(
        _normalize_seed(steering_cfg.get("generation_seed"))
        or mode_training_cfg.get("seed")
        or training_cfg.get("seed")
        or 5
    )


def _resolve_mode_cfg(
    steering_modes: Dict[str, Any],
    steering_mode: str,
    num_phi: Optional[int],
) -> Dict[str, Any]:
    matches = [
        {
            "base_mode_name": base_mode_name,
            "num_phi": phi_value,
            "mode_cfg": mode_cfg,
        }
        for base_mode_name, phi_value, mode_cfg in expand_modes_with_num_phi(
            steering_modes
        )
        if base_mode_name == steering_mode and (num_phi is None or phi_value == num_phi)
    ]
    if not matches:
        raise ValueError(
            f"Steering mode '{steering_mode}' was not found"
            + (f" with num_phi={num_phi}" if num_phi is not None else "")
        )
    if len(matches) > 1:
        phi_values = sorted(match["num_phi"] for match in matches)
        raise ValueError(
            f"Steering mode '{steering_mode}' has multiple num_phi values {phi_values}; "
            "pass --num-phi explicitly."
        )
    return matches[0]


def _resolve_weights_path(
    training_cfg: Dict[str, Any],
    polytope_modes: Dict[str, Any],
    base_mode_name: str,
    num_phi: int,
    seed: int,
) -> Optional[str]:
    mode_training_cfg = (polytope_modes.get(base_mode_name) or {}).copy()

    if "polytope_model_path" in mode_training_cfg:
        explicit_path = mode_training_cfg["polytope_model_path"]
        if not explicit_path:
            return None
        return os.path.abspath(explicit_path)

    reuse_from = mode_training_cfg.get("reuse_weights_from")
    reuse_steering_from = mode_training_cfg.get("reuse_steering_vector_from")
    if reuse_from:
        if reuse_steering_from:
            base_output_dir = mode_training_cfg.get(
                "base_output_dir"
            ) or training_cfg.get("base_output_dir")
            if not base_output_dir:
                return None
            cast_suffix = mode_training_cfg.get("base_exp_ident_suffix", base_mode_name)
            return generate_polytope_model_path(
                base_output_dir, cast_suffix, num_phi, seed=seed
            )

        source_mode_cfg = (polytope_modes.get(reuse_from) or {}).copy()
        if not source_mode_cfg:
            raise ValueError(
                f"Mode '{base_mode_name}' reuses weights from missing mode '{reuse_from}'"
            )
        explicit_source_path = source_mode_cfg.get("polytope_model_path")
        if explicit_source_path:
            return os.path.abspath(explicit_source_path)
        base_output_dir = source_mode_cfg.get("base_output_dir") or training_cfg.get(
            "base_output_dir"
        )
        if not base_output_dir:
            return None
        source_suffix = source_mode_cfg.get("base_exp_ident_suffix", reuse_from)
        return generate_polytope_model_path(
            base_output_dir, source_suffix, num_phi, seed=seed
        )

    if base_mode_name not in polytope_modes:
        return None

    base_output_dir = mode_training_cfg.get("base_output_dir") or training_cfg.get(
        "base_output_dir"
    )
    if not base_output_dir:
        return None
    own_suffix = mode_training_cfg.get("base_exp_ident_suffix", base_mode_name)
    return generate_polytope_model_path(base_output_dir, own_suffix, num_phi, seed=seed)


def _resolve_circuit_breaker_adapter_path(
    training_cfg: Dict[str, Any],
    mode_training_cfg: Dict[str, Any],
    base_mode_name: str,
    seed: int,
) -> Optional[str]:
    if mode_training_cfg.get("model_type") != "circuit_breaker":
        return None

    explicit_path = mode_training_cfg.get(
        "custom_adapter_path"
    ) or mode_training_cfg.get("adapter_path")
    if explicit_path:
        return os.path.abspath(explicit_path)

    base_output_dir = mode_training_cfg.get("base_output_dir") or training_cfg.get(
        "base_output_dir"
    )
    if not base_output_dir:
        return None
    suffix = mode_training_cfg.get("base_exp_ident_suffix", base_mode_name)
    return os.path.abspath(os.path.join(base_output_dir, f"{suffix}_seed{seed}"))


def _merge_cbf_cfg(
    steering_cfg: Dict[str, Any], mode_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    global_cbf = steering_cfg.get("cbf", {})
    if isinstance(global_cbf, dict):
        merged.update(global_cbf)
    mode_cbf = mode_cfg.get("cbf", {})
    if isinstance(mode_cbf, dict):
        merged.update(mode_cbf)
    return merged


def list_available_modes(config_path: str, model_name: str) -> List[Dict[str, Any]]:
    config = load_config(config_path)
    resolved_model_name = _resolve_model_name(config, model_name)
    model_cfg = get_model_config(config, resolved_model_name)
    steering_modes = (model_cfg.get("steering") or {}).get("modes") or {}
    return [
        {
            "mode": base_mode_name,
            "num_phi": phi_value,
            "output_suffix": mode_cfg.get("exp_ident_suffix", base_mode_name),
        }
        for base_mode_name, phi_value, mode_cfg in expand_modes_with_num_phi(
            steering_modes
        )
    ]


def resolve_steering_config(
    config_path: str,
    model_name: str,
    steering_mode: str,
    num_phi: Optional[int] = None,
    seed: Optional[int] = None,
    cbf_k: Optional[float] = None,
    cbf_kappa: Optional[float] = None,
    lambda_weight: Optional[float] = None,
) -> Dict[str, Any]:
    config = load_config(config_path)
    resolved_model_name = _resolve_model_name(config, model_name)
    model_cfg = get_model_config(config, resolved_model_name)

    steering_cfg = (model_cfg.get("steering") or {}).copy()
    training_cfg = (model_cfg.get("polytope_training") or {}).copy()
    steering_modes = (steering_cfg.get("modes") or {}).copy()
    polytope_modes = (training_cfg.get("modes") or {}).copy()

    resolved_mode = _resolve_mode_cfg(steering_modes, steering_mode, num_phi)
    base_mode_name = resolved_mode["base_mode_name"]
    resolved_num_phi = int(resolved_mode["num_phi"])
    mode_cfg = dict(resolved_mode["mode_cfg"])
    mode_training_cfg = (polytope_modes.get(base_mode_name) or {}).copy()
    effective_seed = _resolve_effective_seed(
        steering_cfg, training_cfg, mode_training_cfg, seed
    )

    custom_adapter_path = _resolve_circuit_breaker_adapter_path(
        training_cfg=training_cfg,
        mode_training_cfg=mode_training_cfg,
        base_mode_name=base_mode_name,
        seed=effective_seed,
    )
    polytope_weight_path = None
    if custom_adapter_path is None:
        polytope_weight_path = _resolve_weights_path(
            training_cfg=training_cfg,
            polytope_modes=polytope_modes,
            base_mode_name=base_mode_name,
            num_phi=resolved_num_phi,
            seed=effective_seed,
        )

    cbf_cfg = _merge_cbf_cfg(steering_cfg, mode_cfg)
    use_cbf = bool(cbf_cfg.get("use_cbf", cbf_cfg.get("use", False)))
    cbf_params = {
        "use_cbf": use_cbf,
        "cbf_mode": cbf_cfg.get("mode", cbf_cfg.get("cbf_mode", "estimated")),
        "cbf_dt": float(cbf_cfg.get("dt", cbf_cfg.get("cbf_dt", 1.0))),
        "cbf_k": (
            cbf_k if cbf_k is not None else cbf_cfg.get("k", cbf_cfg.get("cbf_k", 1.0))
        ),
        "cbf_w": float(cbf_cfg.get("w", cbf_cfg.get("cbf_w", 1.0))),
        "cbf_p": float(cbf_cfg.get("p", cbf_cfg.get("cbf_p", 10.0))),
        "cbf_kappa": float(
            cbf_kappa
            if cbf_kappa is not None
            else cbf_cfg.get("kappa", cbf_cfg.get("cbf_kappa", 10.0))
        ),
        "cbf_max_constraints": cbf_cfg.get(
            "max_constraints",
            cbf_cfg.get("cbf_max_constraints", cbf_cfg.get("cbf_max", 2)),
        ),
        "cbf_constraint_mode": cbf_cfg.get(
            "constraint_mode",
            cbf_cfg.get("cbf_constraint_mode", "topk"),
        ),
        "cbf_control_radius": float(
            cbf_cfg.get("control_radius", cbf_cfg.get("cbf_control_radius", 1.0))
        ),
        "cbf_num_steps": int(cbf_cfg.get("num_steps", cbf_cfg.get("cbf_num_steps", 1))),
    }

    multi_cbf = cbf_cfg.get("multi_cbf", {})
    if isinstance(multi_cbf, dict):
        cbf_params["multi_cbf_enabled"] = bool(multi_cbf.get("enabled", False))
        if multi_cbf.get("models_dir"):
            cbf_params["multi_cbf_models_dir"] = os.path.abspath(
                multi_cbf["models_dir"]
            )
        if multi_cbf.get("load_attacks"):
            cbf_params["multi_cbf_load_attacks"] = multi_cbf["load_attacks"]

    steering_vector_params = {
        "model_type": mode_training_cfg.get("model_type"),
        "steering_method": mode_training_cfg.get("steering_method"),
        "steering_alpha": mode_training_cfg.get("steering_alpha"),
        "steering_normalize": mode_training_cfg.get("steering_normalize"),
        "reuse_weights_from": mode_training_cfg.get("reuse_weights_from"),
        "reuse_steering_vector_from": mode_training_cfg.get(
            "reuse_steering_vector_from"
        ),
    }

    if resolved_model_name not in _BASE_MODEL_DEFAULTS:
        raise KeyError(
            f"No base model defaults configured for '{resolved_model_name}'. "
            "Add an entry to _BASE_MODEL_DEFAULTS."
        )
    model_path = model_cfg["path"]
    harmbench_model_cfg: Dict[str, Any] = {
        **_BASE_MODEL_DEFAULTS[resolved_model_name],
        "model_name_or_path": model_path,
        "pretrained_model_name_or_path": model_path,
    }
    if polytope_weight_path:
        harmbench_model_cfg["polytope_weight_path"] = polytope_weight_path
        resolved_lambda_weight = (
            lambda_weight
            if lambda_weight is not None
            else mode_cfg.get("lambda_weight", mode_training_cfg.get("lambda_weight"))
        )
        if resolved_lambda_weight is not None:
            harmbench_model_cfg["lambda_weight"] = float(resolved_lambda_weight)
    if custom_adapter_path:
        harmbench_model_cfg["custom_adapter_path"] = custom_adapter_path
    if polytope_weight_path or use_cbf:
        harmbench_model_cfg.update(cbf_params)

    return {
        "config_path": os.path.abspath(config_path),
        "requested_model_name": model_name,
        "model_name": resolved_model_name,
        "requested_steering_mode": steering_mode,
        "steering_mode": base_mode_name,
        "num_phi": resolved_num_phi,
        "seed": effective_seed,
        "model_path": model_path,
        "polytope_weight_path": polytope_weight_path,
        "custom_adapter_path": custom_adapter_path,
        "use_cbf": use_cbf,
        "cbf_params": cbf_params,
        "steering_vector_params": steering_vector_params,
        "output_suffix": mode_cfg.get("exp_ident_suffix", base_mode_name),
        "harmbench_model_cfg": harmbench_model_cfg,
    }


def _shell_value(value: Any) -> str:
    if value is None:
        return "''"
    if isinstance(value, (dict, list, bool)):
        return shlex.quote(json.dumps(value))
    return shlex.quote(str(value))


def _print_shell(result: Dict[str, Any]) -> None:
    env_map = {
        "CONFIG_PATH": result["config_path"],
        "MODEL_NAME": result["model_name"],
        "STEERING_MODE": result["steering_mode"],
        "NUM_PHI": result["num_phi"],
        "SEED": result["seed"],
        "MODEL_PATH": result["model_path"],
        "POLYTOPE_WEIGHT_PATH": result["polytope_weight_path"],
        "CUSTOM_ADAPTER_PATH": result.get("custom_adapter_path"),
        "USE_CBF": result["use_cbf"],
        "OUTPUT_SUFFIX": result["output_suffix"],
        "CBF_PARAMS_JSON": result["cbf_params"],
        "STEERING_VECTOR_PARAMS_JSON": result["steering_vector_params"],
        "HARMBENCH_MODEL_CFG_JSON": result["harmbench_model_cfg"],
    }
    for key, value in env_map.items():
        print(f"{key}={_shell_value(value)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to harmbench.yaml")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--mode", help="Steering mode name")
    parser.add_argument("--num-phi", type=int, default=None, help="num_phi override")
    parser.add_argument(
        "--cbf-k",
        type=float,
        default=None,
        help="CBF steering-strength alpha/k override",
    )
    parser.add_argument(
        "--cbf-kappa", type=float, default=None, help="CBF LSE smoothing kappa override"
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed override")
    parser.add_argument("--format", choices=("shell", "json"), default="shell")
    parser.add_argument("--list-modes", action="store_true")
    args = parser.parse_args()

    if args.list_modes:
        result: Any = list_available_modes(args.config, args.model)
    else:
        if not args.mode:
            parser.error("--mode is required unless --list-modes is set")
        result = resolve_steering_config(
            config_path=args.config,
            model_name=args.model,
            steering_mode=args.mode,
            num_phi=args.num_phi,
            seed=args.seed,
            cbf_k=args.cbf_k,
            cbf_kappa=args.cbf_kappa,
        )

    if args.format == "json" or args.list_modes:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    _print_shell(result)


if __name__ == "__main__":
    main()
