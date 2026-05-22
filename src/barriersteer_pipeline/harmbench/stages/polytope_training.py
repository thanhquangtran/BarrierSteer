"""
Stage 4: Polytope Training
Trains safety polytope constraints using processed hidden states.
"""

import glob
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from steer.common.outputs import ModelResult

try:
    from barriersteer_pipeline.harmbench.utils.config_utils import (
        generate_polytope_model_path,
    )
except ImportError:
    # Fallback for relative import when used as a module
    from ..utils.config_utils import generate_polytope_model_path


class PolytopeTrainingStage:
    """Stage 4: Train safety polytope constraints"""

    def _compose_cast_artifact(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        training_mode: str,
        polytope_model_path: str,
        seed: Optional[int],
    ) -> None:
        modes = (model_config or {}).get("polytope_training", {}).get("modes", {})
        detector_mode = training_config.get("reuse_weights_from")
        steering_mode = training_config.get("reuse_steering_vector_from")
        if not detector_mode or not steering_mode:
            raise ValueError(
                f"CAST mode '{training_mode}' requires both reuse_weights_from and reuse_steering_vector_from"
            )
        detector_cfg = (modes.get(detector_mode) or {}).copy()
        steering_cfg = (modes.get(steering_mode) or {}).copy()
        if not detector_cfg:
            raise ValueError(
                f"Detector mode '{detector_mode}' not found for CAST mode '{training_mode}'"
            )
        if not steering_cfg:
            raise ValueError(
                f"Steering mode '{steering_mode}' not found for CAST mode '{training_mode}'"
            )

        effective_seed = (
            seed
            if seed is not None
            else training_config.get(
                "seed", self.config.get("polytope_training", {}).get("seed", 5)
            )
        )
        num_phi = training_config.get("num_phi", 10)

        parent_training_cfg = (
            (model_config or {}).get("polytope_training", {})
            if isinstance(model_config, dict)
            else {}
        )

        def resolve_path(
            cfg: Dict[str, Any], mode_name: str, default_num_phi: int
        ) -> str:
            explicit = cfg.get("polytope_model_path")
            if explicit:
                return os.path.abspath(os.path.expandvars(explicit))
            base_output_dir = (
                cfg.get("base_output_dir")
                or training_config.get("base_output_dir")
                or parent_training_cfg.get("base_output_dir")
                or self.config.get("polytope_training", {}).get("base_output_dir")
            )
            if not base_output_dir:
                raise ValueError(
                    f"Could not resolve base_output_dir for mode '{mode_name}'"
                )
            suffix = cfg.get("base_exp_ident_suffix", mode_name)
            num_phi_val = cfg.get("num_phi", default_num_phi)
            if isinstance(num_phi_val, list):
                num_phi_val = num_phi_val[0]
            if cfg.get("model_type") == "steering_vector" and "num_phi" not in cfg:
                num_phi_val = 0
            seeded_path = generate_polytope_model_path(
                base_output_dir, suffix, num_phi_val, seed=effective_seed
            )
            if os.path.exists(seeded_path):
                return seeded_path
            unseeded_path = os.path.join(
                os.path.dirname(seeded_path),
                "weights.pth",
            )
            if os.path.exists(unseeded_path):
                return unseeded_path
            return seeded_path

        detector_path = resolve_path(detector_cfg, detector_mode, num_phi)
        steering_path = resolve_path(steering_cfg, steering_mode, num_phi)
        if not os.path.exists(detector_path):
            raise FileNotFoundError(f"CAST detector weights not found: {detector_path}")
        if not os.path.exists(steering_path):
            raise FileNotFoundError(
                f"CAST steering-vector weights not found: {steering_path}"
            )

        detector_model = torch.load(
            detector_path, map_location="cpu", weights_only=False
        )
        steering_model = torch.load(
            steering_path, map_location="cpu", weights_only=False
        )
        if (
            not hasattr(steering_model, "steering_vector")
            or steering_model.steering_vector is None
        ):
            raise ValueError(
                f"Steering source '{steering_mode}' does not contain a steering_vector"
            )
        if not hasattr(detector_model, "threshold") or detector_model.threshold is None:
            raise ValueError(
                f"Detector source '{detector_mode}' does not contain detector thresholds"
            )

        combined = ModelResult(
            getattr(detector_model, "phi", None),
            detector_model.threshold,
            getattr(detector_model, "feature_extractor", None),
            phi_network=getattr(detector_model, "phi_network", None),
            steering_vector=steering_model.steering_vector,
            steering_method=training_config.get(
                "steering_method",
                getattr(
                    steering_model,
                    "steering_method",
                    getattr(steering_model, "method", "actadd"),
                ),
            ),
            steering_alpha=training_config.get(
                "steering_alpha",
                getattr(
                    steering_model,
                    "steering_alpha",
                    getattr(steering_model, "alpha", 1.0),
                ),
            ),
            conditional_steering=True,
        )
        os.makedirs(os.path.dirname(polytope_model_path), exist_ok=True)
        torch.save(combined, polytope_model_path)
        self.logger.info(
            f"Composed CAST artifact for mode '{training_mode}' from detector={detector_path} and steering={steering_path} -> {polytope_model_path}"
        )

    def __init__(self, config: Dict[str, Any], barriersteer_path: str):
        self.config = config
        self.barriersteer_path = barriersteer_path
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        model_name: str,
        processed_data_path: str,
        model_config: Optional[Dict[str, Any]] = None,
        training_mode: Optional[str] = None,
        gpu_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Train polytope constraints

        Args:
            model_name: Name of the model
            processed_data_path: Path to processed hidden state data
            model_config: Model-specific configuration containing polytope training hyperparameters

        """
        self.logger.info(f"Starting polytope training for {model_name}")

        # Change to BarrierSteer directory
        original_dir = os.getcwd()
        os.chdir(self.barriersteer_path)

        try:
            # Ensure processed_data_path is absolute because Hydra changes working directory
            processed_data_path = os.path.abspath(processed_data_path)

            # Train polytope
            self._train_polytope(
                model_name,
                processed_data_path,
                model_config=model_config,
                training_mode=training_mode,
                gpu_id=gpu_id,
                seed=seed,
            )

        finally:
            os.chdir(original_dir)

    def _train_polytope(
        self,
        model_name: str,
        processed_data_path: str,
        model_config: Optional[Dict[str, Any]] = None,
        training_mode: Optional[str] = None,
        gpu_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Train polytope using learn_polytope.py

        Args:
            model_name: Name of the model
            processed_data_path: Path to processed data
            model_config: Model-specific configuration containing polytope training hyperparameters

        """
        self.logger.info(f"Training polytope for {model_name}")

        # Get training configuration - merge global and model-specific configs
        training_config = self.config.get("polytope_training", {}).copy()

        if model_config and "polytope_training" in model_config:
            model_training_config = model_config["polytope_training"]
            self.logger.info(
                f"Merging model-specific polytope training config for {model_name}"
            )
            # Update global config with model-specific overrides (shallow merge)
            training_config.update(model_training_config)
        else:
            self.logger.info(f"Using global polytope training config for {model_name}")

        # Optional per-mode overrides
        mode_cfg: Dict[str, Any] = {}
        if training_mode and isinstance(training_config, dict):
            modes = training_config.get("modes", {})
            if isinstance(modes, dict) and training_mode in modes:
                mode_cfg = modes.get(training_mode, {}) or {}
                if mode_cfg:
                    self.logger.info(
                        f"Using polytope training mode '{training_mode}' overrides"
                    )

        # Merge mode overrides (shallow)
        if mode_cfg:
            training_config = {**training_config, **mode_cfg}

        # Check if per-attack training is enabled
        per_attack_training = training_config.get("per_attack_training", False)
        if per_attack_training:
            attack_types_to_train = training_config.get("attack_types_to_train", [])
            if not attack_types_to_train:
                raise ValueError(
                    f"per_attack_training is enabled but attack_types_to_train is empty for {training_mode}"
                )

            self.logger.info(
                f"Per-attack training enabled for {len(attack_types_to_train)} attack types"
            )
            self.logger.info(f"Attack types: {attack_types_to_train}")

            # Train a separate model for each attack type
            for attack_type in attack_types_to_train:
                self.logger.info(f"\\n{'='*80}")
                self.logger.info(f"Training CBF model for attack type: {attack_type}")
                self.logger.info(f"{'='*80}\\n")

                # Create modified config with attack type filter
                attack_training_config = training_config.copy()
                attack_training_config["_per_attack_type_filter"] = attack_type

                # Call run method recursively with attack-specific path
                # Temporarily disable per_attack_training to avoid infinite recursion
                attack_training_config["per_attack_training"] = False

                # Use a temporary training mode name to avoid path conflicts
                attack_training_mode = f"{training_mode}__{attack_type}"

                # Generate distinct file path in the SHARED directory
                # First, determine the shared directory based on the original configuration
                original_suffix = training_config.get(
                    "base_exp_ident_suffix", training_mode
                )
                base_output_dir = training_config.get("base_output_dir", None)
                num_phi = training_config.get("num_phi", 30)

                # Get the standard path for this configuration (weights.pth)
                if base_output_dir:
                    standard_path = generate_polytope_model_path(
                        base_output_dir, original_suffix, num_phi
                    )
                    shared_dir = os.path.dirname(standard_path)
                else:
                    # Fallback if base_output_dir is missing (should verify usually)
                    shared_dir = os.path.join(
                        self.barriersteer_path, "outputs", "harmbench_models"
                    )

                # Explicitly set the destination path for this attack's weights
                # e.g., .../cbf_per_attack_num_phi_1/weights_AutoDAN.pth
                attack_training_config["polytope_model_path"] = os.path.join(
                    shared_dir, f"weights_{attack_type}.pth"
                )

                # Update base_exp_ident_suffix ONLY for Hydra logging purposes (so logs don't overwrite)
                # This does not affect the destination path because we explicitly set polytope_model_path above
                attack_training_config["base_exp_ident_suffix"] = (
                    f"{original_suffix}_{attack_type}"
                )

                # Update the mode in the model config temporarily
                temp_model_config = model_config.copy()
                temp_model_config["polytope_training"] = {
                    **temp_model_config.get("polytope_training", {}),
                    "modes": {attack_training_mode: attack_training_config},
                }

                # Train for this specific attack type
                self._train_polytope(
                    model_name=model_name,
                    processed_data_path=processed_data_path,
                    model_config=temp_model_config,
                    training_mode=attack_training_mode,
                    gpu_id=gpu_id,
                )

            self.logger.info(
                f"\\nCompleted per-attack training for all {len(attack_types_to_train)} attack types"
            )
            return  # Early return after per-attack training completes

        execution_mode = self.config.get("polytope_training", {}).get(
            "execution_mode", "slurm"
        )

        # Get the output path for saving weights
        # Support both explicit polytope_model_path and dynamic generation from base_output_dir
        polytope_model_path = training_config.get("polytope_model_path", None)

        if not polytope_model_path:
            # Try to generate from base_output_dir and base_exp_ident_suffix
            base_output_dir = training_config.get("base_output_dir", None)
            base_exp_ident_suffix = training_config.get(
                "base_exp_ident_suffix", training_mode
            )
            num_phi = training_config.get("num_phi", 10)

            if base_output_dir:
                # Get seed for filename: prefer explicit parameter, then config
                effective_seed = (
                    seed
                    if seed is not None
                    else training_config.get(
                        "seed", self.config.get("polytope_training", {}).get("seed", 5)
                    )
                )
                polytope_model_path = generate_polytope_model_path(
                    base_output_dir, base_exp_ident_suffix, num_phi, seed=effective_seed
                )

        if not polytope_model_path:
            raise ValueError(
                f"polytope_model_path or base_output_dir must be specified in polytope_training.modes.{training_mode} for model {model_name}"
            )

        model_type = training_config.get("model_type", "polytope")
        if model_type == "conditional_steering_vector" and training_config.get(
            "reuse_steering_vector_from"
        ):
            self._compose_cast_artifact(
                model_name=model_name,
                model_config=model_config or {},
                training_config=training_config,
                training_mode=training_mode or "cast",
                polytope_model_path=polytope_model_path,
                seed=seed,
            )
            return

        # Check for per-category training
        # NOTE: This is reserved for training SEPARATE models for each category.
        # Use 'use_category_data' if you want to train a SINGLE model on all category data.
        per_category_training = training_config.get("per_category_training", False)
        if per_category_training:
            self.logger.info(
                f"Per-category training enabled. Processed data path: {processed_data_path}"
            )
            # Find all category files: {category}_train.pt
            category_files = glob.glob(os.path.join(processed_data_path, "*_train.pt"))
            self.logger.info(f"Found {len(category_files)} category files")

            # Check for target categories filter (support alias category_types_to_train)
            target_categories = training_config.get("target_categories", None)
            if not target_categories:
                target_categories = training_config.get("category_types_to_train", None)

            if target_categories:
                self.logger.info(f"Filtering to target categories: {target_categories}")

            for cat_file in category_files:
                filename = os.path.basename(cat_file)
                # Parse category name (remove _train.pt)
                category = filename.replace("_train.pt", "")

                # Filter if target_categories is specified
                if target_categories and category not in target_categories:
                    self.logger.info(
                        f"Skipping category {category} (not in target_categories)"
                    )
                    continue

                self.logger.info(f"Training polytope for category: {category}")

                # Determine output path for this category
                # If polytope_model_path is a file (weights.pth), we make it weights_{category}.pth
                # If it's a directory, we put weights_{category}.pth inside it

                if polytope_model_path.endswith(".pth"):
                    dir_path = os.path.dirname(polytope_model_path)
                    base_name = os.path.basename(polytope_model_path)
                    name_part, ext = os.path.splitext(base_name)
                    cat_model_path = os.path.join(
                        dir_path, f"{name_part}_{category}{ext}"
                    )
                else:
                    # Assume directory
                    cat_model_path = os.path.join(
                        polytope_model_path, f"weights_{category}.pth"
                    )

                self.logger.info(f"Output model path: {cat_model_path}")
                os.makedirs(os.path.dirname(cat_model_path), exist_ok=True)

                # Recursive call with modified path and dataset
                # Temporarily disable per_category_training to avoid endless recursion
                cat_training_config = training_config.copy()
                cat_training_config["per_category_training"] = False

                # Create a temporary config to override process_data_path and training config
                # We need to construct a new model_config dict
                cat_model_config = model_config.copy() if model_config else {}

                # We also need to override the training config attached to it
                cat_model_config["polytope_training"] = cat_training_config

                # Override polytope_model_path in the config so it's picked up
                cat_model_config["polytope_training"][
                    "polytope_model_path"
                ] = cat_model_path

                # IMPORTANT: Set attack_methods to only this category so only this file is loaded
                # We pass the directory to _train_polytope, but learn_polytope needs to know which file to load
                cat_model_config["attack_methods"] = [category]

                # Use a specific training mode string to potentially avoid directory conflicts in Hydra
                cat_training_mode = (
                    f"{training_mode}__{category}"
                    if training_mode
                    else f"train__{category}"
                )

                # Ensure the mode config exists if we are using it
                if "modes" not in cat_model_config["polytope_training"]:
                    cat_model_config["polytope_training"]["modes"] = {}
                cat_model_config["polytope_training"]["modes"][
                    cat_training_mode
                ] = cat_training_config

                self._train_polytope(
                    model_name=model_name,
                    processed_data_path=os.path.dirname(
                        cat_file
                    ),  # Pass the directory!
                    model_config=cat_model_config,
                    training_mode=cat_training_mode,
                    gpu_id=gpu_id,
                )

            return  # Done with all categories

        # Ensure output directory exists (original logic)
        os.makedirs(os.path.dirname(polytope_model_path), exist_ok=True)

        exp_ident = training_config.get("exp_ident", None)
        if not exp_ident:
            suffix = training_config.get("exp_ident_suffix", None)
            exp_ident = f"harmbench_polytope_{model_name}"
            if suffix:
                exp_ident = f"{exp_ident}_{suffix}"

        # Get num_phi from mode config, fall back to training_config, then global default
        num_phi = training_config.get("num_phi", None)
        if num_phi is None:
            # Fall back to global polytope_training config
            num_phi = self.config.get("polytope_training", {}).get("num_phi", 30)

        # Prepare command arguments with model-specific hyperparameters using Hydra override syntax
        # Get num_epochs from training_config (which includes mode overrides), fall back to global config
        num_epochs = training_config.get("num_epochs", None)
        if num_epochs is None:
            num_epochs = self.config.get("polytope_training", {}).get("num_epochs", 1)

        cmd = [
            "python",
            "src/steer/polytope/learn_polytope.py",
            "dataset=harmbench",
            f"dataset.hidden_states_path={processed_data_path}",
            f'seed={seed if seed is not None else self.config.get("polytope_training", {}).get("seed", 5)}',
            f"dataset.num_epochs={num_epochs}",
            f"dataset.num_phi={num_phi}",
            f'learning_rate={training_config.get("learning_rate", 0.01)}',
            f'batch_size={training_config.get("batch_size", 128)}',
            f'feature_dim={training_config.get("feature_dim", 16384)}',
            f'entropy_weight={training_config.get("entropy_weight", 1.0)}',
            f'unsafe_weight={training_config.get("unsafe_weight", 3.0)}',
            f'f_l1_weight={training_config.get("lambda_constraint", 1.0)}',
            f'phi_l1_weight={training_config.get("phi_l1_weight", 0.0001)}',
            f'margin={training_config.get("margin", 1.0)}',
            f'loss_type={training_config.get("loss_type", "relu")}',
            f'log_interval={training_config.get("log_interval", 100)}',
            f"exp_ident={exp_ident}",
        ]

        # Add attack_methods if specified in model_config
        # Check if we should use category data (implies using all found categories as "methods")
        use_category_data = training_config.get("use_category_data", False)

        if use_category_data:
            self.logger.info("use_category_data=True: Training on category data.")

        # Pass attack_methods (or categories) if explicitly configured
        # This handles both:
        # 1. Standard attack training where attack_methods are filters
        # 2. Per-category training where the loop sets explicit [category] as the method
        if model_config and "attack_methods" in model_config:
            attack_methods = model_config["attack_methods"]
            # Format list as Hydra override: [method1,method2,method3]
            methods_str = "[" + ",".join(attack_methods) + "]"
            cmd.append(f"dataset.attack_methods={methods_str}")

            if use_category_data:
                self.logger.info(f"Filtering to specific categories: {attack_methods}")
        elif use_category_data:
            self.logger.info(
                "No specific categories requested. learn_polytope will discover all available categories."
            )

        # Add per-attack type filter if specified (overrides attack_methods for single-attack training)
        per_attack_type = training_config.get("_per_attack_type_filter", None)
        if per_attack_type:
            # Override with single attack type for this training run
            cmd.append(f"dataset.attack_methods=[{per_attack_type}]")
            self.logger.info(f"Training with single attack type: {per_attack_type}")

        # Add use_nonlinear parameter if specified
        if "use_nonlinear" in training_config:
            cmd.append(f'use_nonlinear={training_config.get("use_nonlinear", False)}')

        # Add neural phi parameters if specified
        if training_config.get("use_neural_phi", False):
            cmd.append(f'use_neural_phi={training_config.get("use_neural_phi", False)}')
            cmd.append(f'phi_hidden_dim={training_config.get("phi_hidden_dim", 512)}')
            cmd.append(
                f'neural_phi_architecture={training_config.get("neural_phi_architecture", "mlp2")}'
            )
            neural_phi_hidden_dims = training_config.get("neural_phi_hidden_dims", None)
            if neural_phi_hidden_dims:
                # Format list as Hydra override: [val1,val2,val3]
                dims_str = "[" + ",".join(str(d) for d in neural_phi_hidden_dims) + "]"
                cmd.append(f"neural_phi_hidden_dims={dims_str}")

            # Add final activation if specified
            final_activation = training_config.get("neural_phi_final_activation", None)
            if final_activation:
                cmd.append(f"neural_phi_final_activation={final_activation}")

        # Add descent loss parameters if specified
        if "desc_loss_weight" in training_config:
            cmd.append(
                f'desc_loss_weight={training_config.get("desc_loss_weight", 0.0)}'
            )
        if "control_radius" in training_config:
            cmd.append(f'control_radius={training_config.get("control_radius", 1.0)}')
        if "cbf_k" in training_config:
            cmd.append(f'cbf_k={training_config.get("cbf_k", 1.0)}')

        if "safe_margin" in training_config:
            cmd.append(f'+safe_margin={training_config.get("safe_margin", 1.0)}')
        if "unsafe_margin" in training_config:
            cmd.append(f'+unsafe_margin={training_config.get("unsafe_margin", 1.0)}')

        # Add model_type parameter (supports "polytope", "mlp", "steering_vector", "conditional_steering_vector")
        model_type = training_config.get("model_type", "polytope")
        cmd.append(f"model_type={model_type}")

        # Add steering vector parameters for ActAdd/DirAblate/CAST
        if model_type in {"steering_vector", "conditional_steering_vector"}:
            steering_method = training_config.get("steering_method", "actadd")
            cmd.append(f"steering_method={steering_method}")
            cmd.append(f'steering_alpha={training_config.get("steering_alpha", 1.0)}')
            cmd.append(
                f'steering_normalize={training_config.get("steering_normalize", True)}'
            )

        if execution_mode == "slurm":
            cmd.append("--multirun")

        log_params = [
            f"num_phi={num_phi}",
            f"learning_rate={training_config.get('learning_rate', 0.01)}",
            f"batch_size={training_config.get('batch_size', 128)}",
            f"feature_dim={training_config.get('feature_dim', 16384)}",
            f"entropy_weight={training_config.get('entropy_weight', 1.0)}",
            f"lambda_constraint={training_config.get('lambda_constraint', 1.0)}",
            f"margin={training_config.get('margin', 1.0)}",
            f"loss_type={training_config.get('loss_type', 'relu')}",
            f"safe_margin={training_config.get('safe_margin', 'default')}",
            f"unsafe_margin={training_config.get('unsafe_margin', 'default')}",
        ]
        if "use_nonlinear" in training_config:
            log_params.append(
                f"use_nonlinear={training_config.get('use_nonlinear', False)}"
            )
        if training_config.get("use_neural_phi", False):
            log_params.append(
                f"use_neural_phi={training_config.get('use_neural_phi', False)}"
            )
            log_params.append(
                f"phi_hidden_dim={training_config.get('phi_hidden_dim', 512)}"
            )
            log_params.append(
                f"neural_phi_architecture={training_config.get('neural_phi_architecture', 'mlp2')}"
            )
        self.logger.info(f"Training with hyperparameters: {', '.join(log_params)}")

        self.logger.info(f"Running command: {' '.join(cmd)}")

        # Set GPU environment variable if specified
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            self.logger.info(f"Using GPU {gpu_id} for training")

        # Run training with real-time output streaming
        self.logger.info("Starting training process...")
        result = subprocess.run(cmd, env=env)

        if result.returncode != 0:
            self.logger.error(
                f"Polytope training failed with return code: {result.returncode}"
            )
            raise RuntimeError(
                f"Polytope training failed with return code: {result.returncode}"
            )

        # Copy weights from Hydra output directory to specified path
        # Hydra saves to outputs/harmbench/{date}/{exp_ident}-{timestamp}/weights.pth
        # We need to find the latest output and copy it to the specified path

        # Find the latest weights.pth in the Hydra output directory
        hydra_output_base = Path(self.barriersteer_path) / "outputs" / "harmbench"
        pattern = str(hydra_output_base / "*" / f"{exp_ident}-*" / "weights.pth")
        matching_paths = glob.glob(pattern)

        if not matching_paths:
            raise RuntimeError(
                f"Trained weights not found after training. Expected pattern: {pattern}"
            )

        # Get the latest weights file
        latest_weights = max(matching_paths, key=lambda p: Path(p).stat().st_mtime)

        # Copy to specified path
        self.logger.info(
            f"Copying weights from {latest_weights} to {polytope_model_path}"
        )
        shutil.copy2(latest_weights, polytope_model_path)

        self.logger.info(
            f"Successfully trained polytope for {model_name} and saved to {polytope_model_path}"
        )
