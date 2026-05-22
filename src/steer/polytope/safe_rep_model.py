import os
import time
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import find_executable_batch_size
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from steer.cbf import (
    BatchedSafetyConstraint,
    CBFController,
    EstimatedDynamics,
    LearnedDynamics,
    SafetyConstraint,
)


def get_elapsed_time(start_time: float) -> str:
    return f"[{time.time() - start_time:.2f}s]"


def print_memory_stats(prefix=""):
    """Print current GPU memory usage with an optional prefix."""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(
        f"{prefix} GPU Memory: {allocated:.2f}GB allocated,"
        f"{reserved:.2f}GB reserved"
    )


class SafeRepModel(PreTrainedModel, GenerationMixin):
    requires_grad_for_generation = True

    def __init__(
        self,
        pretrained_model_name_or_path=None,
        steer_layer=20,
        steer_first_n_tokens=20,
        use_backup_response=False,
        lambda_weight=2.0,
        safe_violation_weight=0.0001,
        projection=False,
        backup_response="Sorry, I can't answer your request.",
        device_map="auto",
        **model_kwargs,
    ):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        super().__init__(config)
        self.model_name_or_path = pretrained_model_name_or_path

        if pretrained_model_name_or_path:
            # Filter out SafeRepModel-specific and CBF-related arguments that shouldn't be passed to the base model
            safe_rep_model_args = {
                "steer_layer",
                "steer_first_n_tokens",
                "use_backup_response",
                "lambda_weight",
                "safe_violation_weight",
                "projection",
                "backup_response",
                "polytope_weight_path",
                "use_cbf",
                "cbf_mode",
                "cbf_dt",
                "cbf_k",
                "cbf_w",
                "cbf_p",
                "cbf_kappa",
                "cbf_max_constraints",
                "cbf_constraint_mode",
                "cbf_control_radius",
                "cbf_num_steps",
                "multi_cbf_enabled",
                "multi_cbf_models_dir",
                "steering_method",
                "seed",
                "multi_cbf_load_attacks",
            }
            filtered_kwargs = {
                k: v for k, v in model_kwargs.items() if k not in safe_rep_model_args
            }
            self.lm_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                device_map=device_map,
                **filtered_kwargs,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
            )
            self.model = self.lm_model.model
            self.lm_head = self.lm_model.lm_head

        self.lm_head_device = next(self.lm_head.parameters()).device

        # Move backup_ids to the same device as lm_head
        if use_backup_response:
            self.backup_ids = self.tokenizer(
                backup_response, return_tensors="pt"
            ).input_ids.to(self.lm_head_device)

        self.phi, self.threshold, self.feature_extractor = None, None, None
        self.phi_network = None  # For neural phi polytope models
        self.rejection_logit = None
        self.backup_logit = None

        # Steering vector model (ActAdd/DirAblate)
        self.steering_vector = None
        self.steering_method = None
        self.steering_alpha = 1.0

        self.use_backup_response = use_backup_response
        self.projection = projection
        self.lambda_weight = lambda_weight
        self.safe_violation_weight = safe_violation_weight
        if use_backup_response:
            self.backup_ids = self.tokenizer(
                backup_response, return_tensors="pt"
            ).input_ids
            self.backup_logit = None

        self.safe = None
        if steer_layer is None:
            self.steer_layer = len(self.model.layers) - 1
        else:
            self.steer_layer = steer_layer
        self.steer_first_n_tokens = steer_first_n_tokens
        self.tokens_steered = 0
        self.is_new_generation = True
        # CBF steering
        self.cbf_controller: Optional[CBFController] = None
        self.cbf_constraints: List[SafetyConstraint] = []
        # Store CBF-related arguments from model_kwargs if provided
        self.cbf_mode: str = model_kwargs.get(
            "cbf_mode", "estimated"
        )  # "estimated" or "learned"
        self.cbf_dynamics: Optional[Union[EstimatedDynamics, LearnedDynamics]] = None
        self.cbf_lyapunov_ref: Optional[torch.Tensor] = None
        self.cbf_constraint_mode: str = model_kwargs.get("cbf_constraint_mode", "topk")
        # In pure QP mode, we conceptually "do not pass" a max constraint count.
        # Even if the upstream config specifies cbf_max_constraints, ignore it here.
        if self.cbf_constraint_mode == "qp":
            self.cbf_max_constraints = None
        else:
            self.cbf_max_constraints: Optional[int] = model_kwargs.get(
                "cbf_max_constraints", None
            )
        self._cbf_prev_state: Optional[torch.Tensor] = None
        # Store other CBF parameters for potential use
        self._cbf_dt = model_kwargs.get("cbf_dt", 1.0)
        self._cbf_k = model_kwargs.get("cbf_k", 1.0)
        self._cbf_w = model_kwargs.get("cbf_w", 1.0)
        self._cbf_p = model_kwargs.get("cbf_p", 10.0)
        self._cbf_kappa = model_kwargs.get("cbf_kappa", 10.0)
        self._use_cbf = model_kwargs.get("use_cbf", False)
        self.cbf_control_radius = model_kwargs.get("cbf_control_radius", None)
        self.cbf_num_steps = max(1, int(model_kwargs.get("cbf_num_steps", 1)))
        self.conditional_steering = model_kwargs.get("conditional_steering", False)

        # Load single polytope model if path is provided
        self.polytope_weight_path = model_kwargs.get("polytope_weight_path", None)
        if self.polytope_weight_path and os.path.exists(self.polytope_weight_path):
            print(f"Loading polytope model from {self.polytope_weight_path}")
            try:
                polytope_model = torch.load(
                    self.polytope_weight_path, map_location="cpu", weights_only=False
                )

                device = (
                    self.lm_head_device if hasattr(self, "lm_head_device") else "cpu"
                )

                if hasattr(polytope_model, "phi") and polytope_model.phi is not None:
                    self.phi = polytope_model.phi.to(device)
                if (
                    hasattr(polytope_model, "threshold")
                    and polytope_model.threshold is not None
                ):
                    self.threshold = polytope_model.threshold.to(device)
                if (
                    hasattr(polytope_model, "feature_extractor")
                    and polytope_model.feature_extractor is not None
                ):
                    self.feature_extractor = polytope_model.feature_extractor.to(device)
                if (
                    hasattr(polytope_model, "phi_network")
                    and polytope_model.phi_network is not None
                ):
                    self.phi_network = polytope_model.phi_network.to(device)

                # Check if this is a steering vector model
                if (
                    hasattr(polytope_model, "steering_vector")
                    and polytope_model.steering_vector is not None
                ):
                    self.steering_vector = polytope_model.steering_vector.to(device)
                    self.conditional_steering = bool(
                        model_kwargs.get(
                            "conditional_steering",
                            getattr(polytope_model, "conditional_steering", False),
                        )
                    )

                    # Persist usage of explicit steering method if provided, else use saved method, else default to 'actadd'
                    explicit_method = model_kwargs.get("steering_method", None)
                    if explicit_method is not None:
                        self.steering_method = explicit_method
                    else:
                        self.steering_method = getattr(
                            polytope_model,
                            "steering_method",
                            getattr(polytope_model, "method", "actadd"),
                        )

                    self.steering_alpha = getattr(
                        polytope_model,
                        "steering_alpha",
                        getattr(polytope_model, "alpha", 1.0),
                    )
                    print(
                        f"Loaded steering vector model: method={self.steering_method}, "
                        f"alpha={self.steering_alpha}, conditional={self.conditional_steering}"
                    )

                print(
                    f"Successfully loaded Polytope Model from {self.polytope_weight_path}"
                )
            except Exception as e:
                print(f"Error loading polytope model: {e}")

        # Check if we should auto-configure Multi-CBF
        if model_kwargs.get("multi_cbf_enabled", False):
            print("Auto-configuring Multi-CBF from initialization args...")
            self.configure_multi_cbf(model_kwargs)

    def configure_cbf(
        self,
        controller: CBFController,
        constraints: List[SafetyConstraint],
        mode: str = "estimated",
        dynamics: Optional[Union[EstimatedDynamics, LearnedDynamics]] = None,
        lyapunov_ref: Optional[torch.Tensor] = None,
        max_constraints: Optional[int] = None,
    ):
        """
        Attach a CBF controller. Training of constraints is unchanged; this only
        changes how they are enforced at inference.
        """
        self.cbf_controller = controller

        # Optimization: Automatically wrap constraints in BatchedSafetyConstraint if possible
        # Check if we have multiple constraints and they are compatible with batching
        # Specifically, we need key components like feature_extractor, threshold, etc.
        # This is a heuristic; direct construction of BatchedSafetyConstraint is preferred in configure_multi_cbf.

        # If we have a list of standard SafetyConstraints, we might leave them as is
        # unless we want to force batching here.
        # For now, we trust the caller (configure_multi_cbf) to do the wrapping,
        # but we can add a check or conversion if self.feature_extractor is available.
        # However, converting generic h_fn back to (phi, threshold) is not trivial.

        self.cbf_constraints = constraints
        self.cbf_mode = mode
        self.cbf_dynamics = dynamics
        self.cbf_lyapunov_ref = lyapunov_ref
        self.cbf_max_constraints = max_constraints

    def configure_multi_cbf(self, cbf_cfg: dict):
        """
        Configure multi-CBF by loading multiple attack-specific models and setting up ensemble constraints.

        Args:
            cbf_cfg: Configuration dict containing:
                - multi_cbf_models_dir: Directory containing weights_{attack_type}.pth files
                - cbf_dt, cbf_k, cbf_w, cbf_p, cbf_kappa: Controller parameters
                - cbf_max_constraints: Max constraints for topk/merge modes
                - cbf_control_radius: Control radius R for ||u|| <= R enforcement
        """
        import glob
        import os

        from steer.cbf import build_polytope_constraints

        models_dir = cbf_cfg.get("multi_cbf_models_dir")
        if not models_dir:
            raise ValueError("multi_cbf_models_dir must be specified")

        # Parse load_attacks filter if present
        load_attacks_json = cbf_cfg.get("multi_cbf_load_attacks")
        load_attacks = None
        if load_attacks_json:
            try:
                import json

                load_attacks = set(json.loads(load_attacks_json))
                print(f"Multi-CBF: Filtering models. Only loading: {load_attacks}")
            except Exception as e:
                print(
                    f"Warning: Failed to parse load_attacks ({load_attacks_json}): {e}"
                )

        # Get seed for filtering files (optional but recommended)
        target_seed = cbf_cfg.get("seed", None)
        if target_seed is not None:
            print(f"Multi-CBF: Filtering models by seed={target_seed}")

        # Auto-discover all weight*.pth files (handles weights_*.pth and weight_SEED_*.pth)
        pattern = os.path.join(models_dir, "weight*.pth")
        model_paths = glob.glob(pattern)

        if not model_paths:
            raise ValueError(f"No attack-specific models found in {models_dir}")

        # Parse load_attacks filter if present
        load_attacks_json = cbf_cfg.get("multi_cbf_load_attacks")
        load_attacks = None
        if load_attacks_json:
            try:
                import json

                load_attacks = set(json.loads(load_attacks_json))
                print(f"Multi-CBF: Filtering models. Only loading: {load_attacks}")
            except Exception as e:
                print(
                    f"Warning: Failed to parse load_attacks ({load_attacks_json}): {e}"
                )

        print(f"Loading attack-specific CBF models from {models_dir}")

        # Load each model and extract constraints
        all_constraints = []
        import re

        for model_path in sorted(model_paths):
            filename = os.path.basename(model_path)
            name_no_ext = filename.replace(".pth", "")
            if name_no_ext.startswith("weights_"):
                attack_type = name_no_ext[8:]
            elif name_no_ext.startswith("weight_"):
                attack_type = name_no_ext[7:]
            else:
                attack_type = name_no_ext

            # Check for seed mismatch if target_seed is provided
            # Expected format: weight_SEED_CATEGORY.pth
            seed_match = re.search(r"^weight_(\d+)_", filename)
            if seed_match:
                file_seed = int(seed_match.group(1))
                if target_seed is not None and file_seed != int(target_seed):
                    # Skip if seed doesn't match
                    # print(f"  Skipping {filename} (seed mismatch: {file_seed} != {target_seed})")
                    continue

            # Remove leading generic seed (digits + underscore) if present
            attack_type = re.sub(r"^\d+_", "", attack_type)

            # Filter check
            if load_attacks and attack_type not in load_attacks:
                # print(f"  Skipping {attack_type} (not in load list)")
                continue

            print(f"  Loading model for attack: {attack_type}")

            # Load the polytope model
            polytope_model = torch.load(model_path, weights_only=False)

            # Extract components
            feature_extractor = (
                polytope_model.feature_extractor
                if hasattr(polytope_model, "feature_extractor")
                else None
            )
            threshold = (
                polytope_model.threshold
                if hasattr(polytope_model, "threshold")
                else None
            )
            phi = polytope_model.phi if hasattr(polytope_model, "phi") else None
            phi_network = (
                polytope_model.phi_network
                if hasattr(polytope_model, "phi_network")
                else None
            )

            if (
                feature_extractor is None
                or threshold is None
                or (phi is None and phi_network is None)
            ):
                print(
                    f"  WARNING: Skipping {attack_type} - missing required components"
                )
                continue

            # Build constraints for this attack type

            # Set self.feature_extractor if not already set (use the first one found)
            if self.feature_extractor is None:
                self.feature_extractor = feature_extractor
                print(f"  Initialized self.feature_extractor from {attack_type}")

            # Build constraints for this attack type
            constraints = build_polytope_constraints(
                feature_extractor=feature_extractor,
                threshold=threshold,
                phi=phi,
                phi_network=phi_network,
                epsilon=0.0,
            )

            # constraints is a BatchedSafetyConstraint (single object), not a list
            all_constraints.append(constraints)
            print(f"  Added {len(constraints)} constraints from {attack_type}")

        if not all_constraints:
            raise ValueError(
                "No valid constraints loaded from any attack-specific models"
            )

        total_cons = sum(len(c) for c in all_constraints)
        print(f"Total constraints loaded: {total_cons}")

        # Create controller with parameters from config
        controller = CBFController(
            dt=cbf_cfg.get("cbf_dt", 1.0),
            k=cbf_cfg.get("cbf_k", 1.0),
            w=cbf_cfg.get("cbf_w", 1.0),
            p=cbf_cfg.get("cbf_p", 10.0),
            kappa=cbf_cfg.get("cbf_kappa", 10.0),
            constraint_mode=cbf_cfg.get(
                "cbf_constraint_mode", "topk"
            ),  # Pass constraint mode!
        )

        # Create dynamics
        dynamics = EstimatedDynamics(dt=cbf_cfg.get("cbf_dt", 1.0))

        # -------------------------------------------------------------
        # OPTIMIZATION: Use BatchedSafetyConstraint for fused evaluation
        # -------------------------------------------------------------

        # We need to fuse all loaded constraints into a single BatchedSafetyConstraint.
        # This requires concatenating phi/thresholds from all models.
        # Assumption: All models share the same feature_extractor structure.

        # Initialize lists to collect components
        all_thresholds = []
        all_phis = []  # Linear phi
        all_phi_networks = []  # Neural phi

        # We need to iterate model_paths again or store the components during the first loop.
        # Let's refactor the loading loop slightly, or just re-extract from the loaded constraints list?
        # Re-extracting from `build_polytope_constraints` output is hard (they are generic closures).
        # Better to collect the raw tensors during the loading loop.

        # Reset and redo the loading loop to collect raw components directly
        # (The previous implementation created a list of closures, which we can't easily merge).

        print(f"Re-loading models to build fused BatchedSafetyConstraint...")

        # Reset feature extractor to ensure it matches the fused constraint
        # self.feature_extractor = None
        # Actually, keep the one we found, but we'll verify consistency.

        fused_thresholds_list = []
        fused_phi_list = []  # For linear
        fused_phi_network_list = (
            []
        )  # For neural (not easily batchable unless we make a ModuleList)

        use_neural = False

        for model_path in sorted(model_paths):
            filename = os.path.basename(model_path)
            name_no_ext = filename.replace(".pth", "")
            if name_no_ext.startswith("weights_"):
                attack_type = name_no_ext[8:]
            elif name_no_ext.startswith("weight_"):
                attack_type = name_no_ext[7:]
            else:
                attack_type = name_no_ext

            # Keep fusion semantics identical to the diagnostic/loading pass
            # above.  Without this filter the log can report the seed-specific
            # set (e.g. 14 constraints) while the final fused constraint silently
            # includes every seed in the directory (e.g. 42 constraints).
            seed_match = re.search(r"^weight_(\d+)_", filename)
            if seed_match:
                file_seed = int(seed_match.group(1))
                if target_seed is not None and file_seed != int(target_seed):
                    continue

            attack_type = re.sub(r"^\d+_", "", attack_type)
            if load_attacks and attack_type not in load_attacks:
                continue

            polytope_model = torch.load(model_path, weights_only=False)

            fe = (
                polytope_model.feature_extractor
                if hasattr(polytope_model, "feature_extractor")
                else None
            )
            th = (
                polytope_model.threshold
                if hasattr(polytope_model, "threshold")
                else None
            )
            phi = polytope_model.phi if hasattr(polytope_model, "phi") else None
            phi_net = (
                polytope_model.phi_network
                if hasattr(polytope_model, "phi_network")
                else None
            )

            if fe is None or th is None:
                continue

            # Check consistency (roughly)
            # if self.feature_extractor is not None and type(fe) != type(self.feature_extractor):
            #     print(f"Warning: feature extractor type mismatch in {attack_type}")

            fused_thresholds_list.append(th)

            if phi_net is not None:
                use_neural = True
                fused_phi_network_list.append(phi_net)
                if phi is not None:
                    print(
                        f"Warning: {attack_type} has BOTH phi and phi_network. Using phi_network."
                    )
            elif phi is not None:
                if phi.dim() == 1:
                    phi = phi.unsqueeze(0)
                fused_phi_list.append(phi)
            else:
                print(f"Warning: {attack_type} has neither phi nor phi_network")

        # Create the fused constraint
        if not fused_thresholds_list:
            raise ValueError("No valid constraints found for fusion")

        # Concatenate thresholds
        # Note: threshold in Polytope is usually (num_constraints,)
        final_threshold = torch.cat(fused_thresholds_list, dim=0)
        print(f"Total fused constraints: {final_threshold.shape[0]}")

        final_phi = None
        final_phi_network = None

        from steer.cbf import BatchedSafetyConstraint

        if use_neural:
            # If we have mixed linear and neural, or multiple neural networks,
            # managing a single batched call is tricky without a custom container.
            # BatchedSafetyConstraint supports `phi_network` which can be a ModuleDict or similar?
            # Or we can create a wrapper module that runs all sub-networks.

            # combined_net = CombinedPhiNetwork(fused_phi_network_list, fused_phi_list)
            # Use the global class defined below
            combined_net = CombinedPhiNetwork(fused_phi_network_list, fused_phi_list)

            # Move to correct device
            device = self.lm_head_device if hasattr(self, "lm_head_device") else "cpu"
            combined_net.to(device)

            final_phi_network = combined_net

        else:
            # Pure linear case - easy fusion
            if fused_phi_list:
                final_phi = torch.cat(
                    fused_phi_list, dim=0
                )  # (total_constraints, feat_dim)
            else:
                raise ValueError("No phi components found")

        batched_constraint = BatchedSafetyConstraint(
            feature_extractor=self.feature_extractor,
            threshold=final_threshold,
            phi=final_phi,
            phi_network=final_phi_network,
            epsilon=0.0,
        )

        # Configure CBF with the single fused constraint
        self.configure_cbf(
            controller=controller,
            constraints=batched_constraint,  # Pass single object instead of list
            mode="estimated",
            dynamics=dynamics,
            max_constraints=cbf_cfg.get("cbf_max_constraints"),
        )

        # Store control radius for enforcement during steering
        self.cbf_control_radius = cbf_cfg.get("cbf_control_radius", 1.0)
        print(f"Multi-CBF configured with control_radius={self.cbf_control_radius}")

    def _apply_backup_response(
        self,
        logits: torch.Tensor,
        safe_mask: torch.Tensor,
        num_logits_to_keep: int,
        start_time: float,
    ) -> torch.Tensor:
        """Apply backup response logits to unsafe samples.

        Args:
            logits: Model output logits of shape [batch_size, seq_len,
                vocab_size]
            safe_mask: Boolean mask indicating which samples are unsafe
            num_logits_to_keep: Number of last logits to keep
            start_time: Start time for logging
        """
        if num_logits_to_keep > 0:
            logits = logits[:, -num_logits_to_keep:, :]
            current_token_idx = num_logits_to_keep - 1
        else:
            current_token_idx = -1

        # Get the current position in the backup response
        backup_position = self.tokens_steered
        backup_token_id = self.backup_ids[0, backup_position]

        current_logits = torch.zeros(
            self.model.config.vocab_size,
            device=logits.device,
            dtype=logits.dtype,
        )
        current_logits[backup_token_id] = 100.0

        # Apply modified logits
        logits[safe_mask, current_token_idx, :] = current_logits

        self.tokens_steered += 1
        if self.tokens_steered == self.backup_ids.shape[1] - 1:
            self.tokens_steered = 0
        return logits

    def _apply_optimization(
        self,
        hidden_states: torch.Tensor,
        safe_mask: torch.Tensor,
        start_time: float,
    ) -> torch.Tensor:
        """Apply hidden state optimization for unsafe samples."""
        # print("Starting optimization...")
        optimized_states = self.optimize_hidden_states(hidden_states, safe_mask)
        self.tokens_steered += 1
        return optimized_states

    def edit_logit_forward(self, input_ids, attention_mask, **kwargs):
        start_time = time.time()

        # Ensure input_ids and attention_mask are on the correct device
        input_device = input_ids.device

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=kwargs.get("position_ids"),
            past_key_values=kwargs.get("past_key_values"),
            inputs_embeds=kwargs.get("inputs_embeds"),
            use_cache=kwargs.get("use_cache"),
            output_attentions=kwargs.get("output_attentions"),
            output_hidden_states=True,
            return_dict=True,
            cache_position=kwargs.get("cache_position"),
        )

        # Get hidden states and move to lm_head device if needed
        hidden_states = outputs.hidden_states[self.steer_layer]
        token_hidden_states = hidden_states[:, -1, :].to(self.lm_head_device)

        # Check safety using features on the correct device
        features = self.feature_extractor(token_hidden_states)
        unsafe_mask = ~self.check_constraint(
            features, hidden_states=token_hidden_states
        )

        # Move last_hidden_state to lm_head device for logits computation
        last_hidden = outputs.last_hidden_state.to(self.lm_head_device)
        logits = self.lm_head(last_hidden)

        if unsafe_mask.any():
            logits = self._apply_backup_response(
                logits,
                unsafe_mask,
                kwargs.get("num_logits_to_keep", 0),
                start_time,
            )

        if kwargs.get("num_logits_to_keep", 0) > 0:
            logits = logits[:, -kwargs["num_logits_to_keep"] :, :]

        # Move outputs back to input device if needed
        if logits.device != input_device:
            logits = logits.to(input_device)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def steer_forward(self, input_ids, attention_mask, verbose=False, **args):
        """
        Hook-based steering forward pass.
        Delegates to the base model and intercepts hidden states at steer_layer via hooks.
        """
        num_logits_to_keep = args.get("num_logits_to_keep", 0)

        # Track if this is a new sequence
        if input_ids is not None and input_ids.shape[1] != 1:
            self.is_new_generation = True
            self.tokens_steered = 0

        # Initialize timing info
        if not hasattr(self, "_timing_info"):
            self._timing_info = {
                "check_overhead": 0.0,
                "solve_overhead": 0.0,
                "violated_tokens": 0,
                "not_violated_tokens": 0,
                "max_steering_norm": 0.0,
            }
        else:
            # Ensure keys exist
            expected_keys = [
                "check_overhead",
                "solve_overhead",
                "violated_tokens",
                "not_violated_tokens",
                "max_steering_norm",
            ]
            for k in expected_keys:
                if k not in self._timing_info:
                    self._timing_info[k] = 0.0

        # Flag to track if steering happened
        self._steer_applied = False

        def steering_hook(module, input, output):
            """Hook to apply steering at the steer layer."""
            # output is a tuple: (hidden_states, ...) for decoder layers
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            if self.tokens_steered >= self.steer_first_n_tokens:
                return output  # No steering after first N tokens

            # Events for precise timing
            start_check = None
            end_check = None
            start_solve = None
            end_solve = None

            if torch.cuda.is_available():
                start_check = torch.cuda.Event(enable_timing=True)
                end_check = torch.cuda.Event(enable_timing=True)
                start_solve = torch.cuda.Event(enable_timing=True)
                end_solve = torch.cuda.Event(enable_timing=True)

            # --- PHASE 1: Safety Check / Feature Extraction ---
            if start_check:
                start_check.record()
            check_start_cpu = time.perf_counter()

            # Get last token's hidden state
            token_state = hidden_states[:, -1, :]

            # Steering Vector methods (ActAdd / DirAblate / CAST)
            if self.steering_vector is not None:
                r = self.steering_vector.to(token_state.device).to(token_state.dtype)

                if self.conditional_steering:
                    if self.feature_extractor is None:
                        raise RuntimeError(
                            "Conditional steering requires a trained feature_extractor."
                        )
                    try:
                        fe_device = next(self.feature_extractor.parameters()).device
                        if fe_device != token_state.device:
                            self.feature_extractor.to(token_state.device)
                    except StopIteration:
                        self.feature_extractor.to(token_state.device)

                    features = self.feature_extractor(token_state)
                    unsafe_mask = ~self.check_constraint(
                        features, hidden_states=token_state
                    )
                    num_violated = unsafe_mask.sum().item()
                    num_not_violated = (~unsafe_mask).sum().item()
                else:
                    unsafe_mask = torch.ones(
                        token_state.shape[0],
                        dtype=torch.bool,
                        device=token_state.device,
                    )
                    num_violated = token_state.shape[0]
                    num_not_violated = 0

                if end_check:
                    end_check.record()

                if start_solve:
                    start_solve.record()

                steered_state = token_state.clone()
                if unsafe_mask.any():
                    if self.steering_method == "actadd":
                        steered_state[unsafe_mask] = (
                            token_state[unsafe_mask] + self.steering_alpha * r
                        )
                    elif self.steering_method == "dirablate":
                        unsafe_states = token_state[unsafe_mask]
                        dot_product = torch.einsum("bd,d->b", unsafe_states, r)
                        projection = dot_product.unsqueeze(-1) * r
                        steered_state[unsafe_mask] = unsafe_states - projection

                hidden_states = hidden_states.clone()
                hidden_states[:, -1, :] = steered_state
                self._steer_applied = bool(unsafe_mask.any())

                if end_solve:
                    end_solve.record()

            else:
                # Polytope / CBF Logic

                # Ensure feature extractor is on correct device
                try:
                    fe_device = next(self.feature_extractor.parameters()).device
                    if fe_device != token_state.device:
                        self.feature_extractor.to(token_state.device)
                except StopIteration:
                    self.feature_extractor.to(token_state.device)

                features = self.feature_extractor(token_state)
                unsafe_mask = ~self.check_constraint(
                    features, hidden_states=token_state
                )

                if end_check:
                    end_check.record()

                num_violated = unsafe_mask.sum().item()
                num_not_violated = (~unsafe_mask).sum().item()

                # --- PHASE 2: Solve / Correction (Only if unsafe) ---
                if start_solve:
                    start_solve.record()

                if unsafe_mask.any():
                    if self.cbf_controller is not None and (
                        len(self.cbf_constraints) > 0
                        or isinstance(self.cbf_constraints, BatchedSafetyConstraint)
                    ):
                        # CBF steering
                        if (
                            self._cbf_prev_state is None
                            or self._cbf_prev_state.shape != token_state.shape
                        ):
                            self._cbf_prev_state = token_state.detach()

                        unsafe_indices = torch.nonzero(
                            unsafe_mask, as_tuple=False
                        ).view(-1)
                        if len(unsafe_indices) > 0:
                            x_t_batch = token_state[unsafe_indices]
                            x_prev_batch = self._cbf_prev_state[unsafe_indices]

                            if self.cbf_mode == "learned" and isinstance(
                                self.cbf_dynamics, LearnedDynamics
                            ):
                                sol = self.cbf_controller.step_learned_iterative(
                                    x_t=x_t_batch,
                                    x_prev=x_prev_batch,
                                    constraints=self.cbf_constraints,
                                    dynamics=self.cbf_dynamics,
                                    lyapunov_ref=self.cbf_lyapunov_ref,
                                    max_constraints=self.cbf_max_constraints,
                                    num_steps=self.cbf_num_steps,
                                )
                            else:
                                dyn = (
                                    self.cbf_dynamics
                                    if isinstance(self.cbf_dynamics, EstimatedDynamics)
                                    else EstimatedDynamics(self.cbf_controller.dt)
                                )
                                sol = self.cbf_controller.step_estimated_iterative(
                                    x_t=x_t_batch,
                                    x_prev=x_prev_batch,
                                    constraints=self.cbf_constraints,
                                    dynamics=dyn,
                                    max_constraints=self.cbf_max_constraints,
                                    num_steps=self.cbf_num_steps,
                                )
                            x_next = sol.x_next.to(
                                device=hidden_states.device, dtype=hidden_states.dtype
                            )

                            # Calculate u_norm for logging
                            u = x_next - x_t_batch
                            u_norm = torch.norm(u, dim=-1, keepdim=True)

                            if hasattr(self, "_timing_info"):
                                current_max = u_norm.max().item()
                                if current_max > self._timing_info.get(
                                    "max_steering_norm", 0.0
                                ):
                                    self._timing_info["max_steering_norm"] = current_max

                            hidden_states = hidden_states.clone()
                            hidden_states[unsafe_indices, -1, :] = x_next
                            self._cbf_prev_state = hidden_states[:, -1, :].detach()
                    else:
                        # SAP optimization
                        optimized = self._apply_optimization(
                            hidden_states[:, -1, :], unsafe_mask, time.time()
                        )
                        hidden_states = hidden_states.clone()
                        hidden_states[:, -1, :] = optimized

                    self._steer_applied = True
                else:
                    self._cbf_prev_state = token_state.detach()

                if end_solve:
                    end_solve.record()

            self.tokens_steered += 1

            # --- Accumulate Timing ---
            if torch.cuda.is_available() and end_check and end_solve:
                end_solve.synchronize()  # Wait for everything to finish

                check_t = start_check.elapsed_time(end_check) / 1000.0  # ms -> s
                solve_t = start_solve.elapsed_time(end_solve) / 1000.0  # ms -> s

                self._timing_info["check_overhead"] += check_t
                self._timing_info["solve_overhead"] += solve_t
            else:
                # Fallback to CPU timing (approximate)
                # Note: This path is unlikely given we check cuda.is_available, but kept for robustness
                pass

            self._timing_info["violated_tokens"] += num_violated
            self._timing_info["not_violated_tokens"] += num_not_violated

            # Return modified output
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states

        # Register hook on the steer layer
        hook_handle = self.model.layers[self.steer_layer].register_forward_hook(
            steering_hook
        )

        try:
            # Delegate to the base model's forward
            outputs = self.lm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=args.get("position_ids"),
                past_key_values=args.get("past_key_values"),
                inputs_embeds=args.get("inputs_embeds"),
                use_cache=args.get("use_cache"),
                output_attentions=args.get("output_attentions", False),
                output_hidden_states=args.get("output_hidden_states", False),
                return_dict=True,
                cache_position=args.get("cache_position"),
            )
        finally:
            # Always remove the hook
            hook_handle.remove()

        # Handle num_logits_to_keep
        logits = outputs.logits
        if num_logits_to_keep > 0:
            logits = logits[:, -num_logits_to_keep:, :]

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if self.use_backup_response:
            forward_fn = self.edit_logit_forward
        else:
            forward_fn = self.steer_forward

        output = forward_fn(
            input_ids,
            attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        return output

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        return self.lm_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )

    def check_constraint(self, features, hidden_states=None):
        # 1. cbf_constraints path (Multi-CBF or configured Single-CBF)
        if (
            hasattr(self, "cbf_constraints")
            and self.cbf_constraints
            and hidden_states is not None
        ):
            if isinstance(self.cbf_constraints, BatchedSafetyConstraint):
                # Efficient check using BatchedSafetyConstraint
                # forward_b returns (batch, num_constraints)
                # Safe if all b >= 0
                b_all = self.cbf_constraints.forward_b(hidden_states)
                is_safe = (b_all >= 0).all(dim=1)  # (batch,)
                return is_safe
            else:
                is_safe = torch.ones(
                    hidden_states.shape[0],
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
                for con in self.cbf_constraints:
                    if isinstance(con, BatchedSafetyConstraint):
                        b_all = con.forward_b(hidden_states)
                        con_safe = (b_all >= 0).all(dim=1)
                        is_safe = is_safe & con_safe
                    else:
                        b_val = con.b(hidden_states).view(-1)
                        is_safe = is_safe & (b_val >= 0)
                return is_safe

        # 2. Legacy Polytope path
        # Polytope path: threshold must exist AND either phi or phi_network provided
        if (
            hasattr(self, "threshold")
            and self.threshold is not None
            and (
                (hasattr(self, "phi") and self.phi is not None)
                or (hasattr(self, "phi_network") and self.phi_network is not None)
            )
        ):
            xi = self.threshold.to(features.device).to(features.dtype)
            if hasattr(self, "phi_network") and self.phi_network is not None:
                # Ensure phi_network is on the correct device
                try:
                    net_device = next(self.phi_network.parameters()).device
                    if net_device != features.device:
                        self.phi_network.to(features.device)
                except StopIteration:
                    pass
                cost = self.phi_network(features)
            else:
                phi = self.phi.to(features.device).to(features.dtype)
                cost = torch.matmul(features, phi.T)
            constraints = cost <= xi
            return constraints.all(dim=1)

        # MLP path (fallback)
        if hasattr(self, "hidden_layer") and hasattr(self, "classifier"):
            logits = self.classifier(F.relu(self.hidden_layer(features)))
            probs = torch.sigmoid(logits)
            return (probs >= 0.5).squeeze(1)

        # If neither is available, default to allowing (avoid crash)
        warnings.warn(
            "No constraint or classifier found; defaulting to safe mask = True for all."
        )
        return torch.ones(features.size(0), dtype=torch.bool, device=features.device)

    def optimize_hidden_states(
        self,
        hidden_state,
        mask,
        num_iterations=100,
        verbose=True,
        initial_batch_size=32,
    ):
        # Add a guard to ensure we're not in inference mode
        if torch._C._get_tracing_state() or torch.is_inference_mode_enabled():
            warnings.warn(
                "optimize_hidden_states called in inference mode or while tracing. "
                "Returning original hidden states without optimization."
            )
            return hidden_state.clone()  # Return unmodified hidden states

        optimized_hidden_states = hidden_state.clone()

        unsafe_indices = torch.nonzero(mask).view(-1)

        device = hidden_state.device
        dtype = hidden_state.dtype

        # Initialize variables for nested function
        phi = None
        phi_network = None
        use_neural_phi = False
        threshold = None

        # Check if we're using polytope or MLP
        # Polytope can have either linear phi (parameter matrix) or neural phi (network)
        is_polytope = (
            hasattr(self, "threshold")
            and self.threshold is not None
            and (
                (hasattr(self, "phi") and self.phi is not None)
                or (hasattr(self, "phi_network") and self.phi_network is not None)
            )
        )

        if is_polytope:
            threshold = self.threshold.to(device).to(dtype)
            # Handle both linear phi and neural phi
            if hasattr(self, "phi") and self.phi is not None:
                phi = self.phi.to(device).to(dtype)
                use_neural_phi = False
            elif hasattr(self, "phi_network") and self.phi_network is not None:
                phi_network = self.phi_network.to(device).to(dtype)
                use_neural_phi = True
            else:
                raise ValueError("Polytope model must have either phi or phi_network")

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_extractor.eval()

        @find_executable_batch_size(starting_batch_size=initial_batch_size)
        def optimize_batch(batch_size):
            nonlocal optimized_hidden_states, unsafe_indices, phi, phi_network, use_neural_phi

            # Get total number of unsafe samples
            num_unsafe = unsafe_indices.size(0)
            rep_dim = hidden_state.size(-1)

            for start_idx in range(0, num_unsafe, batch_size):
                end_idx = min(start_idx + batch_size, num_unsafe)
                batch_indices = unsafe_indices[start_idx:end_idx]

                # Initialize optimization for this batch
                batch_hs = (
                    hidden_state[batch_indices].clone().detach().requires_grad_(True)
                )
                original_hs = hidden_state[batch_indices].clone().detach()
                optimizer = torch.optim.SGD([batch_hs], lr=0.01)

                # Optimization loop for this batch
                for iter_idx in range(num_iterations):
                    optimizer.zero_grad()
                    features = self.feature_extractor(batch_hs)
                    # Note: This line below is very important - It forces
                    # PyTorch to create a new memory for features instead of
                    # reusing the cached one, so the loss can actually go down.

                    if is_polytope:
                        # features shape: (actual_batch_size, feature_dim)
                        if use_neural_phi:
                            # Use neural phi network
                            cost = phi_network(features)  # [batch, num_phi]
                            violations = cost - threshold  # threshold broadcasts
                        else:
                            # Use linear phi matrix
                            # features: (batch, feature_dim), phi.T: (feature_dim, num_phi)
                            violations = (
                                torch.matmul(features, phi.T)
                                - threshold  # threshold broadcasts
                            )
                        safety_violation = violations.sum()
                        positive_violations = torch.relu(violations)
                        positive_violation_penalty = torch.sum(positive_violations)
                    else:
                        # MLP optimization
                        hidden_output = F.relu(self.hidden_layer(features))
                        logits = self.classifier(hidden_output)
                        probs = torch.sigmoid(logits)
                        # For MLP, we want to maximize the probability of being safe
                        safety_violation = -torch.sum(probs)
                        positive_violations = (
                            -logits
                        )  # Negative logits indicate unsafe predictions
                        positive_violation_penalty = torch.sum(
                            torch.relu(positive_violations)
                        )

                    distance = torch.sum(torch.abs(batch_hs - original_hs))

                    loss = (
                        distance / rep_dim
                        + self.safe_violation_weight * safety_violation
                        + self.lambda_weight * positive_violation_penalty
                    )

                    # print(f"Loss: {loss.item()}")

                    # Check if loss is inf or NaN before backpropagation
                    if torch.isinf(loss).any() or torch.isnan(loss).any():
                        print(
                            f"Warning: Detected inf/NaN in loss at iteration {iter_idx}. Skipping optimization."
                        )
                        # Reset this batch to original hidden states
                        with torch.no_grad():
                            batch_hs.copy_(original_hs)
                        break  # Exit the optimization loop for this batch
                    else:
                        loss.backward()
                        optimizer.step()

                # After optimization, update the main tensor
                with torch.no_grad():
                    features = self.feature_extractor(batch_hs)

                    if is_polytope:
                        if use_neural_phi:
                            # Use neural phi network
                            cost = phi_network(features)  # [batch, num_phi]
                            final_violations = cost - threshold
                        else:
                            # Use linear phi matrix
                            final_violations = torch.matmul(features, phi.T) - threshold
                        final_positive_violations = torch.relu(final_violations)
                        final_safety_violation = final_violations.sum()
                    else:
                        hidden_output = F.relu(self.hidden_layer(features))
                        logits = self.classifier(hidden_output)
                        probs = torch.sigmoid(logits)
                        final_violations = -logits
                        final_positive_violations = torch.relu(final_violations)
                        final_safety_violation = -torch.sum(probs)

                    final_distance = torch.sum(torch.abs(batch_hs - original_hs))

                    optimized_hidden_states[batch_indices] = batch_hs.detach()

                    del features, final_violations, final_positive_violations
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            return optimized_hidden_states

        with torch.set_grad_enabled(True):
            result = optimize_batch()
            return result

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        polytope_weight_path=None,
        **kwargs,
    ):
        # Create the base model first
        safe_model = cls(pretrained_model_name_or_path, **kwargs)

        if polytope_weight_path is not None:
            # Load the saved safety model
            safety_model = torch.load(polytope_weight_path, weights_only=False)

            # Check if we loaded a PolytopeConstraint or BaselineMLP
            if (
                hasattr(safety_model, "steering_vector")
                and safety_model.steering_vector is not None
            ):
                # Steering vector / CAST model
                safe_model.steering_vector = safety_model.steering_vector
                safe_model.steering_method = getattr(
                    safety_model,
                    "steering_method",
                    getattr(safety_model, "method", "actadd"),
                )
                safe_model.steering_alpha = getattr(
                    safety_model,
                    "steering_alpha",
                    getattr(safety_model, "alpha", 1.0),
                )
                safe_model.conditional_steering = bool(
                    getattr(safety_model, "conditional_steering", False)
                )
                if hasattr(safety_model, "phi"):
                    safe_model.phi = safety_model.phi
                if hasattr(safety_model, "phi_network"):
                    safe_model.phi_network = safety_model.phi_network
                if hasattr(safety_model, "threshold"):
                    safe_model.threshold = safety_model.threshold
                if hasattr(safety_model, "feature_extractor"):
                    safe_model.feature_extractor = safety_model.feature_extractor
                print(
                    f"Loaded steering vector model: method={safe_model.steering_method}, "
                    f"alpha={safe_model.steering_alpha}, conditional={safe_model.conditional_steering}"
                )
            elif hasattr(safety_model, "threshold"):
                # Polytope model (can have either phi parameter or phi_network)
                if hasattr(safety_model, "phi"):
                    safe_model.phi = safety_model.phi
                if hasattr(safety_model, "phi_network"):
                    safe_model.phi_network = safety_model.phi_network
                safe_model.threshold = safety_model.threshold
                safe_model.feature_extractor = safety_model.feature_extractor
                model_type = (
                    "neural polytope"
                    if hasattr(safety_model, "phi_network")
                    and safety_model.phi_network is not None
                    else "polytope"
                )
                print(f"Loaded {model_type} constraint model")
            else:
                # MLP model
                safe_model.feature_extractor = safety_model.feature_extractor
                safe_model.hidden_layer = safety_model.hidden_layer
                safe_model.classifier = safety_model.classifier
                print("Loaded MLP classifier model")

        # Move all components to the correct device and dtype
        device = next(safe_model.model.parameters()).device
        dtype = next(safe_model.model.parameters()).dtype

        # Move feature extractor
        if hasattr(safe_model, "feature_extractor") and isinstance(
            safe_model.feature_extractor, nn.Module
        ):
            safe_model.feature_extractor.to(device=device, dtype=dtype)
            safe_model.feature_extractor.eval()  # Disable dropout during inference

        # Move phi and threshold for polytope models
        if hasattr(safe_model, "phi") and safe_model.phi is not None:
            safe_model.phi = nn.Parameter(safe_model.phi.to(device=device, dtype=dtype))
        if hasattr(safe_model, "phi_network") and safe_model.phi_network is not None:
            safe_model.phi_network.to(device=device, dtype=dtype)
            safe_model.phi_network.eval()  # Disable dropout during inference
        if hasattr(safe_model, "threshold") and safe_model.threshold is not None:
            safe_model.threshold = nn.Parameter(
                safe_model.threshold.to(device=device, dtype=dtype)
            )

        # Move hidden_layer and classifier for MLP models
        if hasattr(safe_model, "hidden_layer"):
            safe_model.hidden_layer.to(device=device, dtype=dtype)
        if hasattr(safe_model, "classifier"):
            safe_model.classifier.to(device=device, dtype=dtype)

        # Store device information for easier access
        safe_model.lm_head_device = device

        print(f"Feature extractor type: {type(safe_model.feature_extractor)}")
        print(f"Polytope weight path: {polytope_weight_path}")
        print(f"Model device: {device}")

        # Auto-configure CBF if requested (AFTER loading weights)
        if getattr(safe_model, "_use_cbf", False):
            print("Auto-configuring CBF controller from pretrained...")
            try:
                controller = CBFController(
                    dt=safe_model._cbf_dt,  # Access via instance since we are in classmethod
                    k=safe_model._cbf_k,
                    w=safe_model._cbf_w,
                    p=safe_model._cbf_p,
                    kappa=safe_model._cbf_kappa,
                    constraint_mode=safe_model.cbf_constraint_mode,
                )

                constraints = []
                if safe_model.threshold is not None and (
                    safe_model.phi is not None or safe_model.phi_network is not None
                ):
                    # Construct BatchedSafetyConstraint
                    # Ensure we have a feature extractor
                    fe = (
                        safe_model.feature_extractor
                        if safe_model.feature_extractor is not None
                        else nn.Identity()
                    )
                    # feature_extractor is already moved to device above

                    constr = BatchedSafetyConstraint(
                        feature_extractor=fe,
                        threshold=safe_model.threshold,
                        phi=safe_model.phi,
                        phi_network=safe_model.phi_network,
                        epsilon=(
                            safe_model.safe_margin
                            if hasattr(safe_model, "safe_margin")
                            and safe_model.safe_margin is not None
                            else 0.0
                        ),
                    )
                    constraints.append(constr)
                    print(
                        f"Converted loaded polytope to BatchedSafetyConstraint (num={len(constr)})"
                    )

                safe_model.configure_cbf(
                    controller=controller,
                    constraints=constraints,
                    mode=safe_model.cbf_mode,
                    dynamics=None,  # Will be estimated by default if None
                    lyapunov_ref=safe_model.cbf_lyapunov_ref,
                    max_constraints=safe_model.cbf_max_constraints,
                )
                print("CBF Controller configured successfully.")
            except Exception as e:
                print(f"Failed to auto-configure CBF: {e}")

        return safe_model


class SafeRepConfig(PretrainedConfig):
    model_type = "safe_rep_model"

    def __init__(
        self,
        pretrained_model_path=None,
        polytope_weight_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pretrained_model_path = pretrained_model_path
        self.polytope_weight_path = polytope_weight_path
        self.model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }


class CombinedPhiNetwork(nn.Module):
    def __init__(self, networks: List[nn.Module], linear_phis: List[torch.Tensor]):
        super().__init__()
        self.networks = nn.ModuleList(networks)
        # Treat linear phis as buffers
        self.linear_phis_tensors = linear_phis

    def forward(self, x):
        results = []
        # Run neural networks
        for net in self.networks:
            res = net(x)  # (batch, num_constraints_i)
            results.append(res)

        # Run linear phis
        for phi in self.linear_phis_tensors:
            phi = phi.to(x.device)
            res = torch.matmul(x, phi.T)
            results.append(res)

        return torch.cat(results, dim=1)

    def forward_subset(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Evaluate only selected constraint logits.
        Falls back to full forward + gather when subset dispatch isn't available.
        """
        if indices.numel() == 0:
            return x.new_zeros((x.shape[0], 0))

        # Build contiguous [start, end) slices for each submodule output block.
        blocks = []
        cursor = 0
        for net in self.networks:
            if hasattr(net, "num_phi"):
                width = int(net.num_phi)
            elif hasattr(net, "final") and hasattr(net.final, "out_features"):
                width = int(net.final.out_features)
            elif hasattr(net, "net") and len(list(net.net.children())) > 0:
                last = list(net.net.children())[-1]
                if isinstance(last, nn.Linear):
                    width = int(last.out_features)
                else:
                    penultimate = list(net.net.children())[-2]
                    width = int(penultimate.out_features)
            else:
                width = int(net(x[:1]).shape[1])
            blocks.append((cursor, cursor + width, net, "net"))
            cursor += width

        for phi in self.linear_phis_tensors:
            width = int(phi.shape[0])
            blocks.append((cursor, cursor + width, phi, "linear"))
            cursor += width

        batch_size, k = indices.shape
        out = x.new_empty((batch_size, k))

        for start, end, block_obj, block_type in blocks:
            mask = (indices >= start) & (indices < end)
            if not mask.any():
                continue

            local_indices = torch.where(
                mask, indices - start, torch.zeros_like(indices)
            )

            if block_type == "net":
                if hasattr(block_obj, "forward_subset"):
                    block_vals = block_obj.forward_subset(x, local_indices)
                else:
                    block_all = block_obj(x)
                    block_vals = torch.gather(block_all, 1, local_indices)
            else:
                phi = block_obj.to(x.device)
                block_all = torch.matmul(x, phi.T)
                block_vals = torch.gather(block_all, 1, local_indices)

            out = torch.where(mask, block_vals, out)

        return out
