import argparse
import os
import sys
import time

import numpy as np
import torch
import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from barriersteer_pipeline.data.safety_data import get_hidden_states_dataloader
from barriersteer_pipeline.harmbench.utils.config_utils import load_config
from steer.cbf import EstimatedDynamics, LearnedDynamics
from steer.polytope.safe_rep_model import BatchedSafetyConstraint, SafeRepModel


def benchmark(args):
    # 1. Load Config
    config = load_config(args.config)

    # Extract model config
    model_config_all = {m["name"]: m for m in config.get("models", [])}
    if args.model not in model_config_all:
        raise ValueError(f"Model {args.model} not found in config {args.config}")

    model_config = model_config_all[args.model]

    # 2. Setup Model Args
    # Filter modes if specified
    mode_name = (
        args.mode if args.mode else "sap"
    )  # Default to 'sap' if not specified, but check config
    if "steering" in model_config and "modes" in model_config["steering"]:
        if mode_name not in model_config["steering"]["modes"]:
            print(
                f"Warning: Mode {mode_name} not found in config modes for {args.model}. Using available keys if possible."
            )

        # Merge mode-specific config into top-level kwargs for SafeRepModel
        mode_cfg = model_config["steering"]["modes"].get(mode_name, {})
        # If mode config has specific sub-sections like 'cbf', flatten them somewhat or pass them

        # We need to construct arguments compatible with SafeRepModel.from_pretrained
        # Common args
        model_kwargs = {
            "pretrained_model_name_or_path": model_config["path"],
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }

        # CBF params
        model_kwargs["use_cbf"] = mode_cfg.get("cbf", {}).get("use", False)
        model_kwargs["cbf_mode"] = mode_cfg.get("cbf", {}).get("mode", "estimated")

        # Map CLI args to CBF config keys
        cbf_cfg = mode_cfg.get("cbf", {})

        # Apply CLI overrides to cbf_cfg copy to pass to model
        if args.cbf_k is not None:
            cbf_cfg["k"] = args.cbf_k
        if args.cbf_dt is not None:
            cbf_cfg["dt"] = args.cbf_dt
        if args.cbf_w is not None:
            cbf_cfg["w"] = args.cbf_w
        if args.cbf_p is not None:
            cbf_cfg["p"] = args.cbf_p
        if args.cbf_kappa is not None:
            cbf_cfg["kappa"] = args.cbf_kappa
        if args.cbf_max_constraints is not None:
            cbf_cfg["max_constraints"] = (
                None if args.cbf_max_constraints <= 0 else args.cbf_max_constraints
            )
        if args.cbf_num_steps is not None:
            cbf_cfg["num_steps"] = args.cbf_num_steps
        if args.control_radius is not None:
            cbf_cfg["control_radius"] = args.control_radius

        # Flatten into model_kwargs for SafeRepModel
        model_kwargs["cbf_k"] = cbf_cfg.get("k", 1.0)
        model_kwargs["cbf_dt"] = cbf_cfg.get("dt", 1.0)
        model_kwargs["cbf_w"] = cbf_cfg.get("w", 1.0)
        model_kwargs["cbf_p"] = cbf_cfg.get("p", 10.0)
        model_kwargs["cbf_kappa"] = cbf_cfg.get("kappa", 10.0)
        max_constraints = cbf_cfg.get("max_constraints", None)
        if max_constraints is not None:
            max_constraints = int(max_constraints)
            if max_constraints <= 0:
                max_constraints = None
        model_kwargs["cbf_max_constraints"] = max_constraints
        model_kwargs["cbf_control_radius"] = cbf_cfg.get("control_radius", None)

        model_kwargs["cbf_constraint_mode"] = cbf_cfg.get("constraint_mode", "topk")
        model_kwargs["cbf_num_steps"] = max(
            1, int(cbf_cfg.get("num_steps", cbf_cfg.get("cbf_num_steps", 1)))
        )

        if "multi_cbf" in cbf_cfg and cbf_cfg["multi_cbf"].get("enabled", False):
            model_kwargs["multi_cbf_enabled"] = True
            model_kwargs["multi_cbf_models_dir"] = cbf_cfg["multi_cbf"].get(
                "models_dir"
            )
            model_kwargs["multi_cbf_load_attacks"] = cbf_cfg["multi_cbf"].get(
                "load_attacks"
            )

        # Polytope path
        poly_path = args.polytope_model_path
        if poly_path is None:
            # Attempt auto-discovery
            # Pattern: outputs/harmbench/{model_name}/{mode_suffix}/weights.pth
            # mode_suffix typically depends on mode.
            # For 'cbf_topk_nonlinear', it might include phi count e.g. 'cbf_topk_nonlinear_phi004'
            # This is hard to guess exactly without parsing logs or configs deeper.
            # However, we can try to find *any* matching weights in the output dir for this model.

            base_out = f"outputs/harmbench/{args.model}"
            if os.path.exists(base_out):
                # Construct expected suffix prefix
                # e.g. cbf_topk_nonlinear -> cbf_topk_nonlinear
                # But typically folders are appended with _phiXXX

                # Check if we should reuse weights from another mode
                target_mode = mode_cfg.get("reuse_weights_from", mode_name)
                modes_to_check = [target_mode]
                if target_mode in ["cbf_qp_nonlinear", "cbf_merge_nonlinear"]:
                    modes_to_check.append("cbf_topk_nonlinear")

                import glob

                for m_check in modes_to_check:
                    candidates = glob.glob(os.path.join(base_out, f"{m_check}*"))
                    if candidates:
                        # Check candidates for weights
                        for cand in sorted(candidates, reverse=True):
                            # Priority 1: weights.pth
                            w_path = os.path.join(cand, "weights.pth")
                            if os.path.exists(w_path):
                                poly_path = w_path
                                break

                            # Priority 2: weight_*.pth (e.g. weight_42.pth) - pick first found (sorted)
                            w_cands = glob.glob(os.path.join(cand, "weight_*.pth"))
                            if w_cands:
                                w_cands.sort(reverse=True)  # Deterministic pick
                                poly_path = w_cands[0]
                                break

                    if poly_path is not None:
                        print(
                            f"Auto-detected polytope weights in mode {m_check}: {poly_path}"
                        )
                        break

        if poly_path is None and not model_kwargs.get("multi_cbf_enabled", False):
            print(
                "Warning: No polytope weight path provided and auto-discovery failed. Model might have no constraints!"
            )

        model_kwargs["polytope_weight_path"] = poly_path

        # SaP specific
        if args.unsafe_weight is not None:
            model_kwargs["safe_violation_weight"] = args.unsafe_weight
        elif "unsafe_weight" in mode_cfg:
            model_kwargs["safe_violation_weight"] = mode_cfg["unsafe_weight"]

        if args.lambda_weight is not None:
            model_kwargs["lambda_weight"] = args.lambda_weight
        elif "lambda_weight" in mode_cfg:
            model_kwargs["lambda_weight"] = mode_cfg["lambda_weight"]

    else:
        # Generic fallback
        model_kwargs = {
            "pretrained_model_name_or_path": model_config["path"],
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        if args.polytope_model_path:
            model_kwargs["polytope_weight_path"] = args.polytope_model_path

    print(f"Loading model with kwargs keys: {list(model_kwargs.keys())}")

    # 3. Load Model
    model = SafeRepModel.from_pretrained(**model_kwargs)

    # 4. Load Data
    hs_path = args.hidden_states_path
    if hs_path is None:
        # Try to find from config
        if (
            "polytope_training" in model_config
            and "dataset" in model_config["polytope_training"]
        ):
            hs_path = model_config["polytope_training"]["dataset"]["hidden_states_path"]
            # If relative, prepend workspace root?
            if not os.path.isabs(hs_path):
                workspace_root = os.environ.get("WORKSPACE_ROOT", ".")
                hs_path = os.path.join(workspace_root, hs_path)

    if hs_path is None:
        raise ValueError("Hidden states path not provided and not found in config.")

    print(f"Loading hidden states from {hs_path}")

    data_dict = {}

    if os.path.isfile(hs_path):
        # User provided specific file (e.g. GCG_test.pt)
        print(f"Loading file: {hs_path}")
        try:
            loaded_data = torch.load(hs_path, map_location="cpu", weights_only=False)
            # Infer method name from filename if possible, else 'default'
            basename = os.path.basename(hs_path)
            method_name = basename.split("_")[0] if "_" in basename else "default"
            data_dict[method_name] = loaded_data
        except Exception as e:
            raise ValueError(f"Failed to load data file {hs_path}: {e}")

    elif os.path.isdir(hs_path):
        # Check if user wants specific split or just load all *test.pt
        # Default logic: look for GCG_test.pt as per hint, or all *_test.pt

        # Priority: GCG_test.pt -> any *_test.pt
        gcg_path = os.path.join(hs_path, "GCG_test.pt")
        if os.path.exists(gcg_path):
            print("Found GCG_test.pt, loading...")
            data_dict["GCG"] = torch.load(
                gcg_path, map_location="cpu", weights_only=False
            )
        else:
            # Load all *_test.pt
            import glob

            test_files = glob.glob(os.path.join(hs_path, "*_test.pt"))
            if not test_files:
                # Fallback to *_train.pt
                test_files = glob.glob(os.path.join(hs_path, "*_train.pt"))
                if not test_files:
                    # Try just any .pt?
                    test_files = glob.glob(os.path.join(hs_path, "*.pt"))

            if not test_files:
                raise ValueError(f"No .pt files found in {hs_path}")

            print(f"Found {len(test_files)} data files. Loading...")
            for fpath in test_files:
                basename = os.path.basename(fpath)
                method = basename.split("_")[0]
                # print(f"  Loading {basename} ({method})...")
                try:
                    data_dict[method] = torch.load(
                        fpath, map_location="cpu", weights_only=False
                    )
                except Exception as e:
                    print(f"  Warning: Failed to load {basename}: {e}")

    else:
        raise ValueError(f"Path {hs_path} does not exist.")

    if not data_dict:
        raise ValueError("No valid data loaded.")

    dataloader = get_hidden_states_dataloader(
        data_dict, batch_size=args.batch_size, shuffle=True
    )

    # 5. Benchmarking Loop
    print("Starting benchmark...")

    total_violated_tokens = 0
    steer_times_ms = []  # List to store time per batch
    total_steered_batches = 0

    # CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    model.eval()

    processed_batches = 0
    with torch.no_grad():
        for batch_idx, (hidden_states, labels) in enumerate(tqdm.tqdm(dataloader)):
            if args.limit_batches and processed_batches >= args.limit_batches:
                break

            hidden_states = hidden_states.to(model.lm_head_device).to(model.dtype)

            # --- Check Phase ---
            # Manually trigger check logic to get unsafe mask
            # We want to use model components directly

            # Note: SafeRepModel.check_constraint expects (features, hidden_states) usually
            # But typically hidden_states ARE the features for SaP unless feature_extractor is complex

            # We need to manually run feature extractor
            if model.feature_extractor is not None:
                features = model.feature_extractor(hidden_states)
            else:
                features = hidden_states

            unsafe_mask = ~model.check_constraint(features, hidden_states=hidden_states)

            num_unsafe = unsafe_mask.sum().item()

            if num_unsafe > 0:
                # --- Warmup Check ---
                is_warmup = processed_batches < args.warmup

                # --- Steering Phase ---
                # Prepare args for steering
                unsafe_indices = torch.nonzero(unsafe_mask).view(-1)

                start_event.record()

                # Run steering logic
                # We need to mimic steer_forward logic
                if model.cbf_controller is not None:
                    # CBF path
                    if (
                        model._cbf_prev_state is None
                        or model._cbf_prev_state.shape != hidden_states.shape
                    ):
                        model._cbf_prev_state = (
                            hidden_states.detach()
                        )  # Mock prev state

                    x_t_batch = hidden_states[unsafe_indices]
                    x_prev_batch = model._cbf_prev_state[unsafe_indices]

                    if model.cbf_mode == "learned" and isinstance(
                        model.cbf_dynamics, LearnedDynamics
                    ):
                        sol = model.cbf_controller.step_learned_iterative(
                            x_t=x_t_batch,
                            x_prev=x_prev_batch,
                            constraints=model.cbf_constraints,
                            dynamics=model.cbf_dynamics,
                            lyapunov_ref=model.cbf_lyapunov_ref,
                            max_constraints=model.cbf_max_constraints,
                            num_steps=max(1, int(getattr(model, "cbf_num_steps", 1))),
                        )
                    else:
                        dyn = (
                            model.cbf_dynamics
                            if isinstance(model.cbf_dynamics, EstimatedDynamics)
                            else EstimatedDynamics(model.cbf_controller.dt)
                        )
                        sol = model.cbf_controller.step_estimated_iterative(
                            x_t=x_t_batch,
                            x_prev=x_prev_batch,
                            constraints=model.cbf_constraints,
                            dynamics=dyn,
                            max_constraints=model.cbf_max_constraints,
                            num_steps=max(1, int(getattr(model, "cbf_num_steps", 1))),
                        )
                    x_next = sol.x_next  # noqa: F841

                else:
                    # SaP path
                    # optimize_hidden_states(self, hidden_state, mask, num_iterations=100, ...)
                    optimized = model.optimize_hidden_states(
                        hidden_states, unsafe_mask, num_iterations=args.num_iterations
                    )

                end_event.record()
                end_event.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)

                if not is_warmup:
                    steer_times_ms.append(elapsed_ms)
                    total_violated_tokens += num_unsafe
                    total_steered_batches += 1
                elif processed_batches == args.warmup - 1:
                    # Reset controller profiling after warmup completes
                    if (
                        hasattr(model, "cbf_controller")
                        and model.cbf_controller is not None
                    ):
                        model.cbf_controller.reset_profiling_stats()

            processed_batches += 1

    # 6. Report
    print("\n--- Benchmark Results ---")
    print(f"Total Batches Processed: {processed_batches}")
    print(f"Total Steered Batches (excluding warmup): {total_steered_batches}")
    print(f"Total Violated Tokens: {total_violated_tokens}")

    results = {
        "model": args.model,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "limit_batches": args.limit_batches,
        "total_batches": processed_batches,
        "total_steered_batches": total_steered_batches,
        "total_violated_tokens": total_violated_tokens,
        "polytope_weight_path": args.polytope_model_path,
    }

    if steer_times_ms:
        avg_time_ms = np.mean(steer_times_ms)
        std_time_ms = np.std(steer_times_ms)
        total_time_ms = np.sum(steer_times_ms)

        print(f"Steering Time per Batch: {avg_time_ms:.4f} ms ± {std_time_ms:.4f} ms")
        results["steer_avg_ms"] = float(avg_time_ms)
        results["steer_std_ms"] = float(std_time_ms)

        if total_violated_tokens > 0:
            avg_time_per_token = total_time_ms / total_violated_tokens
            print(
                f"Average Steering Time per Violated Token: {avg_time_per_token:.4f} ms"
            )
            results["steer_per_token_ms"] = float(avg_time_per_token)
    else:
        print("No violated tokens found during non-warmup matches.")

    # Print granular profiling stats if available
    if hasattr(model, "cbf_controller") and model.cbf_controller is not None:
        if hasattr(model.cbf_controller, "print_profiling_stats"):
            model.cbf_controller.print_profiling_stats()
        # Save phase timings to results
        if hasattr(model.cbf_controller, "phase_timings"):
            for phase, vals in model.cbf_controller.phase_timings.items():
                arr = np.array(vals)
                results[f"phase_{phase}_avg_ms"] = float(np.mean(arr))
                results[f"phase_{phase}_std_ms"] = float(np.std(arr))

    # Count CBF network parameters
    cbf_params = 0
    cbf_network_desc = "N/A"
    if hasattr(model, "cbf_constraints"):
        con = model.cbf_constraints
        if isinstance(con, BatchedSafetyConstraint):
            if con.phi_network is not None:
                cbf_params = sum(p.numel() for p in con.phi_network.parameters())
                cbf_network_desc = str(con.phi_network)
            num_constraints = len(con)
            results["num_phi"] = num_constraints
        elif isinstance(con, list):
            for c in con:
                if isinstance(c, BatchedSafetyConstraint) and c.phi_network is not None:
                    cbf_params += sum(p.numel() for p in c.phi_network.parameters())
    results["cbf_params"] = cbf_params

    print("\n--- Model Inspection ---")
    print(f"CBF Network Parameters: {cbf_params:,}")
    if model.feature_extractor is not None:
        print(f"Feature Extractor: {model.feature_extractor}")
        if (
            isinstance(model.feature_extractor, torch.nn.Sequential)
            and len(model.feature_extractor) == 0
        ):
            print("Note: Feature Extractor is an EMPTY Sequential module (Identity).")
    else:
        print("Feature Extractor: None")

    # Save results to JSON if output path is provided
    if args.output_json:
        import json

        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Steering Latency")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to pipeline config"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name in config")
    parser.add_argument(
        "--polytope_model_path", type=str, help="Path to trained polytope weights"
    )
    parser.add_argument(
        "--hidden_states_path", type=str, help="Override hidden states path"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup batches")
    parser.add_argument("--limit_batches", type=int, default=1000, help="Limit batches")
    parser.add_argument(
        "--mode", type=str, default="sap", help="Mode (sap, cbf_topk_nonlinear, etc)"
    )

    # Overrides
    parser.add_argument("--cbf_k", type=float)
    parser.add_argument("--cbf_dt", type=float)
    parser.add_argument("--cbf_w", type=float)
    parser.add_argument("--cbf_p", type=float)
    parser.add_argument("--cbf_kappa", type=float)
    parser.add_argument("--cbf_max_constraints", type=int)
    parser.add_argument("--cbf_num_steps", type=int)
    parser.add_argument("--control_radius", type=float)
    parser.add_argument(
        "--unsafe_weight",
        "--safe_weight",
        type=float,
        help="Safe violation weight for SaP",
    )
    parser.add_argument("--lambda_weight", type=float, help="Lambda weight for SaP")
    parser.add_argument(
        "--num_iterations", type=int, default=100, help="Number of iterations for SaP"
    )
    parser.add_argument(
        "--output_json", type=str, default=None, help="Path to save results as JSON"
    )

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Configure stdout to unbuffered
    sys.stdout.reconfigure(line_buffering=True)

    benchmark(args)
