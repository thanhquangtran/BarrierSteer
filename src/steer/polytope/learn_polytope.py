import logging
import os

import hydra
import numpy as np
import torch
import torch.optim as optim
import tqdm
from omegaconf import DictConfig

from barriersteer_pipeline.data.safety_data import get_hidden_states_dataloader
from steer.common.load_util import set_seed
from steer.common.outputs import ModelResult, evaluate_model
from steer.polytope.lm_constraints import BaselineMLP, PolytopeConstraint
from steer.polytope.steering_vector import SteeringVectorModel

log = logging.getLogger("polytope")


def report_loss_callback(epoch, batch, avg_loss):
    loss_str = f"loss: {avg_loss:.3f}"
    log.info(f"[Epoch {epoch+1}, Batch {batch+1}] {loss_str}")


def train_model(
    model,
    dataloader,
    optimizer,
    num_epochs=1,
    disable_tqdm=False,
    callback_every=100,
    callback_fn=report_loss_callback,
):
    model.train()
    model.to(model.device)
    latest_loss = 0
    running_additional_params = {}

    for epoch in range(num_epochs):
        running_loss = []
        with tqdm.tqdm(total=len(dataloader), disable=disable_tqdm) as pbar:
            for i, (inputs, labels) in enumerate(dataloader):
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(model.device)
                labels = labels.float().to(model.device)
                outputs = model(inputs, labels)
                loss = outputs.loss
                additional_params = outputs.additional_params

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)

                running_loss.append(loss.item())

                # Update running averages for additional params
                if additional_params:
                    for key, value in additional_params.items():
                        if key not in running_additional_params:
                            running_additional_params[key] = []
                        running_additional_params[key].append(value)

                if callback_fn and i % callback_every == 0:
                    latest_loss = np.mean(running_loss)
                    callback_fn(epoch, i, latest_loss)

                    # Print running averages of additional params
                    log.info("Running averages of additional parameters:")
                    for key, values in running_additional_params.items():
                        avg_value = np.mean(values)
                        log.info(f"  {key}: {avg_value:.4f}")

                    # Reset running averages after reporting
                    running_loss = []
                    running_additional_params = {}

    latest_loss = np.mean(running_loss)
    return latest_loss.item()


def get_model_input_dim(model_path):
    model_path = model_path.lower()
    if "llama" in model_path or "mistral" in model_path:
        return 4096
    elif "qwen" in model_path:
        return 1536
    else:
        raise NotImplementedError(
            f"Input dimension not defined for model: {model_path}"
        )


@hydra.main(
    config_path=f"{os.getcwd()}/exp_configs",
    config_name="polytope_config",
    version_base="1.1",
)
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    # Get attack methods from config (if specified)
    attack_methods = cfg.dataset.get("attack_methods", None)
    if isinstance(attack_methods, str) and attack_methods == "None":
        attack_methods = None

    # Load hidden states from directory structure
    # Expected structure: {data_dir}/{method}_train.pt, {method}_test.pt
    data_dir = cfg.dataset.hidden_states_path
    log.info(f"Loading hidden states from directory: {data_dir}")

    # Find all available methods in the directory
    import glob

    train_files = glob.glob(os.path.join(data_dir, "*_train.pt"))
    available_methods = [
        os.path.basename(f).replace("_train.pt", "") for f in train_files
    ]
    log.info(f"Available attack methods in directory: {available_methods}")

    # Determine which methods to load
    if attack_methods:
        methods_to_load = [m for m in attack_methods if m in available_methods]
        missing_methods = [m for m in attack_methods if m not in available_methods]
        if missing_methods:
            log.warning(f"Requested methods not found: {missing_methods}")
        if not methods_to_load:
            raise ValueError(
                f"None of the requested attack methods {attack_methods} found in {data_dir}. "
                f"Available: {available_methods}"
            )
        log.info(f"Using configured attack methods for training: {methods_to_load}")
    else:
        methods_to_load = available_methods
        log.info(f"Using all available attack methods for training: {methods_to_load}")

    hs_data = {"train": {}, "test": {}}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def load_method(method, data_dir):
        """Load train and test files for a single method"""
        train_file = os.path.join(data_dir, f"{method}_train.pt")
        test_file = os.path.join(data_dir, f"{method}_test.pt")

        log.info(f"Loading {method}...")
        train_data = torch.load(train_file, weights_only=False, mmap=True)
        test_data = torch.load(test_file, weights_only=False, mmap=True)
        return method, train_data, test_data

    # Load methods in parallel (up to 8 at a time for faster loading)
    with ThreadPoolExecutor(max_workers=min(8, len(methods_to_load))) as executor:
        futures = {
            executor.submit(load_method, method, data_dir): method
            for method in methods_to_load
        }
        for future in as_completed(futures):
            method, train_data, test_data = future.result()
            hs_data["train"][method] = train_data
            hs_data["test"][method] = test_data
            log.info(f"Stored {method} in memory")

    log.info(f"Loaded {len(methods_to_load)} methods successfully")

    # Calculate and log label statistics
    total_label_0 = 0
    total_label_1 = 0

    log.info("=" * 40)
    log.info("Data Label Statistics (Train):")
    for method, data in hs_data["train"].items():
        labels = data["labels"]
        # Assuming labels are 0 and 1
        num_0 = (labels == 0).sum().item()
        num_1 = (labels == 1).sum().item()

        log.info(f"Method: {method}")
        log.info(f"  Label 0 (Harmful/Unsafe): {num_0}")
        log.info(f"  Label 1 (Safe): {num_1}")

        total_label_0 += num_0
        total_label_1 += num_1

    log.info("-" * 20)
    log.info("Total Statistics:")
    log.info(f"  Total Label 0 (Harmful/Unsafe): {total_label_0}")
    log.info(f"  Total Label 1 (Safe): {total_label_1}")
    log.info("=" * 40)

    # Use configured batch size (default 128 via Hydra override) instead of hardcoded default
    dataloader = get_hidden_states_dataloader(
        hs_data["train"], batch_size=cfg.get("batch_size", 128)
    )
    test_dataloader = get_hidden_states_dataloader(
        hs_data["test"], batch_size=cfg.get("batch_size", 128)
    )

    steering_method = cfg.get("steering_method", "actadd")
    steering_alpha = cfg.get("steering_alpha", 1.0)
    steering_normalize = cfg.get("steering_normalize", True)

    if cfg.model_type in ("polytope", "conditional_steering_vector"):
        safety_model = PolytopeConstraint(
            model=None,
            tokenizer=None,
            num_phi=cfg.dataset.num_phi,
            entropy_weight=cfg.entropy_weight,
            valid_edges_threshold=cfg.valid_edges_threshold,
            unsafe_weight=cfg.unsafe_weight,
            train_on_hs=True,
            use_nonlinear=cfg.use_nonlinear,
            use_neural_phi=cfg.get("use_neural_phi", False),
            phi_hidden_dim=cfg.get("phi_hidden_dim", 512),
            neural_phi_architecture=cfg.get("neural_phi_architecture", "mlp2"),
            neural_phi_hidden_dims=cfg.get("neural_phi_hidden_dims", None),
            entropy_assignment=cfg.entropy_assignment,
            feature_dim=cfg.feature_dim,
            f_l1_weight=cfg.f_l1_weight,
            phi_l1_weight=cfg.phi_l1_weight,
            margin=cfg.margin,
            loss_type=cfg.get(
                "loss_type", "relu"
            ),  # "relu", "softplus", or "smooth_hinge"
            desc_loss_weight=cfg.get("desc_loss_weight", 0.0),
            control_radius=cfg.get("control_radius", 1.0),
            cbf_k=cfg.get("cbf_k", 1.0),
            neural_phi_final_activation=cfg.get("neural_phi_final_activation", None),
            safe_margin=cfg.get("safe_margin", None),
            unsafe_margin=cfg.get("unsafe_margin", None),
        )
        # Get hidden states for initialization from first method's data
        # Data format: {method: {hidden_states:, labels:}}
        first_method = list(hs_data["train"].keys())[0]
        init_hidden_states = hs_data["train"][first_method]["hidden_states"]

        safety_model.rand_init_phi_theta(cfg.dataset.num_phi, x=init_hidden_states)
    elif cfg.model_type == "steering_vector":
        # Steering vector model (ActAdd or DirAblate)
        steering_method = cfg.get("steering_method", "actadd")
        safety_model = SteeringVectorModel(
            method=steering_method,
            alpha=cfg.get("steering_alpha", 1.0),
            normalize=cfg.get("steering_normalize", True),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Steering vector doesn't need iterative training - just compute from all data
        log.info(f"Computing steering vector with method: {steering_method}")

        # Collect all hidden states and labels
        all_hidden_states = []
        all_labels = []
        for method, data in hs_data["train"].items():
            all_hidden_states.append(data["hidden_states"])
            all_labels.append(data["labels"])

        combined_hs = torch.cat(all_hidden_states, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)

        # Fit the steering vector
        stats = safety_model.fit(combined_hs, combined_labels)
        log.info(f"Steering vector stats: {stats}")

    else:  # model_type == "mlp"
        # Get input dimension from first method's data
        first_method = list(hs_data["train"].keys())[0]
        input_dim = hs_data["train"][first_method]["hidden_states"].shape[-1]

        safety_model = BaselineMLP(
            input_dim=input_dim,
            hidden_dim=cfg.feature_dim,
            num_edges=cfg.dataset.num_phi,
        )

    # Train model (skip for steering_vector which is already fitted)
    if cfg.model_type != "steering_vector":
        optimizer = optim.Adam(safety_model.parameters(), lr=cfg.learning_rate)

        # Determine model type label
        type_label = cfg.model_type
        if cfg.model_type in ("polytope", "conditional_steering_vector"):
            if cfg.get("use_neural_phi", False):
                type_label = "CBF (Neural Polytope)"
            else:
                type_label = "SaP (Standard Polytope)"
            if cfg.model_type == "conditional_steering_vector":
                type_label = f"CAST ({type_label} + {steering_method})"
        elif cfg.model_type == "steering_vector":
            type_label = f"Steering Vector ({cfg.get('steering_method', 'unknown')})"

        # Log number of parameters
        num_params = sum(
            p.numel() for p in safety_model.parameters() if p.requires_grad
        )
        log.info(f"Model Type: {type_label}")
        log.info(f"Model Number of Parameters: {num_params}")

        loss = train_model(
            safety_model,
            dataloader,
            optimizer,
            num_epochs=cfg.dataset.num_epochs,
            callback_every=cfg.get("log_interval", 100),
        )

    result_dir = os.getcwd()
    weight_save_path = os.path.join(result_dir, "weights.pth")

    if cfg.model_type in ("polytope", "conditional_steering_vector"):
        steering_vector = None
        if cfg.model_type == "conditional_steering_vector":
            cast_model = SteeringVectorModel(
                method=steering_method,
                alpha=steering_alpha,
                normalize=steering_normalize,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            all_hidden_states = []
            all_labels = []
            for method, data in hs_data["train"].items():
                all_hidden_states.append(data["hidden_states"])
                all_labels.append(data["labels"])
            combined_hs = torch.cat(all_hidden_states, dim=0)
            combined_labels = torch.cat(all_labels, dim=0)
            stats = cast_model.fit(combined_hs, combined_labels)
            steering_vector = cast_model.steering_vector
            log.info(f"CAST steering vector stats: {stats}")

        model_result = ModelResult(
            safety_model.phi,
            safety_model.threshold,
            safety_model.feature_extractor,
            phi_network=safety_model.phi_network,
            steering_vector=steering_vector,
            steering_method=(
                steering_method
                if cfg.model_type == "conditional_steering_vector"
                else None
            ),
            steering_alpha=(
                steering_alpha
                if cfg.model_type == "conditional_steering_vector"
                else None
            ),
            conditional_steering=cfg.model_type == "conditional_steering_vector",
        )
    elif cfg.model_type == "steering_vector":
        # Save steering vector model directly
        model_result = safety_model
        log.info(f"Saving steering vector model: {model_result}")
    else:  # model_type == "mlp"
        model_result = safety_model

    torch.save(model_result, weight_save_path)

    # Skip evaluation for steering_vector - it's not a classifier
    if cfg.model_type == "steering_vector":
        log.info("Steering vector model saved. Skipping evaluation (not a classifier).")
        return 1.0  # Dummy accuracy

    plot_const_freq = True if cfg.model_type == "polytope" else False
    test_result = evaluate_model(
        safety_model,
        test_dataloader,
        plot_id="test",
        plot_const_freq=plot_const_freq,
    )
    log.info(f"Test acc: {test_result.accuracy:.3f}")
    log.info(f"Test fpr: {test_result.false_positive_rate:.3f}")
    log.info(f"Test fnr: {test_result.false_negative_rate:.3f}")
    log.info(f"Test f1: {test_result.f1_score:.3f}")

    return test_result.accuracy


if "__main__" == __name__:
    main()
