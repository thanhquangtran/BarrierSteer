from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    PrecisionRecallDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
)


@dataclass
class ConstraintOutputs:
    is_safe: torch.Tensor
    loss: torch.Tensor
    violations: torch.Tensor
    probs: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    violation_idx: Optional[torch.Tensor] = None
    entropy_loss: Optional[torch.Tensor] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize additional_params as empty dict if None
        if self.additional_params is None:
            self.additional_params = {}
        # Filter out None values from additional_params
        self.additional_params = {
            k: v for k, v in self.additional_params.items() if v is not None
        }


@dataclass
class EvalResult:
    accuracy: float
    pr_auc: float
    false_positive_rate: float
    false_negative_rate: float
    f1_score: float
    outputs: ConstraintOutputs


class ModelResult:
    def __init__(
        self,
        phi,
        threshold,
        feature_extractor=None,
        phi_network=None,
        steering_vector=None,
        steering_method=None,
        steering_alpha=None,
        conditional_steering: bool = False,
    ):
        self.phi = phi
        self.threshold = threshold
        self.feature_extractor = feature_extractor
        self.phi_network = phi_network
        self.steering_vector = steering_vector
        self.steering_method = steering_method
        self.steering_alpha = steering_alpha
        self.conditional_steering = conditional_steering


def calculate_accuracy(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(model.device)
            labels = labels.float().to(model.device)
            outputs = model(inputs)
            is_safe_predictions = outputs.is_safe

            flat_labels = labels.view(-1)
            flat_predictions = is_safe_predictions.view(-1)

            correct_predictions += torch.sum(flat_predictions == flat_labels).item()
            total_predictions += len(flat_labels)

    accuracy = correct_predictions / total_predictions
    return accuracy


def get_model_output(
    model,
    tokenizer,
    input_text,
    max_new_tokens: int = 300,
    *,
    do_sample: bool = False,
    num_beams: int = 1,
    temperature: float | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    prompt_length = input_ids.shape[-1]
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_beams": num_beams,
    }

    # Handle sampling parameters
    if do_sample:
        # Only set sampling parameters if do_sample is True
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
        if top_p is not None:
            generate_kwargs["top_p"] = top_p
        if repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = repetition_penalty
    else:
        generate_kwargs["temperature"] = 1.0
        generate_kwargs["top_p"] = 1.0
        generate_kwargs["top_k"] = 0  # 0 disables top_k

    outputs = model.generate(**generate_kwargs)
    generated_text = tokenizer.decode(
        outputs[0][prompt_length:], skip_special_tokens=True
    )

    return generated_text


def plot_constraint_frequencies(
    violation_idx,
    total_constraints,
    fig_name="constraint_frequencies.png",
    fig_title="Frequency of Activated Constraints",
):
    active_constraints = violation_idx[:, 1]
    active_constraints = active_constraints.cpu().detach().numpy()
    unsafe_idx = violation_idx[:, 0]
    unsafe_idx = unsafe_idx.cpu().detach().numpy()
    num_pred_unsafe = len(np.unique(unsafe_idx))
    fig_title += f" for {num_pred_unsafe} Predicted Unsafe Data"

    unique, counts = np.unique(active_constraints, return_counts=True)
    frequencies = dict(zip(unique, counts))

    all_constraints = range(total_constraints)
    all_frequencies = [frequencies.get(i, 0) for i in all_constraints]

    # Plot the frequencies
    plt.figure(figsize=(12, 6))
    plt.bar(all_constraints, all_frequencies)
    plt.xlabel("Constraint Index")
    plt.ylabel("Frequency")
    plt.title(fig_title)
    plt.xticks(all_constraints)
    plt.savefig(fig_name)


def evaluate_model(
    model,
    dataloader,
    plot_pr_curve=True,
    plot_const_freq=True,
    plot_id="",
):
    model.eval()
    model.to(model.device)

    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_predictions = []
    all_outputs = ConstraintOutputs(
        is_safe=[],
        loss=None,
        violations=[],
        probs=[],
        hidden_states=None,
        violation_idx=[],
    )
    batch_count = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(model.device)
            labels = labels.float().to(model.device)
            outputs = model(inputs, label=labels)
            is_safe_predictions = outputs.is_safe

            flat_labels = labels.view(-1)
            flat_predictions = is_safe_predictions.view(-1)
            all_predictions.extend(flat_predictions.cpu().numpy())

            correct_predictions += torch.sum(flat_predictions == flat_labels).item()
            total_predictions += len(flat_labels)

            all_labels.extend(labels.cpu().numpy())
            all_outputs.probs.extend(outputs.probs.cpu().numpy())

            # The 0-th idx is the batch idx for predicted unsafe data.
            # The 1st idx is the idx of the constraint being violated.
            violation_idx = outputs.violation_idx
            if violation_idx is not None:
                batch_size = labels.shape[0]
                violation_idx[:, 0] += batch_size * batch_count
                all_outputs.violation_idx.extend(violation_idx)

            all_outputs.is_safe.extend(is_safe_predictions)
            batch_count += 1

    accuracy = correct_predictions / total_predictions

    if all_outputs.violation_idx and len(all_outputs.violation_idx) > 0:
        all_outputs.violation_idx = torch.stack(all_outputs.violation_idx)
    else:
        all_outputs.violation_idx = None

    probs = np.stack(all_outputs.probs, axis=0)
    precision, recall, _ = precision_recall_curve(all_labels, probs)
    pr_auc = auc(recall, precision)
    if plot_pr_curve:
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()
        plt.title(f"Precision-Recall Curve (AUC = {pr_auc:.2f})")
        plt.savefig(f"{plot_id}_precision_recall_curve.png")

    if plot_const_freq and all_outputs.violation_idx is not None:
        plot_constraint_frequencies(
            all_outputs.violation_idx,
            model.num_phi,
            fig_name=f"{plot_id}_constraint_frequencies.png",
        )

    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return EvalResult(
        accuracy,
        pr_auc,
        false_positive_rate,
        false_negative_rate,
        f1_score,
        all_outputs,
    )
