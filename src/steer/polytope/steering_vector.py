"""
Steering Vector Model for ActAdd and DirAblate baselines.

ActAdd (Activation Addition): ρ_steer(x, r) = x + α*r
DirAblate (Directional Ablation): ρ_steer(x, r) = x - r(r⊤x)

Reference:
- Turner et al., 2024 (Activation Addition)
- Arditi et al., 2024 (Directional Ablation)
"""

import torch
import torch.nn as nn


class SteeringVectorModel(nn.Module):
    """
    Computes and applies steering vectors for safety intervention.

    The steering vector r is computed as the difference of means:
    r = mean(safe_states) - mean(harmful_states)

    This vector points toward the "safe" direction in activation space.
    """

    def __init__(
        self,
        method: str = "actadd",
        alpha: float = 1.0,
        normalize: bool = True,
        device: str = "cuda",
    ):
        """
        Args:
            method: Steering method - "actadd" or "dirablate"
            alpha: Steering strength coefficient (ActAdd only)
            normalize: Whether to normalize the steering vector to unit length
            device: Device for computations
        """
        super().__init__()
        self.method = method
        self.alpha = alpha
        self.normalize = normalize
        self.device = device

        # Steering vector (computed via fit())
        self.register_buffer("steering_vector", None)

        # Statistics for logging
        self.harmful_mean = None
        self.safe_mean = None

    def fit(self, hidden_states: torch.Tensor, labels: torch.Tensor):
        """
        Compute steering vector from labeled hidden states.

        Args:
            hidden_states: Tensor of shape (N, hidden_dim)
            labels: Tensor of shape (N,) with 0=harmful, 1=safe
        """
        # Use CPU if CUDA not available
        device = (
            self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu"
        )
        hidden_states = hidden_states.to(device)
        labels = labels.to(device)

        harmful_mask = labels == 0
        safe_mask = labels == 1

        n_harmful = harmful_mask.sum().item()
        n_safe = safe_mask.sum().item()

        if n_harmful == 0:
            raise ValueError("No harmful samples (label=0) found in dataset")
        if n_safe == 0:
            raise ValueError("No safe samples (label=1) found in dataset")

        # Compute means
        self.harmful_mean = hidden_states[harmful_mask].float().mean(dim=0)
        self.safe_mean = hidden_states[safe_mask].float().mean(dim=0)

        # r points toward safety: mean(safe) - mean(harmful)
        r = self.safe_mean - self.harmful_mean

        # Store pre-normalization stats
        r_norm = r.norm().item()

        if self.normalize:
            r = r / r.norm()

        self.steering_vector = r

        # Log statistics
        stats = {
            "n_harmful": n_harmful,
            "n_safe": n_safe,
            "r_norm_before_normalize": r_norm,
            "method": self.method,
            "alpha": self.alpha,
            "normalize": self.normalize,
        }

        return stats

    def steer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply steering function ρ_steer to hidden states.

        Args:
            x: Hidden states tensor of shape (..., hidden_dim)

        Returns:
            Steered hidden states of same shape
        """
        if self.steering_vector is None:
            raise RuntimeError("Steering vector not computed. Call fit() first.")

        r = self.steering_vector.to(x.device).to(x.dtype)

        if self.method == "actadd":
            # Steer toward safety: x + α*r
            return x + self.alpha * r
        elif self.method == "dirablate":
            # Project out the direction from safe to harmful (remove safety direction)
            # Actually: we want to remove the *harmful* direction
            # r points safe->harmful direction would be -r
            # So we project out -r direction: x - (-r)((-r)⊤x) = x - r(r⊤x)
            # Wait, our r = safe - harmful, so r points harmful->safe
            # To remove harmful component, we want to project onto orthogonal complement of (harmful - safe)
            # Which is -r. So: x - (-r)((-r)⊤x) / ||(-r)||² = x - r(r⊤x) for unit r

            # For unit vector r: projection matrix P = r @ r.T
            # x_projected = x - P @ x = x - r(r⊤x)
            if x.dim() == 1:
                # Single vector
                projection = r * torch.dot(r, x)
            else:
                # Batch: x is (..., hidden_dim), r is (hidden_dim,)
                # r⊤x: (...,) via einsum or matmul
                dot_product = torch.einsum("...d,d->...", x, r)
                projection = dot_product.unsqueeze(-1) * r
            return x - projection
        else:
            raise ValueError(f"Unknown steering method: {self.method}")

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor = None, label: torch.Tensor = None
    ):
        """
        Forward pass for training compatibility.

        During training (with labels): computes and stores steering vector
        During inference (no labels): applies steering

        Args:
            x: Hidden states tensor
            labels: Labels tensor (0=harmful, 1=safe)
            label: Alias for labels (compatibility with evaluate_model)
        """
        # Support both 'labels' and 'label' parameter names
        if labels is None and label is not None:
            labels = label

        if labels is not None:
            # Training mode: compute steering vector
            stats = self.fit(x, labels)
            # Return ConstraintOutputs for compatibility with evaluate_model
            from steer.common.outputs import ConstraintOutputs

            batch_size = x.shape[0]
            # Steering vector doesn't do classification - treat all as "safe" (1)
            # since the steering will be applied during inference
            is_safe = torch.ones(batch_size, device=x.device)
            violations = torch.zeros(batch_size, device=x.device)
            probs = torch.ones(batch_size, device=x.device)  # Probability of being safe

            return ConstraintOutputs(
                is_safe=is_safe,
                loss=torch.tensor(0.0, device=x.device),
                violations=violations,
                probs=probs,
                additional_params=stats,
            )
        else:
            # Inference mode: apply steering
            return self.steer(x)

    def __repr__(self):
        return (
            f"SteeringVectorModel(method={self.method}, alpha={self.alpha}, "
            f"normalize={self.normalize}, fitted={self.steering_vector is not None})"
        )
