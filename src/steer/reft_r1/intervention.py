"""Rank-1 ReFT activation intervention utilities.

The intervention keeps the base model weights frozen and learns/uses a single
unit vector ``w`` at one transformer layer. At inference it computes token
scores ``ReLU(h @ w)``, pools the top-k scores per sequence, and adds
``beta * pooled_score * w`` back to the hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass
class ReFTR1Config:
    target_layer: int
    top_k: int = 5
    beta: float = 1.0
    l1_coeff: float = 0.0


def find_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """Return the common decoder layer list for HF causal LMs/wrappers."""
    candidates = [
        "model.layers",
        "lm_model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "model.decoder.layers",
    ]
    for path in candidates:
        obj: Any = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, (nn.ModuleList, list, tuple)):
            return obj
    raise ValueError("Could not locate transformer layers for ReFT-r1 intervention")


class ReFTR1Intervention(nn.Module):
    """Trainable/applicable rank-1 ReFT intervention."""

    def __init__(
        self,
        hidden_size: int,
        config: ReFTR1Config,
        vector: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config = config
        if vector is None:
            vector = torch.randn(hidden_size, dtype=torch.float32)
        vector = vector.detach().float().flatten()
        if vector.numel() != hidden_size:
            raise ValueError(
                f"vector has {vector.numel()} dims, expected hidden_size={hidden_size}"
            )
        vector = vector / vector.norm().clamp_min(1e-6)
        self.vector = nn.Parameter(vector)
        self.last_scores: Optional[torch.Tensor] = None
        self.last_non_topk_l1: Optional[torch.Tensor] = None

    @property
    def unit_vector(self) -> torch.Tensor:
        return self.vector / self.vector.norm().clamp_min(1e-6)

    def renormalize_(self) -> None:
        with torch.no_grad():
            self.vector.div_(self.vector.norm().clamp_min(1e-6))

    def apply(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply h <- h + beta * mean(topk(ReLU(h @ w))) * w."""
        w = self.unit_vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        scores = torch.relu(torch.matmul(hidden_states, w))  # [B, T]
        k = max(1, min(int(self.config.top_k), scores.shape[-1]))
        top_vals, top_idx = torch.topk(scores, k=k, dim=-1)
        mu = top_vals.mean(dim=-1, keepdim=True)  # [B, 1]

        if self.config.l1_coeff > 0 and scores.requires_grad:
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask.scatter_(dim=-1, index=top_idx, value=False)
            self.last_non_topk_l1 = (
                scores.masked_select(mask).mean() if mask.any() else scores.sum() * 0.0
            )
        else:
            self.last_non_topk_l1 = None
        self.last_scores = scores
        return hidden_states + (float(self.config.beta) * mu).unsqueeze(-1) * w


def _hook_output_with_hidden(output: Any, new_hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    return new_hidden


def attach_reft_r1_hook(model: nn.Module, intervention: ReFTR1Intervention):
    """Attach an intervention hook to ``model`` and return the removable handle."""
    layers = find_transformer_layers(model)
    layer_idx = int(intervention.config.target_layer)
    if layer_idx < 0:
        layer_idx = len(layers) + layer_idx
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(
            f"target_layer={intervention.config.target_layer} out of range for {len(layers)} layers"
        )

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        return _hook_output_with_hidden(output, intervention.apply(hidden))

    return layers[layer_idx].register_forward_hook(hook)


def load_reft_r1_intervention(
    path: str, map_location: str | torch.device = "cpu"
) -> ReFTR1Intervention:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "vector" in payload:
        cfg = payload.get("config", {}) or {}
        config = ReFTR1Config(
            target_layer=int(cfg.get("target_layer", payload.get("target_layer", 0))),
            top_k=int(cfg.get("top_k", payload.get("top_k", 5))),
            beta=float(cfg.get("beta", payload.get("beta", 1.0))),
            l1_coeff=float(cfg.get("l1_coeff", payload.get("l1_coeff", 0.0))),
        )
        vector = payload["vector"]
    else:
        raise ValueError(f"Unsupported ReFT-r1 checkpoint format: {path}")
    return ReFTR1Intervention(
        hidden_size=int(vector.numel()), config=config, vector=vector
    )
