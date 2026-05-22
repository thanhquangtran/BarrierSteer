import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from steer.common.outputs import ConstraintOutputs


def get_model_hidden_states(model, tokenizer, inputs):
    encoded_inputs = tokenizer(inputs, return_tensors="pt")
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model(
            **encoded_inputs.to(device),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states
        del encoded_inputs
        del outputs
        return hidden_states


def get_model_hidden_states_loop(
    model, tokenizer, inputs, to_np=True, disable_tqdm=False
):
    hidden_states = []
    with tqdm(total=len(inputs), disable=disable_tqdm) as pbar:
        for in_text in inputs:
            hs = get_model_hidden_states(model, tokenizer, in_text)
            if to_np:
                hs = [hs.cpu().numpy() for hs in hs]
            hidden_states.append(hs)
            pbar.update(1)
    return hidden_states


class PolytopeConstraint(torch.nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        learn_phi=True,
        num_phi=10,
        entropy_weight=1.0,
        train_on_hs=False,
        valid_edges_threshold=0,
        unsafe_weight=1.0,
        feature_dim=256,
        use_nonlinear=False,
        use_neural_phi=False,
        phi_hidden_dim=512,
        neural_phi_architecture="mlp2",
        neural_phi_hidden_dims=None,
        entropy_assignment=True,
        f_l1_weight=0.1,
        phi_l1_weight=0.0001,
        margin=1.0,
        loss_type="relu",  # "relu", "softplus", or "smooth_hinge"
        desc_loss_weight=0.0,  # Weight for descent condition loss
        control_radius=1.0,  # R: maximum control magnitude
        cbf_k=1.0,  # k: CBF safety rate parameter
        neural_phi_final_activation=None,  # Final activation: None, "tanh", or "softsign"
        safe_margin=None,  # Margin for safe samples (default: margin)
        unsafe_margin=None,  # Margin for unsafe samples (default: margin)
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.phi = None
        self.threshold = None
        self.phi_categories = []
        self.learn_phi = learn_phi
        self.entropy_weight = entropy_weight
        self.num_phi = num_phi
        self.train_on_hs = train_on_hs
        self.valid_edges_threshold = valid_edges_threshold
        self.unsafe_weight = unsafe_weight
        self.entropy_assignment = entropy_assignment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim
        self.use_nonlinear = use_nonlinear
        self.use_neural_phi = use_neural_phi
        self.phi_hidden_dim = phi_hidden_dim
        self.neural_phi_architecture = neural_phi_architecture
        self.neural_phi_hidden_dims = (
            list(neural_phi_hidden_dims) if neural_phi_hidden_dims is not None else None
        )
        self.phi_network = None
        self.feature_extractor = None
        self.f_l1_weight = f_l1_weight
        self.phi_l1_weight = phi_l1_weight
        self.margin = margin
        self.safe_margin = safe_margin if safe_margin is not None else margin
        self.unsafe_margin = unsafe_margin if unsafe_margin is not None else margin
        self.loss_type = loss_type
        self.desc_loss_weight = desc_loss_weight
        self.control_radius = control_radius
        self.cbf_k = cbf_k
        self.neural_phi_final_activation = neural_phi_final_activation

        if not self.train_on_hs:
            self.rand_init_phi_theta(num_phi)

    def rand_init_phi_theta(self, num_phi, x="random input"):
        hs_rep = self.get_hidden_states_representation(x)
        rep_dim = hs_rep.shape[1]

        if self.use_nonlinear:
            self.feature_extractor = nn.Sequential(
                nn.Linear(rep_dim, self.feature_dim),
                nn.ReLU(),
            ).to(self.device)
            phi_dim = self.feature_dim
        else:
            self.feature_extractor = nn.Sequential(nn.Identity()).to(self.device)
            phi_dim = rep_dim

        # Initialize phi as either neural network or parameter matrix
        if self.use_neural_phi:
            self.phi_network = self._build_phi_network(phi_dim, num_phi)
            self.phi = None
        else:
            self.phi = torch.nn.Parameter(
                torch.randn(num_phi, phi_dim, device=self.device)
            )
            self.phi_network = None

        self.threshold = torch.nn.Parameter(torch.randn(num_phi, device=self.device))

        self._init_linear_layers(self.feature_extractor)
        if self.phi_network is not None:
            self._init_linear_layers(self.phi_network)

    def _init_linear_layers(self, module):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _build_phi_network(self, phi_dim, num_phi):
        if self.neural_phi_architecture == "deep_mlp":
            hidden_dims = (
                self.neural_phi_hidden_dims
                if self.neural_phi_hidden_dims
                else [self.phi_hidden_dim, self.phi_hidden_dim]
            )
            layer_dims = [phi_dim] + hidden_dims + [num_phi]
            return StablePhiNetwork(
                layer_dims, final_activation=self.neural_phi_final_activation
            ).to(self.device)
        elif self.neural_phi_architecture == "separate_networks":
            # Each constraint has its own separate network with independent parameters
            hidden_dims = (
                self.neural_phi_hidden_dims
                if self.neural_phi_hidden_dims
                else [self.phi_hidden_dim]
            )
            return SeparatePhiNetworks(
                phi_dim,
                num_phi,
                hidden_dims,
                self.device,
                final_activation=self.neural_phi_final_activation,
            )
        else:
            hidden_dims = (
                self.neural_phi_hidden_dims
                if self.neural_phi_hidden_dims
                else [self.phi_hidden_dim]
            )
            layer_dims = [phi_dim] + hidden_dims + [num_phi]
            return SimplePhiNetwork(
                layer_dims, final_activation=self.neural_phi_final_activation
            ).to(self.device)

    def get_hidden_states_representation(self, x):
        if self.train_on_hs:
            return x

        hidden_states = get_model_hidden_states_loop(
            self.model, self.tokenizer, x, to_np=False, disable_tqdm=True
        )
        hs_rep = torch.stack([hs[-1][:, -1, :] for hs in hidden_states], dim=1)
        hs_rep = hs_rep.squeeze(0)
        return hs_rep

    def get_safety_prediction(self, hs_rep, return_cost=False):
        features = self.feature_extractor(hs_rep)

        # Apply polytope edges (either neural network or linear)
        if self.use_neural_phi:
            cost = self.phi_network(features)  # [batch, num_phi]
        else:
            cost = torch.matmul(features, self.phi.t())  # [batch, num_phi]

        is_safe = torch.all(cost < self.threshold, dim=1)

        if return_cost:
            return is_safe, cost
        return is_safe

    def calculate_entropy(self, violation_edges, label):
        num_unsafe = torch.sum(label == 0).float()
        total_num_edges = self.num_phi

        # Ensure violation_edges and label have the same first dimension
        batch_size = min(violation_edges.shape[0], label.shape[0])
        violation_edges = violation_edges[:batch_size]
        label = label[:batch_size]

        num_constraint_violations = torch.zeros(
            total_num_edges, device=violation_edges.device
        )
        num_constraint_violations.scatter_add_(
            0,
            violation_edges[label == 0].long(),
            torch.ones(torch.sum(label == 0), device=violation_edges.device),
        )

        distribution = num_constraint_violations / (
            num_unsafe + 1e-10
        )  # Add small epsilon to avoid division by zero
        entropy = -torch.sum(distribution * torch.log2(distribution + 1e-10))

        return entropy

    def violation_entropy_assignment(
        self, violations, label, entropy_threshold=None, max_attempts=100
    ):
        batch_size, num_edges = violations.shape
        max_violations, max_violation_edges = torch.max(violations, dim=1)

        if entropy_threshold is None:
            entropy_threshold = 0.5 * np.log2(num_edges)

        # Create new tensors instead of modifying in-place
        new_max_violations = max_violations.clone()
        new_max_violation_edges = max_violation_edges.clone()

        # Get indices of unsafe examples
        unsafe_indices = torch.where(label == 0)[0]

        # If there are no unsafe examples, return original values
        if len(unsafe_indices) == 0:
            current_entropy = self.calculate_entropy(new_max_violation_edges, label)
            return new_max_violations, new_max_violation_edges, current_entropy

        current_entropy = self.calculate_entropy(new_max_violation_edges, label)
        entropy = current_entropy
        attempts = 0
        num_valid_edges = 0

        while entropy < entropy_threshold and attempts < max_attempts:
            # Randomly pick one unsafe batch item
            batch_idx = unsafe_indices[
                random.randint(0, len(unsafe_indices) - 1)
            ].item()

            sorted_violations, sorted_indices = torch.sort(
                violations[batch_idx], descending=True
            )
            valid_edges = sorted_indices[sorted_violations > self.valid_edges_threshold]
            num_valid_edges += len(valid_edges)

            if len(valid_edges) <= 1:
                chosen_edge = sorted_indices[1].item()
            else:
                chosen_edge = valid_edges[
                    random.randint(1, len(valid_edges) - 1)
                ].item()

            # Try reassigning this edge
            temp_edges = new_max_violation_edges.clone()
            temp_edges[batch_idx] = chosen_edge
            new_entropy = self.calculate_entropy(temp_edges, label)

            if new_entropy > entropy:
                new_max_violations[batch_idx] = violations[batch_idx, chosen_edge]
                new_max_violation_edges[batch_idx] = chosen_edge
                entropy = new_entropy

            attempts += 1

        return new_max_violations, new_max_violation_edges, entropy

    def forward(self, x, label=None, reduction="sum"):
        assert (
            self.phi is not None or self.phi_network is not None
        ) and self.threshold is not None, (
            "Please initialize phi/phi_network and threshold first."
        )

        hs_rep = self.get_hidden_states_representation(x)
        dtype = hs_rep.dtype
        device = hs_rep.device

        is_safe, cost = self.get_safety_prediction(hs_rep, return_cost=True)
        feature = self.feature_extractor(hs_rep)

        violations = cost - self.threshold.to(dtype)
        violation_idx = torch.nonzero(torch.relu(violations))

        batch_size = hs_rep.shape[0]
        probs = torch.zeros(batch_size, dtype=dtype, device=device)

        loss, entropy_loss, additional_params = None, None, None

        if label is not None:
            if self.entropy_assignment:
                max_violations, max_violation_edges, entropy = (
                    self.violation_entropy_assignment(
                        violations, label, entropy_threshold=None
                    )
                )
            else:
                max_violations, max_violation_edges = torch.max(violations, dim=1)
            # Apply loss function based on loss_type
            if self.loss_type == "softplus":
                # Smooth ReLU: log(1 + exp(x)), better for tanh-bounded outputs
                safe_violations = torch.mean(
                    F.softplus(self.safe_margin + violations), axis=1
                )
                unsafe_violations = F.softplus(self.unsafe_margin - max_violations)
            elif self.loss_type == "smooth_hinge":
                # Smooth hinge loss: smooth approximation of ReLU
                # For safe: margin + viol
                safe_margin_viol = self.safe_margin + violations
                safe_positive = safe_margin_viol > 0

                safe_violations = torch.mean(
                    torch.where(
                        safe_positive,
                        torch.where(
                            safe_margin_viol < 1.0,
                            0.5 * safe_margin_viol**2,
                            safe_margin_viol - 0.5,
                        ),
                        torch.zeros_like(safe_margin_viol),
                    ),
                    axis=1,
                )

                # For unsafe: margin - max_viol
                unsafe_margin_viol = self.unsafe_margin - max_violations
                unsafe_positive = unsafe_margin_viol > 0
                unsafe_violations = torch.where(
                    unsafe_positive,
                    torch.where(
                        unsafe_margin_viol < 1.0,
                        0.5 * unsafe_margin_viol**2,
                        unsafe_margin_viol - 0.5,
                    ),
                    torch.zeros_like(unsafe_margin_viol),
                )
            else:  # "relu" (default)
                # Standard ReLU (hard threshold)
                safe_violations = torch.mean(
                    torch.relu(self.safe_margin + violations), axis=1
                )
                unsafe_violations = torch.relu(self.unsafe_margin - max_violations)

            # Calculate regularization losses (mean absolute value)
            f_l1_loss = torch.abs(feature).mean()

            # Phi regularization depends on whether we use neural network or parameter
            if self.use_neural_phi:
                # L1 regularization on neural network weights (mean absolute value)
                total_l1 = sum(
                    torch.norm(p, p=1) for p in self.phi_network.parameters()
                )
                total_params = sum(p.numel() for p in self.phi_network.parameters())
                phi_l1_loss = total_l1 / total_params
            else:
                phi_l1_loss = torch.norm(self.phi, p=1, dim=1).mean()

            f_l1_term = self.f_l1_weight * f_l1_loss
            phi_l1_term = self.phi_l1_weight * phi_l1_loss

            entropy_loss = torch.tensor(0.0, device=feature.device)
            edge_entropy_loss = torch.tensor(0.0, device=feature.device)
            if self.entropy_weight > 0:
                entropy_loss = self.calculate_entropy(max_violation_edges, label)
                edge_entropy_loss = self.entropy_weight * entropy_loss

            # Compute descent loss for unsafe states (only during training)
            desc_loss_value = torch.tensor(0.0, device=feature.device)
            if self.desc_loss_weight > 0 and torch.any(label == 0) and self.training:
                # Extract unsafe features and enable gradient tracking
                unsafe_mask = label == 0
                features_unsafe = feature[unsafe_mask].detach().requires_grad_(True)

                # Recompute cost for unsafe features (with gradients)
                if self.use_neural_phi and self.phi_network is not None:
                    cost_unsafe = self.phi_network(features_unsafe)
                else:
                    cost_unsafe = torch.matmul(features_unsafe, self.phi.t())

                # Recompute h with gradients
                h_values_unsafe = self.threshold.unsqueeze(0) - cost_unsafe
                h_unsafe_grad, _ = torch.min(h_values_unsafe, dim=1)

                # Compute gradient of h w.r.t features for each unsafe example
                grad_outputs = torch.ones_like(h_unsafe_grad)
                grad_h = torch.autograd.grad(
                    outputs=h_unsafe_grad,
                    inputs=features_unsafe,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                )[
                    0
                ]  # [num_unsafe, feature_dim]

                # Compute ||∇h|| for each unsafe example
                grad_h_norm = torch.norm(grad_h, dim=1)  # [num_unsafe]

                # Descent condition: R·||∇h|| >= -k·h
                # Since h is negative for violated constraints, -k·h is positive
                required_rate = (
                    -self.cbf_k * h_unsafe_grad.detach()
                )  # Required rate to return to safety
                max_rate = (
                    self.control_radius * grad_h_norm
                )  # Max rate the dynamic can provide

                # Penalize when max_rate < required_rate
                desc_loss_value = torch.relu(required_rate - max_rate).mean()

            # Apply reduction based on parameter
            if reduction == "sum" or reduction == "mean":
                # Sum the losses
                safe_loss = safe_violations[label == 1]
                unsafe_loss = unsafe_violations[label == 0]
                loss = torch.sum(safe_loss) + self.unsafe_weight * torch.sum(
                    unsafe_loss
                )
                # Add regularization terms (already scalar values)
                loss = loss + f_l1_term + phi_l1_term - edge_entropy_loss
                # Add descent loss if enabled
                if self.desc_loss_weight > 0:
                    loss = loss + self.desc_loss_weight * desc_loss_value
                if reduction == "mean":
                    loss = loss / batch_size
            elif reduction == "none" or reduction is None:
                # Create a tensor with batch_size elements
                loss = torch.zeros(batch_size, device=device, dtype=dtype)

                # Distribute regularization terms evenly across all samples
                reg_term = (f_l1_term + phi_l1_term - edge_entropy_loss) / batch_size

                # Set loss for safe samples
                safe_mask = label == 1
                if torch.any(safe_mask):
                    loss[safe_mask] = safe_violations[safe_mask] + reg_term

                # Set loss for unsafe samples (with unsafe weight)
                unsafe_mask = label == 0
                if torch.any(unsafe_mask):
                    loss[unsafe_mask] = (
                        self.unsafe_weight * unsafe_violations[unsafe_mask] + reg_term
                    )
            else:
                raise ValueError(f"Unsupported reduction mode: {reduction}")

            # Calculate average number of edges with violations >
            # valid_edges_threshold
            unsafe_mask = label == 0
            num_edges_with_violations_unsafe = torch.sum(
                violations[unsafe_mask] > 0, dim=1
            ).float()
            avg_edges_with_violations_unsafe = (
                torch.mean(num_edges_with_violations_unsafe).item()
                if torch.sum(unsafe_mask) > 0
                else 0.0
            )

            additional_params = {
                "safe_loss": (
                    torch.mean(safe_violations[label == 1]).item()
                    if torch.any(label == 1)
                    else 0.0
                ),
                "unsafe_loss": (
                    torch.mean(unsafe_violations[label == 0]).item()
                    if torch.any(label == 0)
                    else 0.0
                ),
                "entropy": entropy_loss.item(),
                "max_violation": max_violations.max().item(),
                "min_violation": max_violations.min().item(),
                "avg_violation": max_violations.mean().item(),
                "avg_activated_edges": avg_edges_with_violations_unsafe,
                "edge_entropy_loss": edge_entropy_loss.item(),
                "f_l1_loss": f_l1_loss.item(),
                "phi_l1_loss": phi_l1_loss.item(),
                "desc_loss": (
                    desc_loss_value.item()
                    if isinstance(desc_loss_value, torch.Tensor)
                    else desc_loss_value
                ),
            }

        return ConstraintOutputs(
            is_safe=is_safe,
            probs=probs,
            violations=violations,
            loss=loss,
            entropy_loss=entropy_loss,
            additional_params=additional_params,
            violation_idx=violation_idx,
        )


class SimplePhiNetwork(nn.Module):
    """Lightweight MLP with GELU activations for neural phi."""

    def __init__(self, layer_dims, final_activation=None):
        super().__init__()
        layers = []
        for idx in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[idx], layer_dims[idx + 1]
            linear = nn.Linear(in_dim, out_dim)
            if idx < len(layer_dims) - 2:
                layers.extend([linear, nn.GELU()])
            else:
                layers.append(linear)
                # Add final activation if specified
                if final_activation == "tanh":
                    layers.append(nn.Tanh())
                elif final_activation == "softsign":
                    layers.append(nn.Softsign())
                # None = no activation (linear output)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def forward_subset(self, x, indices):
        """
        Efficiently evaluate only selected constraint logits.
        x: (batch, input_dim)
        indices: (batch, k)
        Returns: (batch, k)
        """
        layers = list(self.net.children())
        if len(layers) == 0:
            return x

        if isinstance(layers[-1], nn.Linear):
            final_linear = layers[-1]
            feature_net = nn.Sequential(*layers[:-1])
            final_activation = None
        else:
            final_linear = layers[-2]
            feature_net = nn.Sequential(*layers[:-2])
            final_activation = layers[-1]

        hidden = feature_net(x)

        weight = final_linear.weight
        bias = final_linear.bias

        weight_subset = weight[indices]  # (batch, k, hidden_dim)
        out = torch.sum(hidden.unsqueeze(1) * weight_subset, dim=-1)  # (batch, k)

        if bias is not None:
            bias_subset = bias[indices]
            out = out + bias_subset

        if final_activation is not None:
            out = final_activation(out)

        return out


class StablePhiBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_residual = in_dim == out_dim

    def forward(self, x):
        y = self.linear(x)
        y = self.norm(y)
        y = self.activation(y)
        y = self.dropout(y)
        if self.use_residual:
            y = y + x
        return y


class StablePhiNetwork(nn.Module):
    """Deeper MLP with layer norm + residuals to stabilize training."""

    def __init__(self, layer_dims, dropout=0.1, final_activation=None):
        super().__init__()
        blocks = []
        for idx in range(len(layer_dims) - 2):
            in_dim, out_dim = layer_dims[idx], layer_dims[idx + 1]
            blocks.append(StablePhiBlock(in_dim, out_dim, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Linear(layer_dims[-2], layer_dims[-1])
        # Add final activation if specified
        if final_activation == "tanh":
            self.final_activation = nn.Tanh()
        elif final_activation == "softsign":
            self.final_activation = nn.Softsign()
        else:
            self.final_activation = None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x

    def forward_subset(self, x, indices):
        """
        Efficiently evaluate only selected constraint logits.
        x: (batch, input_dim)
        indices: (batch, k)
        Returns: (batch, k)
        """
        for block in self.blocks:
            x = block(x)
        hidden = x

        weight = self.final.weight
        bias = self.final.bias

        weight_subset = weight[indices]
        out = torch.sum(hidden.unsqueeze(1) * weight_subset, dim=-1)

        if bias is not None:
            out = out + bias[indices]

        if self.final_activation is not None:
            out = self.final_activation(out)

        return out


class SeparatePhiNetworks(nn.Module):
    """Each phi constraint has its own separate network with independent parameters."""

    def __init__(
        self, phi_dim, num_phi, hidden_dims, device, dropout=0.1, final_activation=None
    ):
        super().__init__()
        self.num_phi = num_phi
        self.networks = nn.ModuleList()

        for i in range(num_phi):
            # Build a separate network for each constraint
            layers = []
            in_dim = phi_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            # Final output layer (single value per constraint)
            layers.append(nn.Linear(in_dim, 1))
            # Add final activation if specified
            if final_activation == "tanh":
                layers.append(nn.Tanh())
            elif final_activation == "softsign":
                layers.append(nn.Softsign())
            network = nn.Sequential(*layers)
            self.networks.append(network)

        self.to(device)

    def forward(self, x):
        # x: [batch, phi_dim]
        # Return: [batch, num_phi]
        # Process all networks in parallel by stacking outputs
        # This is more efficient than sequential loop as PyTorch can parallelize
        # independent operations on GPU
        outputs = [
            network(x) for network in self.networks
        ]  # List of [batch, 1] tensors
        return torch.cat(outputs, dim=1)  # [batch, num_phi]

    def forward_subset(self, x, indices):
        """
        Efficiently evaluate only selected networks.
        x: (batch, phi_dim)
        indices: (batch, k) - indices of constraints to evaluate
        Returns: (batch, k)
        """
        # Handle simple case: batch size = 1
        if x.shape[0] == 1:
            # indices is (1, k)
            # just extract the unique indices needed
            needed_indices = indices[0]
            # Run only necessary networks
            results = []
            for idx in needed_indices:
                results.append(self.networks[idx.item()](x))

            # results is list of (1, 1). Cat to (1, k)
            return torch.cat(results, dim=1)

        # General batch case
        batch_size, k = indices.shape

        # Identify union of all needed networks
        unique_indices, inverse = torch.unique(indices, return_inverse=True)
        # unique_indices: (U,)
        # inverse: (batch*k,) flattened

        # If we need almost all networks, fallback to full forward might be simpler/faster overhead-wise
        if len(unique_indices) > 0.8 * self.num_phi and self.num_phi > 10:
            return torch.gather(self.forward(x), 1, indices)

        # Run only needed networks
        # We collect outputs in a dict or list mapped by global index
        computed_outputs = {}
        for idx in unique_indices:
            i = idx.item()
            computed_outputs[i] = self.networks[i](x)  # (batch, 1)

        # Reconstruct result (batch, k)
        # Gather effectively
        # Stack computed outputs in order of unique_indices -> Tensor (batch, U)
        subset_results = torch.cat(
            [computed_outputs[i.item()] for i in unique_indices], dim=1
        )

        # inverse mapping is into 'subset_results'.
        # inverse was computed on flattened indices.
        inverse = inverse.view(batch_size, k)

        return torch.gather(subset_results, 1, inverse)


class BaselineMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=16384,
        num_edges=50,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ).to(self.device)

        # Hidden layer matching phi dimensions
        self.hidden_layer = nn.Linear(hidden_dim, num_edges).to(self.device)

        # Fixed classifier layer with output dimension 1
        self.classifier = nn.Linear(num_edges, 1, bias=False).to(self.device)

        # Initialize weights
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.zeros_(self.hidden_layer.bias)

        # Initialize and freeze classifier weights
        with torch.no_grad():
            nn.init.xavier_uniform_(self.classifier.weight)
            self.classifier.weight.requires_grad = False

    def forward(self, x, label=None):
        x = x.to(self.device)
        features = self.feature_extractor(x)
        hidden = F.relu(self.hidden_layer(features))
        logits = self.classifier(hidden)
        probs = torch.sigmoid(logits)

        loss = None
        if label is not None:
            label = label.to(self.device).float().unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, label)

        return ConstraintOutputs(
            is_safe=(probs >= 0.5).squeeze(1),
            probs=probs.squeeze(1),  # probability of being safe
            loss=loss,
            entropy_loss=None,
            additional_params={"bce_loss": loss.item() if loss is not None else None},
            violations=None,
            violation_idx=None,
        )
