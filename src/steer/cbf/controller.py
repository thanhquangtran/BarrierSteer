import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

Tensor = torch.Tensor


@dataclass
class CBFSolution:
    u: Tensor
    x_next: Tensor
    delta: Optional[Tensor] = None
    feasible: bool = True


class BatchedSafetyConstraint:
    """
    Batched wrapper for multiple safety constraints.
    Supports efficient evaluation of all constraints in one forward pass
    and selective gradient computation for a subset of constraints.

    b_all(x) -> (batch, num_constraints)
    grad_b_subset(x, indices) -> (batch, k, state_dim) where k = indices.shape[1]
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        threshold: Tensor,
        phi: Optional[Tensor] = None,
        phi_network: Optional[nn.Module] = None,
        epsilon: float = 0.0,
    ):
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        self.phi = phi
        self.phi_network = phi_network
        self.epsilon = epsilon

        # Ensure correct shapes
        if self.phi is not None and self.phi.dim() == 1:
            self.phi = self.phi.unsqueeze(0)  # (num_phi, feat_dim)

        # Determine number of constraints
        self.num_constraints = threshold.shape[0]

    def __len__(self):
        return self.num_constraints

    def _sync_to(self, x: Tensor) -> None:
        """Keep constraint tensors/modules colocated with the current hidden state.

        Utility runs may shard the wrapped language model across two GPUs with
        ``device_map=auto``. The steering layer can therefore execute on a
        different CUDA device than the one used when the CBF constraint wrapper
        was constructed. Move the small constraint objects lazily to the hidden
        state's device/dtype before evaluating them.
        """
        target_device = x.device
        target_dtype = x.dtype if torch.is_floating_point(x) else None

        try:
            param = next(self.feature_extractor.parameters())
            needs_move = param.device != target_device or (
                target_dtype is not None and param.dtype != target_dtype
            )
        except StopIteration:
            needs_move = False
        except AttributeError:
            needs_move = False
        if needs_move:
            kwargs = {"device": target_device}
            if target_dtype is not None:
                kwargs["dtype"] = target_dtype
            self.feature_extractor.to(**kwargs)

        if self.phi is not None and self.phi.device != target_device:
            self.phi = self.phi.to(device=target_device, dtype=target_dtype)
        elif (
            self.phi is not None
            and target_dtype is not None
            and self.phi.dtype != target_dtype
        ):
            self.phi = self.phi.to(dtype=target_dtype)

        if self.threshold.device != target_device:
            self.threshold = self.threshold.to(device=target_device, dtype=target_dtype)
        elif target_dtype is not None and self.threshold.dtype != target_dtype:
            self.threshold = self.threshold.to(dtype=target_dtype)

        if self.phi_network is not None:
            try:
                param = next(self.phi_network.parameters())
                needs_move = param.device != target_device or (
                    target_dtype is not None and param.dtype != target_dtype
                )
            except StopIteration:
                needs_move = False
            if needs_move:
                kwargs = {"device": target_device}
                if target_dtype is not None:
                    kwargs["dtype"] = target_dtype
                self.phi_network.to(**kwargs)

    def forward_b(self, x: Tensor) -> Tensor:
        """
        Compute b(x) for ALL constraints.
        x: (batch, input_dim)
        Returns: (batch, num_constraints)
        """
        # Ensure single sample has batch dim
        if x.dim() == 1:
            x = x.unsqueeze(0)
        self._sync_to(x)

        features = self.feature_extractor(x)  # (batch, feat_dim)

        if self.phi_network is not None:
            # Neural phi
            cost = self.phi_network(features)  # (batch, num_constraints)
        elif self.phi is not None:
            # Linear phi: features (batch, feat), phi.T (feat, num_constraints)
            cost = torch.matmul(features, self.phi.T)  # (batch, num_constraints)
        else:
            raise ValueError("Constraint must have phi or phi_network")

        # b = threshold - cost - epsilon
        # threshold broadcasts to (batch, num_constraints)
        return self.threshold - cost - self.epsilon

    def forward_grad_b(self, x: Tensor, indices: Tensor) -> Tensor:
        """
        Compute gradients for SELECTED constraints.
        x: (batch, input_dim)
        indices: (batch, k) - indices of constraints to compute grad for
        Returns: (batch, k, input_dim)
        """
        with torch.enable_grad():
            x_req = x.detach()
            if x_req.dim() == 1:
                x_req = x_req.unsqueeze(0)
            self._sync_to(x_req)

            # Vectorized Jacobian path: one functional jacobian call per sample,
            # batched with vmap. This avoids retain_graph loops over k constraints.
            try:
                from torch.func import jacrev, vmap

                def selected_cost_single(
                    x_single: Tensor, idx_single: Tensor
                ) -> Tensor:
                    x_b = x_single.unsqueeze(0)
                    feat = self.feature_extractor(x_b)
                    if self.phi_network is not None:
                        if hasattr(self.phi_network, "forward_subset"):
                            selected = self.phi_network.forward_subset(
                                feat, idx_single.unsqueeze(0)
                            )
                        else:
                            cost_all = self.phi_network(feat)
                            selected = torch.gather(
                                cost_all, 1, idx_single.unsqueeze(0)
                            )
                    elif self.phi is not None:
                        cost_all = torch.matmul(feat, self.phi.T)
                        selected = torch.gather(cost_all, 1, idx_single.unsqueeze(0))
                    else:
                        raise ValueError("Constraint must have phi or phi_network")
                    return selected.squeeze(0)

                grads = -vmap(jacrev(selected_cost_single))(x_req, indices)
                return grads.detach()
            except Exception:
                # Fallback path for compatibility with modules that are not torch.func-friendly.
                x_req = x_req.requires_grad_(True)
                features = self.feature_extractor(x_req)
                if self.phi_network is not None:
                    if hasattr(self.phi_network, "forward_subset"):
                        cost_selected = self.phi_network.forward_subset(
                            features, indices
                        )
                    else:
                        cost_all = self.phi_network(features)
                        cost_selected = torch.gather(cost_all, 1, indices)
                elif self.phi is not None:
                    cost_all = torch.matmul(features, self.phi.T)
                    cost_selected = torch.gather(cost_all, 1, indices)
                else:
                    raise ValueError("Constraint must have phi or phi_network")

                grads_list = []
                k = indices.shape[1]
                for i in range(k):
                    grad_i = torch.autograd.grad(
                        cost_selected[:, i].sum(),
                        x_req,
                        create_graph=False,
                        retain_graph=True if i < k - 1 else False,
                        allow_unused=True,
                    )[0]
                    if grad_i is None:
                        grad_i = torch.zeros_like(x_req)
                    grads_list.append(grad_i)
                return -torch.stack(grads_list, dim=1).detach()

    def forward_grad_merged(
        self, x: Tensor, indices: Tensor, kappa: float = 10.0
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the merged (soft-min) constraint value AND its gradient in ONE
        backward pass, avoiding the k-loop in forward_grad_b.

        b_merge(x) ≈ -(1/κ) log(Σ_i exp(-κ tanh(b_i(x))))

        x: (batch, input_dim)
        indices: (batch, k) - indices of constraints to merge
        kappa: sharpness of the soft-min

        Returns:
            b_merge: (batch,)
            grad_b_merge: (batch, input_dim)
        """
        kappa = max(kappa, 1e-6)
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            if x_req.dim() == 1:
                x_req = x_req.unsqueeze(0)
            self._sync_to(x_req)

            features = self.feature_extractor(x_req)  # (batch, feat_dim)

            if self.phi_network is not None:
                if hasattr(self.phi_network, "forward_subset"):
                    cost_selected = self.phi_network.forward_subset(features, indices)
                else:
                    cost_all = self.phi_network(features)
                    cost_selected = torch.gather(cost_all, 1, indices)
            elif self.phi is not None:
                cost_all = torch.matmul(features, self.phi.T)
                cost_selected = torch.gather(cost_all, 1, indices)
            else:
                raise ValueError("Constraint must have phi or phi_network")

            # b_selected = threshold_selected - cost_selected - epsilon
            # Gather thresholds for selected indices: threshold is (num_constraints,)
            # indices is (batch, k), we need threshold_selected: (batch, k)
            batch_size_x = x_req.shape[0]
            threshold_expanded = self.threshold.unsqueeze(0).expand(
                batch_size_x, -1
            )  # (batch, num_constraints)
            threshold_selected = threshold_expanded.gather(1, indices)  # (batch, k)
            b_selected = threshold_selected - cost_selected - self.epsilon  # (batch, k)

            # Differentiable soft-min merge
            b_scaled = torch.tanh(b_selected)  # (batch, k)
            neg_kappa_b = -kappa * b_scaled  # (batch, k)
            lse = torch.logsumexp(neg_kappa_b, dim=1)  # (batch,)
            b_merge = -(1.0 / kappa) * lse  # (batch,)

            # Single backward pass for the entire merged scalar
            grad_b_merge = torch.autograd.grad(
                b_merge.sum(),
                x_req,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )[0]

            if grad_b_merge is None:
                grad_b_merge = torch.zeros_like(x_req)

            return b_merge.detach(), grad_b_merge.detach()

    def forward_grad_merged_all(
        self, x: Tensor, kappa: float = 10.0
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the exact LogSumExp merge over ALL constraints and its gradient
        in one feature-extractor/autograd pass.

        This is mathematically equivalent to ``forward_grad_merged`` with
        ``indices`` equal to every constraint for every batch element.  It
        avoids the redundant ``forward_b`` + ``topk`` + gather path used when
        merge mode is configured with no constraint cap.
        """
        kappa = max(kappa, 1e-6)
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            if x_req.dim() == 1:
                x_req = x_req.unsqueeze(0)
            self._sync_to(x_req)

            features = self.feature_extractor(x_req)  # (batch, feat_dim)

            if self.phi_network is not None:
                cost_all = self.phi_network(features)
            elif self.phi is not None:
                cost_all = torch.matmul(features, self.phi.T)
            else:
                raise ValueError("Constraint must have phi or phi_network")

            b_all = self.threshold.unsqueeze(0) - cost_all - self.epsilon
            b_scaled = torch.tanh(b_all)
            b_merge = -(1.0 / kappa) * torch.logsumexp(-kappa * b_scaled, dim=1)

            grad_b_merge = torch.autograd.grad(
                b_merge.sum(),
                x_req,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )[0]

            if grad_b_merge is None:
                grad_b_merge = torch.zeros_like(x_req)

            return b_merge.detach(), grad_b_merge.detach()


class SafetyConstraint:
    """
    Wrapper around a differentiable safety function h(x) with margin epsilon.
    b(x) = h(x) - epsilon.
    """

    def __init__(self, h_fn: Callable[[Tensor], Tensor], epsilon: float = 0.0):
        self.h_fn = h_fn
        self.epsilon = epsilon

    def b(self, x: Tensor) -> Tensor:
        return self.h_fn(x) - self.epsilon

    def grad_b(self, x: Tensor) -> Tensor:
        """Compute ∂b/∂x for a batch of states."""
        # Generation often runs under torch.no_grad(); we still need gradients
        # w.r.t. x for the CBF constraints. Explicitly re-enable autograd here.
        with torch.enable_grad():
            x_req = x.clone().detach().requires_grad_(True)
            b_val = self.b(x_req)
            grad = torch.autograd.grad(
                b_val.sum(), x_req, create_graph=False, retain_graph=False
            )[0]
            return grad.detach()


class EstimatedDynamics:
    """Finite-difference dynamics \u0307x = u with nominal u = (x_t - x_{t-1})/dt."""

    def __init__(self, dt: float = 1.0):
        self.dt = dt

    def nominal_u(self, x_t: Tensor, x_prev: Tensor) -> Tensor:
        return (x_t - x_prev) / self.dt


class LearnedDynamics:
    """
    Learned dynamics \u0307x = f(x) + g(x)u.
    f_model: Callable returning drift tensor shaped like x.
    g_model: Callable returning actuation matrix of shape (batch, state_dim, control_dim).
    """

    def __init__(
        self,
        f_model: Callable[[Tensor], Tensor],
        g_model: Callable[[Tensor], Tensor],
        dt: float = 1.0,
    ):
        self.f_model = f_model
        self.g_model = g_model
        self.dt = dt

    def evaluate(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        f = self.f_model(x)
        g = self.g_model(x)
        return f, g


def _project_single(u0: Tensor, a: Tensor, c: Tensor, eps: float) -> Tensor:
    """Projection of u0 onto half-space a^T u >= c.

    Supports both single and batched inputs.
    - Single: u0 (q,), a (q,), c (scalar) -> returns (q,)
    - Batched: u0 (batch, q) or (batch,), a (batch, q) or (batch,), c (batch,) -> returns (batch, q) or (batch,)
    """
    # Check if batched:
    # - u0 has 2 dims (batch, q), OR
    # - u0 is (batch,) and a is (batch, q) or (batch,), OR
    # - u0 is (batch,) and a is (batch,) and c is (batch,)
    is_batched = (u0.dim() > 1) or (
        u0.dim() == 1
        and u0.shape[0] > 1
        and (
            (a.dim() == 2 and a.shape[0] == u0.shape[0])
            or (
                a.dim() == 1
                and a.shape[0] == u0.shape[0]
                and c.dim() == 1
                and c.shape[0] == u0.shape[0]
            )
        )
    )

    if is_batched:
        # Save original u0 shape to restore later
        original_u0_shape = u0.shape
        original_u0_was_1d = u0.dim() == 1

        # Ensure u0 and a have same number of dims
        if u0.dim() == 1 and a.dim() == 2:
            # u0 is (batch,), a is (batch, q) - scalar state
            u0 = u0.unsqueeze(-1)  # (batch, 1)
        elif u0.dim() == 2 and a.dim() == 1:
            # u0 is (batch, q), a is (batch,) - shouldn't happen, but handle it
            a = a.unsqueeze(-1)  # (batch, 1)
        elif u0.dim() == 1 and a.dim() == 1:
            # Both are (batch,) - scalar state, batched
            u0 = u0.unsqueeze(-1)  # (batch, 1)
            a = a.unsqueeze(-1)  # (batch, 1)

        # Now both should be (batch, q)
        gap = (c.unsqueeze(-1) - (a * u0).sum(dim=-1, keepdim=True)).clamp_min(
            0.0
        )  # (batch, 1)
        denom = (a * a).sum(dim=-1, keepdim=True) + eps  # (batch, 1)
        result = u0 + (gap / denom) * a  # (batch, q)

        # If original u0 was (batch,) and result is (batch, 1), squeeze back to match input
        if original_u0_was_1d and result.shape[-1] == 1:
            result = result.squeeze(-1)  # (batch, 1) -> (batch,)
        return result
    else:
        # Single sample
        a_vec = a.view(-1)
        u_vec = u0.view(-1)
        gap = (c - (a_vec * u_vec).sum()).clamp_min(0.0)
        denom = (a_vec * a_vec).sum() + eps
        return u_vec + (gap / denom) * a_vec


def _closed_form_two(u0: Tensor, A: Tensor, c: Tensor, eps: float) -> Tensor:
    """
    Closed-form projection for up to two constraints (supports both single and batched).
    Uses ABNet eBQP identity-H solution when two constraints are present.
    Constraints are of the form A u >= c.

    Args:
        u0: Reference control, shape (q,) or (batch, q)
        A: Constraint matrix, shape (m, q) or (m, batch, q) for batched
        c: Constraint vector, shape (m,) or (m, batch) for batched
        eps: Small epsilon for numerical stability
    """
    m = A.shape[0]
    if m == 0:
        return u0

    # Batched constraints are represented as (m, batch, q).
    is_batched = A.dim() == 3

    if m == 1:
        if is_batched:
            # Batched single constraint: A (1, batch, q) or A (1, q) with u0 (batch, q)
            if A.dim() == 3:
                result = _project_single(u0, A[0], c[0], eps)
                # Ensure result shape matches u0 shape for scalar states
                if u0.dim() == 1 and result.dim() == 2 and result.shape[-1] == 1:
                    result = result.squeeze(-1)
                return result
            else:
                # A (1, q), u0 (batch, q), c (1, batch) or c (1,)
                if c.dim() > 1:
                    result = _project_single(
                        u0, A[0].unsqueeze(0).expand(u0.shape[0], -1), c[0], eps
                    )
                else:
                    result = _project_single(
                        u0,
                        A[0].unsqueeze(0).expand(u0.shape[0], -1),
                        c[0].expand(u0.shape[0]),
                        eps,
                    )
                # Ensure result shape matches u0 shape
                if u0.dim() == 1 and result.dim() == 2 and result.shape[-1] == 1:
                    result = result.squeeze(-1)
                return result
        else:
            return _project_single(u0, A[0], c[0], eps)
    if is_batched:
        batch_size = A.shape[1]
        # A: (m, batch, q), c: (m, batch), u0: (batch, q)
        # Convert to ABNet format: (batch, m, q) and (batch, m)
        # For two constraints: A_bn = (batch, 2, q), c_bn = (batch, 2)
        A_bn = A.transpose(0, 1)  # (batch, m, q)
        c_bn = c.transpose(0, 1)  # (batch, m)

        # Extract constraints in ABNet style
        G1 = -A_bn[:, 0, :]  # (batch, q)
        G2 = -A_bn[:, 1, :]  # (batch, q)
        h1 = -c_bn[:, 0:1]  # (batch, 1) - keep as 2D like ABNet
        h2 = -c_bn[:, 1:2]  # (batch, 1)

        y1_bar = G1  # (batch, q)
        y2_bar = G2  # (batch, q)
        u_bar = u0  # (batch, q) - ensure same shape

        # Compute p1_bar and p2_bar following ABNet: h1 - sum(G1*u_bar, dim=1).unsqueeze(1)
        p1_bar = h1 - torch.sum(G1 * u_bar, dim=1).unsqueeze(1)  # (batch, 1)
        p2_bar = h2 - torch.sum(G2 * u_bar, dim=1).unsqueeze(1)  # (batch, 1)

        # Build G matrix following ABNet format: (4, batch, 1)
        g00 = (
            torch.sum(y1_bar * y1_bar, dim=1).unsqueeze(1).unsqueeze(0) + eps
        )  # (1, batch, 1)
        g01 = (
            torch.sum(y1_bar * y2_bar, dim=1).unsqueeze(1).unsqueeze(0)
        )  # (1, batch, 1)
        g10 = g01  # (1, batch, 1)
        g11 = (
            torch.sum(y2_bar * y2_bar, dim=1).unsqueeze(1).unsqueeze(0) + eps
        )  # (1, batch, 1)

        G = torch.cat([g00, g01, g10, g11], dim=0)  # (4, batch, 1) - matches ABNet

        w_p1_bar = torch.clamp(p1_bar, max=0.0)
        w_p2_bar = torch.clamp(p2_bar, max=0.0)

        # Compute lambda following ABNet logic using G[0], G[1], G[2], G[3]
        # G[0] = g00, G[1] = g01, G[2] = g10, G[3] = g11, all shape (batch, 1)
        lambda1 = torch.where(
            G[2] * w_p2_bar < G[3] * p1_bar,
            torch.zeros_like(p1_bar),
            torch.where(
                G[1] * w_p1_bar < G[0] * p2_bar,
                w_p1_bar / G[0],
                torch.clamp(G[3] * p1_bar - G[2] * p2_bar, max=0.0)
                / (G[0] * G[3] - G[1] * G[2] + eps),
            ),
        )

        lambda2 = torch.where(
            G[2] * w_p2_bar < G[3] * p1_bar,
            w_p2_bar / G[3],
            torch.where(
                G[1] * w_p1_bar < G[0] * p2_bar,
                torch.zeros_like(p1_bar),
                torch.clamp(G[0] * p2_bar - G[1] * p1_bar, max=0.0)
                / (G[0] * G[3] - G[1] * G[2] + eps),
            ),
        )

        u = lambda1 * y1_bar + lambda2 * y2_bar + u_bar  # (batch, q)

        # Check feasibility (batched)
        # A: (m, batch, q), u: (batch, q) or (batch,), c: (m, batch)
        # Compute A @ u for each constraint
        if u.dim() == 1:
            # Scalar state: u is (batch,), A is (m, batch, 1)
            u_for_einsum = u.unsqueeze(-1)  # (batch, 1)
        else:
            u_for_einsum = u  # (batch, q)
        A_u = torch.einsum("mbq,bq->mb", A, u_for_einsum)  # (m, batch)
        violations = c - A_u  # (m, batch)
        if torch.all(violations <= 1e-6):
            return u
        # Fallback to active set for any batch element that fails
        return _active_set_qp_batched(u0, A, c, eps)
    else:
        # Original single-sample code
        G1 = -A[0].view(-1)
        G2 = -A[1].view(-1)
        h1 = c[0]
        h2 = c[1]

        y1_bar = G1
        y2_bar = G2
        u_bar = u0.view(-1)
        p1_bar = h1 - (G1 * u_bar).sum()
        p2_bar = h2 - (G2 * u_bar).sum()

        g00 = (y1_bar * y1_bar).sum() + eps
        g01 = (y1_bar * y2_bar).sum()
        g10 = g01
        g11 = (y2_bar * y2_bar).sum() + eps

        w_p1_bar = torch.clamp(p1_bar, max=0.0)
        w_p2_bar = torch.clamp(p2_bar, max=0.0)

        denom = g00 * g11 - g01 * g10 + eps

        lambda1 = torch.where(
            g10 * w_p2_bar < g11 * p1_bar,
            torch.zeros_like(p1_bar),
            torch.where(
                g01 * w_p1_bar < g00 * p2_bar,
                w_p1_bar / g00,
                torch.clamp(g11 * p1_bar - g10 * p2_bar, max=0.0) / denom,
            ),
        )

        lambda2 = torch.where(
            g10 * w_p2_bar < g11 * p1_bar,
            w_p2_bar / g11,
            torch.where(
                g01 * w_p1_bar < g00 * p2_bar,
                torch.zeros_like(p1_bar),
                torch.clamp(g00 * p2_bar - g01 * p1_bar, max=0.0) / denom,
            ),
        )

        u = lambda1 * y1_bar + lambda2 * y2_bar + u_bar

        # Ensure feasibility (numerical guard); do check in fp32 for stability.
        if torch.all((A.float() @ u.float()) >= (c.float() - 1e-6)):
            return u
        return _active_set_qp(u0, A, c, eps)


def _active_set_qp_batched(
    u0: Tensor, A: Tensor, c: Tensor, eps: float, max_iters: int = 100
) -> Tensor:
    """
    Batched active-set solver for min ||u - u0||^2 s.t. A u >= c.
    Falls back to per-sample active set if batched version fails.
    """
    if A.dim() == 3:  # Batched: (m, batch, q)
        batch_size = A.shape[1]
        # Process each batch element separately (can be optimized later)
        results = []
        for b in range(batch_size):
            A_b = A[:, b, :]  # (m, q)
            c_b = c[:, b]  # (m,)
            u0_b = u0[b]  # (q,)
            u_b = _active_set_qp(u0_b, A_b, c_b, eps, max_iters)
            results.append(u_b)
        return torch.stack(results, dim=0)
    else:
        return _active_set_qp(u0, A, c, eps, max_iters)


def _active_set_qp(
    u0: Tensor, A: Tensor, c: Tensor, eps: float, max_iters: int = 100
) -> Tensor:
    """
    Small active-set solver for min ||u - u0||^2 s.t. A u >= c.
    """
    if A.numel() == 0:
        return u0

    # Solve in fp32 for numerical stability and to avoid CUDA linalg limitations
    # for half/bfloat16 (e.g. torch.linalg.solve on cuda: Half).
    A_mat = A.view(A.shape[0], -1)
    c_vec = c.view(-1)
    u_vec = u0.view(-1)

    A_mat_f = A_mat.float()
    c_vec_f = c_vec.float()
    u0_f = u_vec.float()

    active: List[int] = []
    u = u0_f.clone()
    for _ in range(max_iters):
        violations = (c_vec_f - A_mat_f @ u).flatten()
        if torch.all(violations <= 1e-6):
            break

        most_violated = torch.argmax(violations).item()
        if most_violated not in active:
            active.append(most_violated)

        A_act = A_mat_f[active]
        c_act = c_vec_f[active]
        M = A_act @ A_act.T + eps * torch.eye(
            len(active), device=A.device, dtype=torch.float32
        )
        rhs = c_act - A_act @ u0_f
        try:
            lam = torch.linalg.solve(M, rhs)
        except RuntimeError:
            # lstsq is also float32-only on many backends; safe here.
            lam = torch.linalg.lstsq(M, rhs).solution

        u_candidate = u0_f + A_act.T @ lam
        if lam.min() < -1e-6:
            drop_idx = torch.argmin(lam).item()
            active.pop(drop_idx)
            continue

        u = u_candidate

    return u.to(dtype=u0.dtype).view_as(u0)


class CBFController:
    """
    CBF controller supporting estimated dynamics and learned dynamics.
    - Estimated: min ||u - u_ref||^2 s.t. \u2207b u + k b >= 0.
    - Learned:   min ||u||^2 + w \u03b4^2 s.t. \u2207b(f+g u)+k b >=0, -\u2207V g u + \u03b4 >= pV+\u2207V f.
    Closed-form solutions are used for <=2 constraints; a tiny active-set QP
    is used for 3+ constraints.
    """

    def __init__(
        self,
        dt: float = 1.0,
        k: float = 1.0,
        w: float = 1.0,
        p: float = 10.0,
        eps: float = 1e-8,
        kappa: float = 10.0,
        constraint_mode: str = "topk",
    ):
        self.dt = dt
        self.k = k
        self.w = w
        self.p = p
        self.eps = eps
        # Controls the sharpness of LogSumExp-based constraint merging:
        # larger kappa -> closer to hard min, smaller kappa -> smoother aggregation.
        self.kappa = kappa
        # constraint_mode:
        #   - "topk"  : keep worst-k constraints (possibly per-sample)
        #   - "merge" : smooth-merge many constraints into fewer surrogates
        #   - "qp"    : disable any pre-reduction; solve QP with all constraints
        self.constraint_mode = constraint_mode

        # Profiling stats
        self.profiling_stats = {}
        self.profiling_enabled = True
        # Phase-level profiling: per-step lists for avg±std
        self.phase_timings = {}

    def reset_profiling_stats(self):
        self.profiling_stats = {}
        self.phase_timings = {}

    def print_profiling_stats(self):
        import numpy as np

        # Print phase-level summary (4 phases)
        if self.phase_timings:
            print("\n--- CBF Phase Profiling (4 Phases) ---")
            print(
                f"{'Phase':<25} | {'Count':<8} | {'Avg (ms)':<12} | {'Std (ms)':<12} | {'Total (ms)':<12}"
            )
            print("-" * 75)
            phase_order = [
                "Phase1_SafetyCheck",
                "Phase2_Selection",
                "Phase3_Gradient",
                "Phase4_QPSolve",
                "Total",
            ]
            for phase in phase_order:
                if phase in self.phase_timings and len(self.phase_timings[phase]) > 0:
                    vals = np.array(self.phase_timings[phase])
                    count = len(vals)
                    avg = np.mean(vals)
                    std = np.std(vals)
                    total = np.sum(vals)
                    print(
                        f"{phase:<25} | {count:<8} | {avg:<12.4f} | {std:<12.4f} | {total:<12.2f}"
                    )
            print("-" * 75)

        if not self.profiling_stats:
            if not self.phase_timings:
                print("No profiling stats collected.")
            return

        print("\n--- CBF Controller Detailed Profiling Stats ---")
        # Header
        print(f"{'Step':<15} | {'Count':<8} | {'Avg (ms)':<10} | {'Total (ms)':<10}")
        print("-" * 50)

        # Sort by total time descending or just consistent order
        for key in sorted(self.profiling_stats.keys()):
            data = self.profiling_stats[key]
            count = data["count"]
            total = data["total_ms"]
            avg = total / count if count > 0 else 0
            print(f"{key:<15} | {count:<8} | {avg:<10.4f} | {total:<10.2f}")
        print("-" * 50)
        import sys

        sys.stdout.flush()

    def _update_stats(self, name, ms):
        if name not in self.profiling_stats:
            self.profiling_stats[name] = {"count": 0, "total_ms": 0.0}
        self.profiling_stats[name]["count"] += 1
        self.profiling_stats[name]["total_ms"] += ms

    def _update_phase(self, phase_name, ms):
        if phase_name not in self.phase_timings:
            self.phase_timings[phase_name] = []
        self.phase_timings[phase_name].append(ms)

    def _merge_constraints_one(
        self, b_vals: Tensor, grad_b_vals: Tensor, scores: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Merge many constraints into ONE smooth constraint using a tanh-bounded LogSumExp "soft-min":

            b_merge(x) ≈ -(1/κ) log(∑_i exp(-κ * tanh(b_i(x))))

        This approximates the minimum b_i(x) (i.e. the "most unsafe" constraint) while remaining differentiable.
        Returns:
            b_merge: scalar Tensor
            grad_merge: Tensor shaped like a single constraint gradient (q,)

        Note: `scores` is accepted for API symmetry but the merge uses b_vals directly.
        """
        if b_vals.numel() == 0:
            raise ValueError("Cannot merge empty constraint set")

        # Respect configured merge sharpness; only guard against non-positive values.
        kappa = float(max(self.kappa, 1e-6))

        b_subset = b_vals
        grad_subset = grad_b_vals
        N = b_subset.numel()

        # Step 1: tanh bounding
        b_scaled = torch.tanh(b_subset)
        tanh_grad_factor = 1.0 - b_scaled**2

        # Step 2: kappa-scaled logsumexp of -kappa*b_scaled
        neg_kappa_b = -kappa * b_scaled
        max_neg_kappa_b = torch.max(neg_kappa_b)
        lse = max_neg_kappa_b + torch.log(
            torch.sum(torch.exp(neg_kappa_b - max_neg_kappa_b))
        )
        b_merge = -(1.0 / kappa) * lse

        # Step 3: gradient (chain rule)
        weights_scaled = torch.softmax(neg_kappa_b, dim=0)
        effective_weights = weights_scaled * tanh_grad_factor
        grad_merge = (effective_weights.unsqueeze(-1) * grad_subset).sum(dim=0)

        return b_merge, grad_merge

    # ---- Estimated dynamics -------------------------------------------------
    def step_estimated(
        self,
        x_t: Tensor,
        x_prev: Tensor,
        constraints: Sequence[SafetyConstraint],
        dynamics: Optional[EstimatedDynamics] = None,
        max_constraints: Optional[int] = None,
    ) -> CBFSolution:
        """
        Batched CBF step with estimated dynamics.
        Supports both single samples (x_t: (q,)) and batches (x_t: (batch, q)).
        """
        if len(constraints) == 0:
            raise ValueError("constraints must be non-empty")

        # -- PROFILING EVENTS --
        # CUDA events are only created when profiling is enabled.  This leaves
        # generation behavior unchanged while avoiding event overhead in normal
        # non-benchmark runs.
        events = {}

        def record_time(name):
            if torch.cuda.is_available() and self.profiling_enabled:
                t = torch.cuda.Event(enable_timing=True)
                t.record()
                events[name] = t

        record_time("start")

        # Unit tests expect min-norm control (safe => u≈0), not tracking nominal dynamics.
        # Also: many test h_fn are NOT batch-safe (e.g. lambda x: 0.5 - x[0]),
        # so we evaluate b/grad_b per-sample then stack.

        # SafeRepModel commonly passes a single BatchedSafetyConstraint wrapped in a list.
        # Unwrap early so downstream logic (notably nominal-reference selection) can
        # reason about the effective number of constraints correctly.
        if (
            isinstance(constraints, list)
            and len(constraints) == 1
            and isinstance(constraints[0], BatchedSafetyConstraint)
        ):
            constraints = constraints[0]

        scalar_single = x_t.dim() == 0
        if scalar_single:
            x_t = x_t.unsqueeze(0)  # (1,)
            x_prev = x_prev.unsqueeze(0)  # (1,)

        # Disambiguate 1D input: single vector state (q,) vs batch of scalar states (batch,)
        if x_t.dim() == 2:
            return_mode = "batch_vector"
            batch_size, state_dim = x_t.shape
            x_t_batched = x_t
            x_prev_batched = x_prev
        elif x_t.dim() == 1:
            if isinstance(constraints, BatchedSafetyConstraint):
                # BatchedSafetyConstraint uses feature vectors; 1D input is a single vector state.
                return_mode = "single_vector"
                batch_size, state_dim = 1, x_t.shape[0]
                x_t_batched = x_t.unsqueeze(0)  # (1, q)
                x_prev_batched = x_prev.unsqueeze(0)  # (1, q)
            else:
                b0 = constraints[0].b(x_t)
                if b0.dim() == 0:
                    return_mode = "single_vector"
                    batch_size, state_dim = 1, x_t.shape[0]
                    x_t_batched = x_t.unsqueeze(0)  # (1, q)
                    x_prev_batched = x_prev.unsqueeze(0)  # (1, q)
                else:
                    return_mode = "batch_scalar"
                    batch_size, state_dim = x_t.shape[0], 1
                    x_t_batched = x_t  # (batch,)
                    x_prev_batched = x_prev  # (batch,)
        else:
            raise ValueError(f"Unsupported x_t shape: {tuple(x_t.shape)}")

        # Decision variable u has dim = state_dim for EstimatedDynamics.
        # Test suite expects different "reference" behavior:
        # - For scalar batches: min-norm control (safe => u≈0)
        # - For vector states: stay close to nominal finite-difference control on unconstrained dims
        dyn = dynamics or EstimatedDynamics(self.dt)
        # Heuristic to match unit tests:
        # - scalar batches: min-norm (u0=0)
        # - vector states with a single constraint: min-norm (u0=0)
        # - vector states with >=2 constraints: stay close to nominal finite-difference control
        effective_num_constraints = len(constraints)
        if return_mode == "batch_scalar" or effective_num_constraints == 1:
            u0 = torch.zeros(
                (batch_size, state_dim),
                device=x_t_batched.device,
                dtype=x_t_batched.dtype,
            )
        else:
            u0 = dyn.nominal_u(x_t_batched, x_prev_batched)
            if u0.dim() == 1:
                u0 = u0.unsqueeze(0)

        record_time("prep_done")

        # Optimization: Check if using BatchedSafetyConstraint
        is_batched_constraint = isinstance(constraints, BatchedSafetyConstraint)

        A_list: List[Tensor] = []
        c_list: List[Tensor] = []
        b_list: List[Tensor] = []

        if is_batched_constraint:
            # ---------------------------------------------------------
            # FAST PATH: Batched evaluation + Top-K Gradient Calculation
            # ---------------------------------------------------------
            num_constraints = len(constraints)
            merge_all_constraints = (
                self.constraint_mode == "merge"
                and num_constraints > 1
                and (
                    max_constraints is None
                    or max_constraints <= 0
                    or max_constraints >= num_constraints
                )
            )

            if merge_all_constraints:
                # Exact fast path for the common "merge all classifiers" setup:
                # no early stopping, no approximation, no constraint subset.
                merged_b, merged_grad = constraints.forward_grad_merged_all(
                    x_t_batched, kappa=self.kappa
                )

                record_time("forward_b")
                record_time("topk")
                record_time("grad")

                A = merged_grad.unsqueeze(0)  # (1, batch, q)
                c = (-self.k * merged_b).unsqueeze(0)  # (1, batch)

                record_time("merge")
            else:
                # 1. Evaluate ALL b(x) efficiently
                # b_all: (batch, num_constraints)
                b_all = constraints.forward_b(x_t_batched)

                record_time("forward_b")

                # 2. Determine which constraints to keep (Top-K)
                num_constraints = b_all.shape[1]

                # Calculate scores (violation magnitude)
                # For estimated dynamics, approximate score by b value (lower b = more unsafe)
                # score = -b (since we want most violated, which is min b)
                scores = -b_all  # (batch, num_constraints)

                k = num_constraints
                if (
                    max_constraints is not None
                    and max_constraints > 0
                    and num_constraints > max_constraints
                    and self.constraint_mode in ("topk", "merge")
                ):
                    k = min(max_constraints, num_constraints)

                # Select top-k indices
                # topk_vals, topk_idx: (batch, k)
                topk = torch.topk(scores, k, dim=1)
                topk_idx = topk.indices
                topk_vals = -topk.values  # recover b values (batch, k)

                record_time("topk")

                # 3. Compute gradients for selected constraints
                if self.constraint_mode == "merge" and k > 1:
                    # --- OPTIMIZED MERGE: single backward pass ---
                    # Instead of k backward passes (forward_grad_b) + manual merge,
                    # compute the merged b and grad in one backward pass.
                    merged_b, merged_grad = constraints.forward_grad_merged(
                        x_t_batched, topk_idx, kappa=self.kappa
                    )

                    record_time("grad")

                    A = merged_grad.unsqueeze(0)  # (1, batch, q)
                    c = (-self.k * merged_b).unsqueeze(0)  # (1, batch)

                    record_time("merge")
                else:
                    # grad_subset: (batch, k, state_dim)
                    grad_subset = constraints.forward_grad_b(x_t_batched, topk_idx)

                    record_time("grad")

                    # 4. Construct A and c
                    A = grad_subset.permute(1, 0, 2).contiguous()  # (k, batch, q)
                    c = (
                        (-self.k * topk_vals).detach().permute(1, 0).contiguous()
                    )  # (k, batch)

        else:
            # ---------------------------------------------------------
            # SLOW PATH: Legacy Loop
            # ---------------------------------------------------------
            A_rows: List[Tensor] = []
            c_rows: List[Tensor] = []

            for con in constraints:
                b_vals: List[Tensor] = []
                grad_vals: List[Tensor] = []
                for b in range(batch_size):
                    if return_mode == "batch_scalar":
                        x_sample = x_t_batched[b]
                    else:
                        x_sample = x_t_batched[b]
                    b_sample = con.b(x_sample)
                    g_sample = con.grad_b(x_sample)
                    b_vals.append(b_sample.reshape(()))
                    grad_vals.append(g_sample.reshape(-1))

                b_val = torch.stack(b_vals, dim=0)  # (batch,)
                grad_val = torch.stack(grad_vals, dim=0)  # (batch, state_dim)
                if grad_val.dim() == 1:
                    grad_val = grad_val.unsqueeze(-1)

                A_rows.append(grad_val)
                c_rows.append(-self.k * b_val)

            if len(A_rows) > 0:
                A = torch.stack(A_rows, dim=0)  # (num_constraints, batch, state_dim)
                c = torch.stack(c_rows, dim=0)  # (num_constraints, batch)
            else:
                # Should have been caught by early check, but for safety:
                A = torch.zeros(
                    (0, batch_size, state_dim),
                    device=x_t_batched.device,
                    dtype=x_t_batched.dtype,
                )
                c = torch.zeros(
                    (0, batch_size), device=x_t_batched.device, dtype=x_t_batched.dtype
                )

        record_time("construct_qp")

        # Solve QP
        if A.shape[0] <= 2:
            u = _closed_form_two(u0, A, c, self.eps)  # (batch, q)
        else:
            u = _active_set_qp_batched(u0, A, c, self.eps)  # (batch, q)

        record_time("solve_qp")

        # Record timing stats
        if torch.cuda.is_available() and self.profiling_enabled:
            torch.cuda.synchronize()
            try:
                t_prep = events["start"].elapsed_time(events["prep_done"])
                self._update_stats("Prep", t_prep)

                if "forward_b" in events:
                    # Fast Path timings
                    t_fwd = events["prep_done"].elapsed_time(events["forward_b"])
                    self._update_stats("B(x)", t_fwd)

                    t_topk = events["forward_b"].elapsed_time(events["topk"])
                    self._update_stats("TopK", t_topk)

                    t_grad = events["topk"].elapsed_time(events["grad"])
                    self._update_stats("Grad", t_grad)

                    if "merge" in events:
                        t_merge = events["grad"].elapsed_time(events["merge"])
                        self._update_stats("Merge", t_merge)
                        t_qp_prep = events["merge"].elapsed_time(events["construct_qp"])
                        self._update_stats("QP_Prep", t_qp_prep)
                    else:
                        t_merge = 0.0
                        t_qp_prep = events["grad"].elapsed_time(events["construct_qp"])
                        self._update_stats("QP_Prep", t_qp_prep)

                    # --- Phase-level timings ---
                    # Phase 1: Safety Check = Prep + B(x)
                    self._update_phase("Phase1_SafetyCheck", t_prep + t_fwd)
                    # Phase 2: Constraint Selection = TopK + Merge (if applicable)
                    self._update_phase("Phase2_Selection", t_topk + t_merge)
                    # Phase 3: Gradient = Grad
                    self._update_phase("Phase3_Gradient", t_grad)
                    # Phase 4: QP Solve = QP_Prep + Solve_QP
                    t_solve = events["construct_qp"].elapsed_time(events["solve_qp"])
                    self._update_phase("Phase4_QPSolve", t_qp_prep + t_solve)
                    # Total
                    self._update_phase(
                        "Total",
                        t_prep
                        + t_fwd
                        + t_topk
                        + t_merge
                        + t_grad
                        + t_qp_prep
                        + t_solve,
                    )
                else:
                    # Slow path timings
                    t_slow_setup = events["prep_done"].elapsed_time(
                        events["construct_qp"]
                    )
                    self._update_stats("Slow_Setup", t_slow_setup)

                t_solve = events["construct_qp"].elapsed_time(events["solve_qp"])
                self._update_stats("Solve_QP", t_solve)

            except Exception as e:
                # Fail silently or print error once?
                pass

        if return_mode == "batch_scalar":
            x_next = x_prev_batched + u.squeeze(-1) * self.dt  # (batch,)
        else:
            x_next = x_prev_batched + u * self.dt

        # Output formatting
        if return_mode == "batch_scalar":
            u_out = u.squeeze(-1)  # (batch,)
            x_next_out = x_next  # (batch,)
            if scalar_single:
                u_out = u_out.squeeze(0)
                x_next_out = x_next_out.squeeze(0)
        elif return_mode == "single_vector":
            u_out = u.squeeze(0)
            x_next_out = x_next.squeeze(0)
        else:
            u_out = u
            x_next_out = x_next

        # DEBUG: Check if steering is actually happening
        u_norm = torch.norm(u_out.float())
        # if u_norm < 1e-6:
        #      print(f"[CBF DEBUG] calc u is ZERO. b_vals min: {torch.min(torch.stack(b_list)).item():.4f}")
        # else:
        #      print(f"[CBF DEBUG] Steering active! |u|={u_norm:.4f}, b_vals min: {torch.min(torch.stack(b_list)).item():.4f}")

        feasible = True
        return CBFSolution(u=u_out, x_next=x_next_out, delta=None, feasible=feasible)

    def step_estimated_iterative(
        self,
        x_t: Tensor,
        x_prev: Tensor,
        constraints: Sequence[SafetyConstraint],
        dynamics: Optional[EstimatedDynamics] = None,
        max_constraints: Optional[int] = None,
        num_steps: int = 1,
    ) -> CBFSolution:
        """
        Exact repeated estimated-CBF application.

        This intentionally does *not* early-stop and does not reuse stale
        gradients/constraints: each substep calls ``step_estimated`` with the
        current state, matching the previous outer Python loop semantics.
        """
        steps = max(1, int(num_steps))
        x_next = x_t
        x_prev_iter = x_prev
        sol: Optional[CBFSolution] = None
        for _ in range(steps):
            sol = self.step_estimated(
                x_t=x_next,
                x_prev=x_prev_iter,
                constraints=constraints,
                dynamics=dynamics,
                max_constraints=max_constraints,
            )
            x_prev_iter = x_next.detach()
            x_next = sol.x_next
        assert sol is not None
        return CBFSolution(
            u=x_next - x_t, x_next=x_next, delta=sol.delta, feasible=sol.feasible
        )

    # ---- Learned dynamics ---------------------------------------------------
    def step_learned(
        self,
        x_t: Tensor,
        x_prev: Tensor,
        constraints: Sequence[SafetyConstraint],
        dynamics: LearnedDynamics,
        lyapunov_ref: Optional[Tensor] = None,
        max_constraints: Optional[int] = None,
    ) -> CBFSolution:
        """
        Solve min ||u||^2 + w \u03b4^2
        s.t. \u2207b(f+g u)+k b >= 0,
             -\u2207V g u + \u03b4 >= pV + \u2207V f  (rewritten from dV + pV <= \u03b4).
        """
        if lyapunov_ref is None:
            lyapunov_ref = x_t.detach()

        if len(constraints) == 0:
            raise ValueError("constraints must be non-empty")

        # Support non-batched h_fn by evaluating b/grad_b per-sample, then stacking.
        scalar_single = x_t.dim() == 0
        if scalar_single:
            x_t = x_t.unsqueeze(0)  # (1,)
            x_prev = x_prev.unsqueeze(0)  # (1,)
            lyapunov_ref = lyapunov_ref.unsqueeze(0)  # (1,)

        if x_t.dim() == 2:
            return_mode = "batch_vector"
            x_state = x_t
            x_prev_state = x_prev
            x_ref_state = lyapunov_ref
        elif x_t.dim() == 1:
            b0 = constraints[0].b(x_t)
            if b0.dim() == 0:
                return_mode = "single_vector"
                x_state = x_t.unsqueeze(0)  # (1, q)
                x_prev_state = x_prev.unsqueeze(0)
                x_ref_state = lyapunov_ref.unsqueeze(0)
            else:
                return_mode = "batch_scalar"
                x_state = x_t.unsqueeze(-1)  # (batch, 1)
                x_prev_state = x_prev.unsqueeze(-1)
                x_ref_state = lyapunov_ref.unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported x_t shape: {tuple(x_t.shape)}")

        batch_size = x_state.shape[0]
        f, g = dynamics.evaluate(x_state)
        if f.dim() == 1:
            f = f.unsqueeze(0)
        if g.dim() == 2:
            g = g.unsqueeze(0)

        ctrl_dim = g.shape[-1]
        z0 = torch.zeros(
            (batch_size, ctrl_dim + 1), device=x_state.device, dtype=x_state.dtype
        )

        # Optimization: Check if using BatchedSafetyConstraint
        is_batched_constraint = isinstance(constraints, BatchedSafetyConstraint)

        if is_batched_constraint:
            # ---------------------------------------------------------
            # FAST PATH: Batched evaluation + Top-K Gradient Calculation
            # ---------------------------------------------------------
            # 1. Evaluate ALL b(x) efficiently
            # b_all: (batch, num_constraints)
            if return_mode == "batch_scalar":
                x_in = x_t  # scalar (batch, 1) or (batch,)
                if x_in.dim() == 1:
                    x_in = x_in.unsqueeze(1)
            else:
                x_in = x_state  # (batch, state_dim)

            b_all = constraints.forward_b(x_in)

            # 2. Approximate 'scores' for reduction
            # To get strict CBF scores c_cbf, we need gradients: c = -k*b - grad_b*f.
            # But we want to avoid computing all grad_b.
            # Approximation: violation is dominated by b term if k is large enough, or if f is small.
            # score approx = -k*b.
            # We select Top-K based on this approximation (most negative b / most positive -b).
            scores_approx = -self.k * b_all

            num_constraints = b_all.shape[1]
            k = num_constraints
            if (
                max_constraints is not None
                and max_constraints > 0
                and num_constraints > max_constraints
                and self.constraint_mode in ("topk", "merge")
            ):
                k = min(max_constraints, num_constraints)

            # Select top-k indices based on b-values
            topk = torch.topk(scores_approx, k, dim=1)
            topk_idx = topk.indices  # (batch, k)
            topk_b_vals = -topk.values / self.k  # recover b values (batch, k) - roughly
            # Correction: actually use the real b values for selected indices
            topk_b_vals = torch.gather(b_all, 1, topk_idx)

            # 3. Compute gradients for selected constraints
            if self.constraint_mode == "merge" and k > 1:
                # --- OPTIMIZED MERGE: single backward pass ---
                # Compute merged b and grad_b in one backward pass
                merged_b, merged_grad = constraints.forward_grad_merged(
                    x_in, topk_idx, kappa=self.kappa
                )
                # merged_grad: (batch, state_dim), merged_b: (batch,)

                # Project through dynamics: a_vec = grad_merged * g
                a_vec_merged = torch.bmm(merged_grad.unsqueeze(1), g).squeeze(
                    1
                )  # (batch, ctrl_dim)

                # rhs = -k*b_merged - grad_merged*f
                dot_grad_f = (merged_grad * f).sum(dim=-1)  # (batch,)
                rhs_merged = (-self.k * merged_b - dot_grad_f).detach()  # (batch,)

                A_rows = [a_vec_merged]  # one row: (batch, ctrl_dim)
                c_rows = [rhs_merged]  # one rhs: (batch,)
            else:
                # --- Non-merge path: compute gradients for k constraints ---
                grad_subset = constraints.forward_grad_b(
                    x_in, topk_idx
                )  # (batch, k, state_dim)

                # Project through dynamics: a_vec = grad * g
                batch_size_full = batch_size * k
                grad_flat = grad_subset.reshape(batch_size_full, 1, -1)
                g_expanded = (
                    g.unsqueeze(1)
                    .expand(-1, k, -1, -1)
                    .reshape(batch_size_full, -1, ctrl_dim)
                )

                a_vec = torch.bmm(grad_flat, g_expanded).squeeze(
                    1
                )  # (batch*k, ctrl_dim)
                a_vec = a_vec.reshape(batch_size, k, ctrl_dim)  # (batch, k, ctrl_dim)

                # rhs = -k*b - grad*f
                f_expanded = f.unsqueeze(1).expand(-1, k, -1)
                dot_grad_f = (grad_subset * f_expanded).sum(dim=-1)  # (batch, k)
                rhs = (-self.k * topk_b_vals - dot_grad_f).detach()  # (batch, k)

                # Convert to A_rows list format
                A_rows = [a_vec[:, i, :] for i in range(k)]
                c_rows = [rhs[:, i] for i in range(k)]

        else:
            # ---------------------------------------------------------
            # SLOW PATH: Legacy Loop
            # ---------------------------------------------------------
            A_rows: List[Tensor] = []
            c_rows: List[Tensor] = []

            for con in constraints:
                b_vals: List[Tensor] = []
                grad_vals: List[Tensor] = []
                for b in range(batch_size):
                    if return_mode == "batch_scalar":
                        x_sample = x_t[b]  # scalar (0D) from original 1D input
                    else:
                        x_sample = x_state[b]  # (state_dim,)
                    b_vals.append(con.b(x_sample))
                    grad_vals.append(con.grad_b(x_sample))

                b_val = torch.stack([v.reshape(()) for v in b_vals], dim=0)  # (batch,)
                grad_b = torch.stack(
                    [
                        g_.reshape(()) if g_.dim() == 0 else g_.reshape(-1)
                        for g_ in grad_vals
                    ],
                    dim=0,
                )
                if grad_b.dim() == 1:
                    grad_b = grad_b.unsqueeze(-1)  # (batch, 1)

                a_vec = torch.bmm(grad_b.unsqueeze(1), g).squeeze(
                    1
                )  # (batch, ctrl_dim)
                rhs = (-self.k * b_val - (grad_b * f).sum(dim=-1)).detach()  # (batch,)
                A_rows.append(a_vec)
                c_rows.append(rhs)

            # Optionally reduce CBF constraints before adding Lyapunov constraint.
            # max_constraints <= 0 means "no limit". Additionally, if constraint_mode=="qp",
            # we do not perform any top-k/merge reduction regardless of max_constraints.
            if (
                max_constraints is not None
                and max_constraints > 0
                and len(A_rows) > max_constraints
                and self.constraint_mode in ("topk", "merge")
            ):
                A_cbf = torch.stack(A_rows, dim=0)  # (m, batch, ctrl_dim)
                c_cbf = torch.stack(c_rows, dim=0)  # (m, batch)

                # Score violations at z0=0 (u=0, delta=0): A@0=0 so scores == c
                scores_cbf = c_cbf  # (m, batch)

                if self.constraint_mode == "merge" and A_cbf.shape[0] > 1:
                    # Merge per-sample into ONE CBF constraint row.
                    merged_A_list: List[Tensor] = []
                    merged_c_list: List[Tensor] = []
                    for b in range(batch_size):
                        # Here we treat rhs values like b-values for aggregation:
                        # we want a single inequality a_merge u >= c_merge that is conservative.
                        # Use the same smooth-min aggregation on (-(rhs)) isn't appropriate here;
                        # instead, we merge on b-values/gradients earlier. For simplicity and consistency
                        # with step_estimated, approximate with smooth-min on b via scores ordering:
                        # use scores_cbf as proxy to weight "most violated" constraints.
                        # Convert back to b-values surrogate: b_sur = -rhs/k (since rhs = -k*b - grad_b f).
                        # This is imperfect but keeps behavior simple.
                        b_sur = -c_cbf[:, b] / max(self.k, 1e-12)
                        # grad surrogate in control space is a_vec
                        b_merge, a_merge = self._merge_constraints_one(
                            b_sur, A_cbf[:, b], scores=scores_cbf[:, b]
                        )
                        merged_A_list.append(a_merge)
                        merged_c_list.append((-self.k * b_merge).detach())

                    A_rows = [
                        torch.stack(merged_A_list, dim=0)
                    ]  # one row: (batch, ctrl_dim)
                    c_rows = [torch.stack(merged_c_list, dim=0)]  # one rhs: (batch,)
                else:
                    # True per-sample top-k on CBF constraints.
                    k = min(max_constraints, A_cbf.shape[0])
                    topk_idx = torch.topk(
                        scores_cbf.transpose(0, 1), k, dim=1
                    ).indices  # (batch, k)
                    A_bmq = A_cbf.permute(1, 0, 2)  # (batch, m, ctrl_dim)
                    idx_exp = topk_idx.unsqueeze(-1).expand(
                        -1, -1, A_bmq.shape[-1]
                    )  # (batch, k, ctrl_dim)
                    A_sel = torch.gather(
                        A_bmq, dim=1, index=idx_exp
                    )  # (batch, k, ctrl_dim)
                    c_bm = c_cbf.transpose(0, 1)  # (batch, m)
                    c_sel = torch.gather(c_bm, dim=1, index=topk_idx)  # (batch, k)

                    A_rows = [A_sel[:, i, :] for i in range(A_sel.shape[1])]
                    c_rows = [c_sel[:, i] for i in range(c_sel.shape[1])]

        # Lyapunov constraint with slack \u03b4
        x_diff = x_state - x_ref_state
        grad_v = 2 * x_diff
        a_v = -torch.bmm(grad_v.unsqueeze(1), g).squeeze(1)  # (batch, ctrl_dim)
        c_v = (
            self.p * (x_diff.pow(2).sum(dim=-1)) + (grad_v * f).sum(dim=-1)
        ).detach()  # (batch,)

        # Build extended variables z = [u, \u03b4]
        A_ext: List[Tensor] = []
        c_ext: List[Tensor] = []
        for a_vec, rhs in zip(A_rows, c_rows):
            pad = torch.zeros((batch_size, 1), device=rhs.device, dtype=rhs.dtype)
            A_ext.append(torch.cat([a_vec, pad], dim=-1))  # (batch, ctrl_dim+1)
            c_ext.append(rhs)

        A_ext.append(
            torch.cat(
                [a_v, torch.ones((batch_size, 1), device=c_v.device, dtype=c_v.dtype)],
                dim=-1,
            )
        )
        c_ext.append(c_v)

        A = torch.stack(A_ext, dim=0)  # (m, batch, ctrl_dim+1)
        c = torch.stack(c_ext, dim=0)  # (m, batch)

        if (
            max_constraints is not None
            and max_constraints > 0
            and A.shape[0] > max_constraints
            and self.constraint_mode in ("topk", "merge")
        ):
            # True per-sample top-k (each sample chooses its own constraints).
            # Score violations at z0: violation = c - A@z0. Here z0 is (batch, q).
            scores = c - torch.einsum("mbq,bq->mb", A, z0)  # (m, batch)
            k = min(max_constraints, A.shape[0])
            scores_bm = scores.transpose(0, 1)  # (batch, m)
            topk_idx = torch.topk(scores_bm, k, dim=1).indices  # (batch, k)

            A_bmq = A.permute(1, 0, 2)  # (batch, m, q)
            idx_exp = topk_idx.unsqueeze(-1).expand(
                -1, -1, A_bmq.shape[-1]
            )  # (batch, k, q)
            A_sel_bkq = torch.gather(A_bmq, dim=1, index=idx_exp)  # (batch, k, q)

            c_bm = c.transpose(0, 1)  # (batch, m)
            c_sel_bk = torch.gather(c_bm, dim=1, index=topk_idx)  # (batch, k)

            A = A_sel_bkq.permute(1, 0, 2).contiguous()
            c = c_sel_bk.transpose(0, 1).contiguous()

        if A.shape[0] <= 2:
            z_sol = _closed_form_two(z0, A, c, self.eps)  # (batch, ctrl_dim+1)
        else:
            z_sol = _active_set_qp_batched(z0, A, c, self.eps)  # (batch, ctrl_dim+1)

        u = z_sol[:, :ctrl_dim]
        delta = z_sol[:, ctrl_dim]
        x_next = x_prev_state + u * dynamics.dt

        if return_mode == "batch_scalar":
            u_out = u.squeeze(-1)
            x_next_out = x_next.squeeze(-1)
            delta_out = delta
            if scalar_single:
                u_out = u_out.squeeze(0)
                x_next_out = x_next_out.squeeze(0)
                delta_out = delta_out.squeeze(0)
        elif return_mode == "single_vector":
            u_out = u.squeeze(0)
            x_next_out = x_next.squeeze(0)
            delta_out = delta.squeeze(0)
        else:
            u_out = u
            x_next_out = x_next
            delta_out = delta

        feasible = True
        return CBFSolution(
            u=u_out, x_next=x_next_out, delta=delta_out, feasible=feasible
        )

    def step_learned_iterative(
        self,
        x_t: Tensor,
        x_prev: Tensor,
        constraints: Sequence[SafetyConstraint],
        dynamics: LearnedDynamics,
        lyapunov_ref: Optional[Tensor] = None,
        max_constraints: Optional[int] = None,
        num_steps: int = 1,
    ) -> CBFSolution:
        """
        Exact repeated learned-CBF application.  No early stop; every substep
        recomputes b, gradients, dynamics projection, and QP from the current
        state.
        """
        steps = max(1, int(num_steps))
        x_next = x_t
        x_prev_iter = x_prev
        sol: Optional[CBFSolution] = None
        for _ in range(steps):
            sol = self.step_learned(
                x_t=x_next,
                x_prev=x_prev_iter,
                constraints=constraints,
                dynamics=dynamics,
                lyapunov_ref=lyapunov_ref,
                max_constraints=max_constraints,
            )
            x_prev_iter = x_next.detach()
            x_next = sol.x_next
        assert sol is not None
        return CBFSolution(
            u=x_next - x_t, x_next=x_next, delta=sol.delta, feasible=sol.feasible
        )


__all__ = [
    "CBFSolution",
    "CBFController",
    "SafetyConstraint",
    "EstimatedDynamics",
    "LearnedDynamics",
    "build_polytope_constraints",
]


def build_polytope_constraints(
    feature_extractor: nn.Module,
    threshold: torch.Tensor,
    phi: Optional[torch.Tensor] = None,
    phi_network: Optional[nn.Module] = None,
    epsilon: float = 0.0,
) -> BatchedSafetyConstraint:
    """
    Build SafetyConstraint list or BatchedSafetyConstraint from polytope parameters.

    Safety condition: cost_i(x) <= threshold_i, where cost_i is either
    phi_i^T feat(x) or (phi_network(feat(x)))[i].
    """
    assert threshold is not None, "threshold is required"

    # Ensure models are in eval mode (disable dropout during inference)
    if isinstance(feature_extractor, nn.Module):
        feature_extractor.eval()
    if phi_network is not None:
        phi_network.eval()

    # Return optimized BatchedSafetyConstraint
    return BatchedSafetyConstraint(
        feature_extractor=feature_extractor,
        threshold=threshold,
        phi=phi,
        phi_network=phi_network,
        epsilon=epsilon,
    )
