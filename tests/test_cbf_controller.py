import torch

from steer.cbf import (
    CBFController,
    EstimatedDynamics,
    LearnedDynamics,
    SafetyConstraint,
    build_polytope_constraints,
)


def test_estimated_single_constraint_projects_to_safe_side():
    dt = 1.0
    controller = CBFController(dt=dt, k=1.0)
    dynamics = EstimatedDynamics(dt=dt)

    h = lambda x: x  # b(x) = x, safe if x >= 0
    constraint = SafetyConstraint(h_fn=h, epsilon=0.0)

    x_prev = torch.tensor(0.0)
    x_t = torch.tensor(-1.0)  # unsafe

    sol = controller.step_estimated(
        x_t=x_t, x_prev=x_prev, constraints=[constraint], dynamics=dynamics
    )

    assert torch.allclose(sol.u, torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(sol.x_next, torch.tensor(1.0), atol=1e-5)
    assert sol.feasible


def test_estimated_two_constraints_closed_form():
    dt = 1.0
    controller = CBFController(dt=dt, k=1.0)
    dynamics = EstimatedDynamics(dt=dt)

    # b1: 1 - x >= 0  -> x <= 1
    c1 = SafetyConstraint(h_fn=lambda x: 1.0 - x, epsilon=0.0)
    # b2: x >= 0
    c2 = SafetyConstraint(h_fn=lambda x: x, epsilon=0.0)

    x_prev = torch.tensor(1.0)
    x_t = torch.tensor(1.5)  # violates upper bound
    sol = controller.step_estimated(
        x_t=x_t, x_prev=x_prev, constraints=[c1, c2], dynamics=dynamics
    )

    # Constraint 1 enforces u <= -0.5; nominal u_ref = 0.5 -> projection gives -0.5
    assert torch.allclose(sol.u, torch.tensor(-0.5), atol=1e-5)
    assert torch.allclose(sol.x_next, torch.tensor(0.5), atol=1e-5)
    assert sol.feasible


def test_learned_dynamics_with_lyapunov_and_cbf():
    dt = 1.0
    controller = CBFController(dt=dt, k=1.0, w=1.0, p=10.0)

    # f(x) = 0, g(x) = I (scalar here)
    f_model = lambda x: torch.zeros_like(x)
    g_model = lambda x: (
        torch.eye(1, device=x.device, dtype=x.dtype).unsqueeze(0)
        if x.dim() == 0
        else torch.eye(x.shape[-1], device=x.device, dtype=x.dtype).unsqueeze(0)
    )
    dynamics = LearnedDynamics(f_model=f_model, g_model=g_model, dt=dt)

    h = lambda x: x  # safe if x >= 0
    constraint = SafetyConstraint(h_fn=h, epsilon=0.0)

    x_prev = torch.tensor(-0.6)
    x_t = torch.tensor(-0.5)  # unsafe
    lyap_ref = torch.tensor(0.0)

    sol = controller.step_learned(
        x_t=x_t,
        x_prev=x_prev,
        constraints=[constraint],
        dynamics=dynamics,
        lyapunov_ref=lyap_ref,
    )

    # CBF requires u >= 0.5. Lyapunov imposes u + delta >= 2.5 (here delta carries slack).
    # Minimum of u^2 + delta^2 under those constraints is u = delta = 1.25.
    assert sol.u >= torch.tensor(0.5) - 1e-5
    assert torch.allclose(sol.u, torch.tensor(1.25), atol=1e-3)
    assert torch.allclose(sol.delta, torch.tensor(1.25), atol=1e-3)
    # New state should move toward safety
    assert sol.x_next > x_t
    assert sol.feasible


def test_estimated_three_constraints_active_set():
    dt = 1.0
    controller = CBFController(dt=dt, k=1.0)
    dynamics = EstimatedDynamics(dt=dt)

    # Constraints: x >= 0, x <= 1, x >= -0.2 (third is redundant; triggers active-set path)
    c1 = SafetyConstraint(h_fn=lambda x: x, epsilon=0.0)
    c2 = SafetyConstraint(h_fn=lambda x: 1.0 - x, epsilon=0.0)
    c3 = SafetyConstraint(h_fn=lambda x: x + 0.2, epsilon=0.0)

    x_prev = torch.tensor(1.0)
    x_t = torch.tensor(1.5)  # violates upper bound
    sol = controller.step_estimated(
        x_t=x_t, x_prev=x_prev, constraints=[c1, c2, c3], dynamics=dynamics
    )

    # Active-set should push to the upper boundary: u = -0.5, x_next = 0.5
    assert torch.allclose(sol.u, torch.tensor(-0.5), atol=1e-5)
    assert torch.allclose(sol.x_next, torch.tensor(0.5), atol=1e-5)
    assert sol.feasible


def test_estimated_high_dim_constraints_projection():
    dt = 1.0
    controller = CBFController(dt=dt, k=1.0)
    dynamics = EstimatedDynamics(dt=dt)

    # In 3D, enforce x0 <= 0.5 and weak lower bound on sum(x)
    c_upper = SafetyConstraint(h_fn=lambda x: 0.5 - x[0], epsilon=0.0)
    c_sum = SafetyConstraint(h_fn=lambda x: x.sum() + 0.1, epsilon=0.0)

    x_prev = torch.zeros(3)
    x_t = torch.tensor([2.0, 0.0, 0.0])  # violates upper bound on x0

    sol = controller.step_estimated(
        x_t=x_t, x_prev=x_prev, constraints=[c_upper, c_sum], dynamics=dynamics
    )

    # Projection should clamp first component toward the bound: u0 = -1.5, others unchanged
    expected_u = torch.tensor([-1.5, 0.0, 0.0])
    assert torch.allclose(sol.u, expected_u, atol=1e-4)
    assert torch.allclose(sol.x_next, expected_u, atol=1e-4)
    assert sol.feasible


def test_estimated_linear_constraint_halfspace_projection():
    dt = 1.0
    controller = CBFController(dt=dt, k=1.0)
    dynamics = EstimatedDynamics(dt=dt)

    # Linear constraint: a^T x >= 0.5 with a = [1,2]
    a = torch.tensor([1.0, 2.0])
    h_fn = lambda x: torch.dot(a, x)  # h(x) = a^T x
    constraint = SafetyConstraint(h_fn=h_fn, epsilon=0.5)  # b(x)=a^T x - 0.5

    x_prev = torch.zeros(2)
    x_t = torch.tensor([-0.5, -0.5])  # a^T x = -1.5, needs correction

    sol = controller.step_estimated(
        x_t=x_t, x_prev=x_prev, constraints=[constraint], dynamics=dynamics
    )

    # Active-set considers multiple equivalent projections; ensure feasibility and minimality
    assert (a @ sol.u) >= 0.5 - 1e-6
    assert sol.feasible


def test_polytope_constraint_builder_with_phi():
    dt = 1.0
    controller = CBFController(dt=dt, k=1.0)
    dynamics = EstimatedDynamics(dt=dt)

    class IdentityFeat(torch.nn.Module):
        def forward(self, x):
            return x

    feat = IdentityFeat()
    phi = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    threshold = torch.zeros(2)

    constraints = build_polytope_constraints(
        feature_extractor=feat, threshold=threshold, phi=phi, phi_network=None
    )
    assert len(constraints) == 2

    x_prev = torch.zeros(2)
    x_t = torch.tensor([1.0, -1.0])  # violates first constraint (x0 <= 0)

    sol = controller.step_estimated(
        x_t=x_t, x_prev=x_prev, constraints=constraints, dynamics=dynamics
    )

    expected_u = torch.tensor([-1.0, -1.0])
    assert torch.allclose(sol.u, expected_u, atol=1e-5)
    assert torch.allclose(sol.x_next, expected_u, atol=1e-5)
    assert sol.feasible


def test_max_constraints_limits_to_worst_two():
    dt = 1.0
    controller = CBFController(dt=dt, k=1.0)
    dynamics = EstimatedDynamics(dt=dt)

    # Three constraints; two tight upper bounds, one loose lower bound
    c1 = SafetyConstraint(h_fn=lambda x: -x, epsilon=0.0)  # x <= 0
    c2 = SafetyConstraint(h_fn=lambda x: -x - 0.1, epsilon=0.0)  # x <= -0.1
    c3 = SafetyConstraint(h_fn=lambda x: x + 10.0, epsilon=0.0)  # x >= -10 (inactive)

    x_prev = torch.tensor(0.0)
    x_t = torch.tensor(1.0)  # violates upper bounds

    sol = controller.step_estimated(
        x_t=x_t,
        x_prev=x_prev,
        constraints=[c1, c2, c3],
        dynamics=dynamics,
        max_constraints=2,
    )

    # Tightest bound is x <= -0.1 => u <= -1.1
    assert torch.allclose(sol.u, torch.tensor(-1.1), atol=1e-4)
    assert torch.allclose(sol.x_next, torch.tensor(-1.1), atol=1e-4)
    assert sol.feasible


def test_estimated_batched_single_constraint():
    """Test that batched CBF produces same results as sequential processing."""
    dt = 1.0
    controller = CBFController(dt=dt, k=1.0)
    dynamics = EstimatedDynamics(dt=dt)

    h = lambda x: x  # b(x) = x, safe if x >= 0
    constraint = SafetyConstraint(h_fn=h, epsilon=0.0)

    # Single sample (should work like before)
    x_prev_single = torch.tensor(0.0)
    x_t_single = torch.tensor(-1.0)  # unsafe
    sol_single = controller.step_estimated(
        x_t=x_t_single,
        x_prev=x_prev_single,
        constraints=[constraint],
        dynamics=dynamics,
    )

    # Batched version (batch_size=1)
    x_prev_batch = x_prev_single.unsqueeze(0)  # (1,)
    x_t_batch = x_t_single.unsqueeze(0)  # (1,)
    sol_batch = controller.step_estimated(
        x_t=x_t_batch, x_prev=x_prev_batch, constraints=[constraint], dynamics=dynamics
    )

    # Results should match
    assert torch.allclose(sol_batch.u, sol_single.u, atol=1e-5)
    assert torch.allclose(sol_batch.x_next, sol_single.x_next, atol=1e-5)
    assert sol_batch.feasible == sol_single.feasible


def test_estimated_batched_multiple_samples():
    """Test batched CBF with multiple samples, some safe, some unsafe."""
    dt = 1.0
    controller = CBFController(dt=dt, k=1.0)
    dynamics = EstimatedDynamics(dt=dt)

    h = lambda x: x  # b(x) = x, safe if x >= 0
    constraint = SafetyConstraint(h_fn=h, epsilon=0.0)

    # Batch of 4 samples: 2 unsafe (negative), 2 safe (positive)
    x_prev_batch = torch.tensor([0.0, 0.0, 0.0, 0.0])
    x_t_batch = torch.tensor([-1.0, -0.5, 0.5, 1.0])  # first two unsafe, last two safe

    sol_batch = controller.step_estimated(
        x_t=x_t_batch, x_prev=x_prev_batch, constraints=[constraint], dynamics=dynamics
    )

    # Check that all samples were processed
    assert sol_batch.u.shape == (4,)
    assert sol_batch.x_next.shape == (4,)
    assert sol_batch.feasible

    # Unsafe samples should be corrected (u > 0)
    assert sol_batch.u[0] > 0  # -1.0 -> corrected
    assert sol_batch.u[1] > 0  # -0.5 -> corrected
    # Safe samples should have minimal correction (u ≈ 0)
    assert sol_batch.u[2] < 0.1  # 0.5 -> safe, minimal correction
    assert sol_batch.u[3] < 0.1  # 1.0 -> safe, minimal correction


def test_estimated_batched_vs_sequential_consistency():
    """Test that batched processing produces same results as sequential."""
    dt = 1.0
    controller = CBFController(dt=dt, k=1.0)
    dynamics = EstimatedDynamics(dt=dt)

    # b1: 1 - x >= 0  -> x <= 1
    c1 = SafetyConstraint(h_fn=lambda x: 1.0 - x, epsilon=0.0)
    # b2: x >= 0
    c2 = SafetyConstraint(h_fn=lambda x: x, epsilon=0.0)

    # Test samples
    x_prev_vals = torch.tensor([1.0, 0.5, 0.0])
    x_t_vals = torch.tensor(
        [1.5, 0.5, -0.5]
    )  # first violates upper, third violates lower

    # Sequential processing
    sols_sequential = []
    for i in range(len(x_t_vals)):
        sol = controller.step_estimated(
            x_t=x_t_vals[i : i + 1],
            x_prev=x_prev_vals[i : i + 1],
            constraints=[c1, c2],
            dynamics=dynamics,
        )
        sols_sequential.append(sol)

    # Batched processing
    sol_batch = controller.step_estimated(
        x_t=x_t_vals,
        x_prev=x_prev_vals,
        constraints=[c1, c2],
        dynamics=dynamics,
    )

    # Compare results
    for i in range(len(x_t_vals)):
        assert torch.allclose(sol_batch.u[i], sols_sequential[i].u.squeeze(), atol=1e-4)
        assert torch.allclose(
            sol_batch.x_next[i], sols_sequential[i].x_next.squeeze(), atol=1e-4
        )


def test_estimated_batched_high_dim():
    """Test batched CBF with high-dimensional states."""
    dt = 1.0
    controller = CBFController(dt=dt, k=1.0)
    dynamics = EstimatedDynamics(dt=dt)

    # In 3D, enforce x0 <= 0.5
    c_upper = SafetyConstraint(h_fn=lambda x: 0.5 - x[0], epsilon=0.0)

    # Batch of 3 samples in 3D
    x_prev_batch = torch.zeros(3, 3)
    x_t_batch = torch.tensor(
        [
            [2.0, 0.0, 0.0],  # violates upper bound on x0
            [0.3, 0.0, 0.0],  # safe
            [1.0, 0.0, 0.0],  # violates upper bound on x0
        ]
    )

    sol_batch = controller.step_estimated(
        x_t=x_t_batch, x_prev=x_prev_batch, constraints=[c_upper], dynamics=dynamics
    )

    # Check shapes
    assert sol_batch.u.shape == (3, 3)
    assert sol_batch.x_next.shape == (3, 3)
    assert sol_batch.feasible

    # First and third samples should have negative u[0] (correction)
    assert sol_batch.u[0, 0] < -1.0  # 2.0 -> needs correction
    assert sol_batch.u[2, 0] < -0.4  # 1.0 -> needs correction
    # Second sample should have minimal correction
    assert abs(sol_batch.u[1, 0]) < 0.1  # 0.3 -> safe


def test_learned_batched_dynamics():
    """Test batched CBF with learned dynamics."""
    dt = 1.0
    controller = CBFController(dt=dt, k=1.0, w=1.0, p=10.0)

    # f(x) = 0, g(x) = I
    f_model = lambda x: torch.zeros_like(x)
    g_model = lambda x: (
        torch.eye(x.shape[-1], device=x.device, dtype=x.dtype).unsqueeze(0)
        if x.dim() == 1
        else torch.eye(x.shape[-1], device=x.device, dtype=x.dtype).expand(
            x.shape[0], -1, -1
        )
    )
    dynamics = LearnedDynamics(f_model=f_model, g_model=g_model, dt=dt)

    h = lambda x: x  # safe if x >= 0
    constraint = SafetyConstraint(h_fn=h, epsilon=0.0)

    # Batch of 2 samples
    x_prev_batch = torch.tensor([-0.6, -0.3])
    x_t_batch = torch.tensor([-0.5, 0.1])  # first unsafe, second safe
    lyap_ref = torch.tensor([0.0, 0.0])

    sol_batch = controller.step_learned(
        x_t=x_t_batch,
        x_prev=x_prev_batch,
        constraints=[constraint],
        dynamics=dynamics,
        lyapunov_ref=lyap_ref,
    )

    # Check shapes
    assert sol_batch.u.shape == (2,)
    assert sol_batch.x_next.shape == (2,)
    assert sol_batch.delta is not None
    assert sol_batch.delta.shape == (2,)
    assert sol_batch.feasible

    # First sample (unsafe) should have larger correction
    assert sol_batch.u[0] > 0.4
    # Second sample (safe) should have minimal correction
    assert abs(sol_batch.u[1]) < 0.2
