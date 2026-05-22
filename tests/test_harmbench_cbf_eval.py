import os
import tempfile

import pytest
import torch

from steer.cbf import SafetyConstraint


def test_grad_b_works_under_no_grad():
    """
    Regression test for HarmBench CBF eval:
    generation typically runs under torch.no_grad(), but CBF needs d(b)/d(x).
    """

    # b(x) = x^2 - 1, grad = 2x
    con = SafetyConstraint(h_fn=lambda x: x.pow(2) - 1.0, epsilon=0.0)

    x = torch.tensor(3.0)
    with torch.no_grad():
        g = con.grad_b(x)
    assert torch.allclose(g, torch.tensor(6.0), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_harmbench_cbf_generate_smoke_real_model_and_polytope():
    """
    Realistic (but optional) end-to-end smoke test:
    loads a real HF model + a real polytope weights file and verifies that
    generation does not crash with CBF enabled.

    Configure via env vars (so CI can skip, but you can run locally):
      - SAFETY_POLYTOPE_TEST_HF_MODEL (e.g. meta-llama/Llama-2-7b-chat-hf)
      - SAFETY_POLYTOPE_TEST_POLYTOPE_WEIGHTS (path to weights.pth)
    """

    model_name_or_path = os.environ.get("SAFETY_POLYTOPE_TEST_HF_MODEL")
    weights_path = os.environ.get("SAFETY_POLYTOPE_TEST_POLYTOPE_WEIGHTS")
    if not model_name_or_path or not weights_path:
        pytest.skip(
            "Set SAFETY_POLYTOPE_TEST_HF_MODEL and SAFETY_POLYTOPE_TEST_POLYTOPE_WEIGHTS to run"
        )

    # Import here so CPU-only environments can still import the test module.
    from steer.cbf import CBFController, EstimatedDynamics, build_polytope_constraints
    from steer.polytope.safe_rep_model import SafeRepModel

    device = torch.device("cuda:0")
    m = SafeRepModel.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        polytope_weight_path=weights_path,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    ).eval()

    # Attach CBF using the loaded polytope params
    constraints = build_polytope_constraints(
        feature_extractor=m.feature_extractor,
        threshold=m.threshold,
        phi=getattr(m, "phi", None),
        phi_network=getattr(m, "phi_network", None),
        epsilon=0.0,
    )
    m.configure_cbf(
        controller=CBFController(dt=1.0, k=1.0, w=1.0, p=10.0),
        constraints=constraints,
        mode="estimated",
        dynamics=EstimatedDynamics(dt=1.0),
        max_constraints=2,
    )

    tok = m.tokenizer
    prompt = "Tell me a short, harmless fact."
    inputs = tok(prompt, return_tensors="pt").to(device)

    # generation runs under no_grad in HarmBench; ensure no crash
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=4, do_sample=False)
    assert out.shape[1] > inputs["input_ids"].shape[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_harmbench_cbf_batched_generation():
    """
    Test that batched generation works correctly with CBF.
    Verifies that multiple prompts can be processed together.
    """
    model_name_or_path = os.environ.get("SAFETY_POLYTOPE_TEST_HF_MODEL")
    weights_path = os.environ.get("SAFETY_POLYTOPE_TEST_POLYTOPE_WEIGHTS")
    if not model_name_or_path or not weights_path:
        pytest.skip(
            "Set SAFETY_POLYTOPE_TEST_HF_MODEL and SAFETY_POLYTOPE_TEST_POLYTOPE_WEIGHTS to run"
        )

    from steer.cbf import CBFController, EstimatedDynamics, build_polytope_constraints
    from steer.polytope.safe_rep_model import SafeRepModel

    device = torch.device("cuda:0")
    m = SafeRepModel.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        polytope_weight_path=weights_path,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    ).eval()

    # Attach CBF using the loaded polytope params
    constraints = build_polytope_constraints(
        feature_extractor=m.feature_extractor,
        threshold=m.threshold,
        phi=getattr(m, "phi", None),
        phi_network=getattr(m, "phi_network", None),
        epsilon=0.0,
    )
    m.configure_cbf(
        controller=CBFController(dt=1.0, k=1.0, w=1.0, p=10.0),
        constraints=constraints,
        mode="estimated",
        dynamics=EstimatedDynamics(dt=1.0),
        max_constraints=2,
    )

    tok = m.tokenizer

    # Test with batch of 2 prompts
    prompts = ["Tell me a short, harmless fact.", "What is the capital of France?"]
    inputs = tok(prompts, return_tensors="pt", padding=True).to(device)

    # Batched generation runs under no_grad in HarmBench; ensure no crash
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=4, do_sample=False)

    # Check that we got outputs for both prompts
    assert out.shape[0] == 2  # batch size
    assert out.shape[1] > inputs["input_ids"].shape[1]  # generated tokens

    # Verify that timing info was tracked
    if hasattr(m, "_timing_info"):
        # Timing info should be present after generation
        assert (
            "violated_tokens" in m._timing_info or "steering_overhead" in m._timing_info
        )
