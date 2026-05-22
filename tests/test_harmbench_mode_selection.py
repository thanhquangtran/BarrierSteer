from barriersteer_pipeline.harmbench.run_harmbench_pipeline import (
    _filter_modes_by_selection,
    _mode_type_from_steering,
    _mode_type_from_training,
    _normalize_selection,
    _stage4_mode_uses_generation_seed,
)


def test_normalize_selection_accepts_csv_and_lists():
    assert _normalize_selection(" circuit_breaker, reft_r1 ") == [
        "circuit_breaker",
        "reft_r1",
    ]
    assert _normalize_selection(["cbf_qp", "cbf_merge,base_model"]) == [
        "cbf_qp",
        "cbf_merge",
        "base_model",
    ]
    assert _normalize_selection("") is None


def test_stage3_filter_selects_circuit_breaker_type_only():
    modes = {
        "cbf_topk_nonlinear": {"num_phi": [4]},
        "circuit_breaker": {"model_type": "circuit_breaker"},
        "reft_r1": {"model_type": "reft_r1"},
    }

    filtered = _filter_modes_by_selection(
        modes,
        lambda _name, cfg: _mode_type_from_training(cfg),
        selected_mode_types=["circuit_breaker"],
    )

    assert list(filtered) == ["circuit_breaker"]


def test_stage4_filter_infers_special_mode_types():
    training_modes = {
        "cbf_topk_nonlinear": {"model_type": "conditional_steering_vector"},
        "circuit_breaker": {"model_type": "circuit_breaker"},
        "reft_r1": {"model_type": "reft_r1"},
    }
    steering_modes = {
        "base_model": {"polytope_model_path": ""},
        "cbf_topk_nonlinear": {"cbf": {"use": True}},
        "circuit_breaker": {
            "circuit_breaker": {"use": True, "training_mode": "circuit_breaker"}
        },
        "reft_r1": {"reft_r1": {"use": True, "training_mode": "reft_r1"}},
    }

    filtered = _filter_modes_by_selection(
        steering_modes,
        lambda name, cfg: _mode_type_from_steering(name, cfg, training_modes),
        selected_mode_types=["circuit_breaker"],
    )

    assert list(filtered) == ["circuit_breaker"]
    assert (
        _mode_type_from_steering(
            "base_model", steering_modes["base_model"], training_modes
        )
        == "base_model"
    )
    assert (
        _mode_type_from_steering("reft_r1", steering_modes["reft_r1"], training_modes)
        == "reft_r1"
    )


def test_stage4_seed_invariant_modes_do_not_use_generation_seed():
    training_modes = {
        "actadd": {"model_type": "steering_vector", "steering_method": "actadd"},
        "dirablate": {"model_type": "steering_vector", "steering_method": "dirablate"},
        "cbf_topk_nonlinear": {"model_type": "conditional_steering_vector"},
    }
    steering_modes = {
        "base_model": {"polytope_model_path": ""},
        "self_reminder": {"defense": {"use": True, "method": "self_reminder"}},
        "actadd": {"steering_vector": {"use": True, "method": "actadd"}},
        "dirablate": {"steering_vector": {"use": True, "method": "dirablate"}},
        "cbf_topk_nonlinear": {"cbf": {"use": True}},
    }

    assert not _stage4_mode_uses_generation_seed(
        "base_model", steering_modes["base_model"], training_modes
    )
    assert not _stage4_mode_uses_generation_seed(
        "self_reminder", steering_modes["self_reminder"], training_modes
    )
    assert not _stage4_mode_uses_generation_seed(
        "actadd", steering_modes["actadd"], training_modes
    )
    assert not _stage4_mode_uses_generation_seed(
        "dirablate", steering_modes["dirablate"], training_modes
    )
    assert _stage4_mode_uses_generation_seed(
        "cbf_topk_nonlinear", steering_modes["cbf_topk_nonlinear"], training_modes
    )


def test_mode_name_and_type_filters_intersect():
    modes = {
        "circuit_breaker": {"model_type": "circuit_breaker"},
        "alternate_cb": {"model_type": "circuit_breaker"},
        "reft_r1": {"model_type": "reft_r1"},
    }

    filtered = _filter_modes_by_selection(
        modes,
        lambda _name, cfg: _mode_type_from_training(cfg),
        selected_modes=["alternate_cb"],
        selected_mode_types=["circuit_breaker"],
    )

    assert list(filtered) == ["alternate_cb"]
