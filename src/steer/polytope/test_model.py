import logging
import os

import hydra
import torch
from omegaconf import DictConfig

from barriersteer_pipeline.data.safety_data import (
    get_dataset,
    get_hidden_states_dataloader,
    get_safety_dataloader,
)
from steer.common.load_util import load_model_and_tokenizer
from steer.common.outputs import evaluate_model
from steer.polytope.lm_constraints import PolytopeConstraint

log = logging.getLogger("test")


@hydra.main(config_path=f"{os.getcwd()}/exp_configs", config_name="test_model_config")
def main(cfg: DictConfig):

    if cfg.dataset.hidden_states_path != "None":
        log.info(f"Loading hidden states from {cfg.dataset.hidden_states_path}")
        hs_data = torch.load(cfg.dataset.hidden_states_path)
        test_dataloader = get_hidden_states_dataloader(hs_data["test"])

        safety_model = PolytopeConstraint(
            model=None,
            tokenizer=None,
            num_phi=cfg.dataset.num_phi,
            train_on_hs=True,
        )
    else:
        _, test_data = get_dataset(cfg.dataset.name_or_path, cfg.dataset)
        model, tokenizer = load_model_and_tokenizer(cfg.model_path)
        safety_model = PolytopeConstraint(model, tokenizer)
        test_dataloader = get_safety_dataloader(
            cfg.dataset.name_or_path,
            test_data,
            batch_size=64,
            dataset_type=cfg.dataset.dataset_type,
        )

    trained_weights = torch.load(cfg.dataset.trained_weights_path)
    safety_model.phi = trained_weights.phi.to(safety_model.device)
    safety_model.threshold = trained_weights.threshold.to(safety_model.device)
    safety_model.feature_extractor = trained_weights.feature_extractor.to(
        safety_model.device
    )

    test_result = evaluate_model(safety_model, test_dataloader, plot_id="test")
    log.info(f"Test acc: {test_result.accuracy:.3f}")
    log.info(f"Test fpr: {test_result.false_positive_rate:.3f}")
    log.info(f"Test fnr: {test_result.false_negative_rate:.3f}")

    torch.save(test_result, "test_outputs.pth")


if "__main__" == __name__:
    main()
