import hydra
import numpy as np
import logging
import os
import math

import torch
import copy
import pickle


from enum import Enum, auto
from laplace.utils import LargestVarianceDiagLaplaceSubnetMask

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from configuration.config import ExperimentConfig
from models.nets import create_mlp
from data.uci_datasets import UCIData
from trainer import ModelTrainer
from metrics import nll_bayesian, nll_map

cs = ConfigStore.instance()
cs.store(name="experiment_config", node=ExperimentConfig)

log = logging.getLogger(name="main")


class ModelType(Enum):
    MAP = auto()
    LA_POSTHOC = auto()
    LA_MARGLIK = auto()


def model_path(checkpoint_path: str, version: int, model_name: str):
    version = f"version_{version}"
    return os.path.join(checkpoint_path, version, model_name)


def get_model_name(
    model_type: ModelType,
    subset_of_weights: str = None,
    hessian_structure: str = None,
):
    if model_type == ModelType.MAP:
        return f"{model_type.name}.pt"
    else:
        return f"{model_type.name}_{subset_of_weights}_{hessian_structure}.pt"


def save_laplace(la, filepath):
    with open(filepath, "wb") as output:
        pickle.dump(la, output)


def load_laplace(filepath):
    with open(filepath, "rb") as input:
        la = pickle.load(input)
    return la


def save_model(model, filepath):
    torch.save(model, filepath)


def load_model(filepath):
    model = torch.load(filepath)
    return model


@hydra.main(config_path="configuration", config_name="uci", version_base=None)
def main(config: ExperimentConfig) -> None:
    seed = 55
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.makedirs(
        os.path.join(config.trainer.checkpoint_path, f"version_{config.version}"),
        exist_ok=True,
    )

    data = UCIData(config.data.path)

    train_dataloader, val_dataloader, test_dataloader = data.get_dataloaders(
        dataset=config.data.name,
        batch_size=config.trainer.batch_size,
        seed=config.data.seed,
        val_size=config.data.val_size,
        split_index=config.data.split_index,
        gap=config.data.gap,
    )

    map_model_path = model_path(
        config.trainer.checkpoint_path,
        config.version,
        get_model_name(ModelType.MAP),
    )

    log.info("Training MAP model")
    model = create_mlp(
        input_size=config.model.input_size,
        hidden_sizes=config.model.hidden_sizes,
        output_size=config.model.output_size,
    )
    model = model.double()

    trainer = ModelTrainer(config.trainer)

    map_model, log_sigma, delta = trainer.train_map(
        model=model,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    prior_precision = (
        delta * train_dataloader.dataset.X.shape[0] * config.model.output_size
    )

    log.info(f"Using prior precision={prior_precision}")
    log.info(f"Using sigma={log_sigma.exp()}")

    map_nll = nll_map(
        model=map_model, sigma=log_sigma.exp(), dataloader=test_dataloader
    )
    log.info(f"MAP NLL={map_nll}")

    model_type = (
        ModelType.LA_POSTHOC if config.trainer.la.posthoc else ModelType.LA_MARGLIK
    )

    log.info(f"Training {model_type.name} model")

    model_copy = copy.deepcopy(map_model)

    la = trainer.train_la_posthoc(
        model=model_copy,
        dataloader=train_dataloader,
        subset_of_weights="all",
        hessian_structure="kron",
        sigma_noise=log_sigma,
        prior_precision=prior_precision,
        prior_mean=config.trainer.la.prior_mean,
    )

    la_diag = trainer.train_la_posthoc(
        model=model_copy,
        dataloader=train_dataloader,
        subset_of_weights="all",
        hessian_structure="diag",
        sigma_noise=log_sigma.exp,
        prior_precision=prior_precision,
        prior_mean=config.trainer.la.prior_mean,
    )
    trainer.fit_sigma_noise_la(model=la_diag, log_sigma=-1.0)

    subnetwork_mask = LargestVarianceDiagLaplaceSubnetMask(
        map_model, n_params_subnet=128, diag_laplace_model=la_diag
    )
    subnetwork_mask.select(train_loader=train_dataloader)
    subnetwork_indices = subnetwork_mask.indices

    for subset_of_weights in config.trainer.la.subset_of_weights:
        for hessian_structure in config.trainer.la.hessian_structure:
            model_copy = copy.deepcopy(map_model)
            if model_type == ModelType.LA_POSTHOC:
                la = trainer.train_la_posthoc(
                    model=model_copy,
                    dataloader=train_dataloader,
                    subset_of_weights=subset_of_weights,
                    hessian_structure=hessian_structure,
                    sigma_noise=log_sigma.exp,
                    prior_precision=prior_precision,
                    prior_mean=config.trainer.la.prior_mean,
                    subnetwork_indices=subnetwork_indices,
                )
            else:
                la = trainer.train_la_marglik(
                    model=model_copy,
                    train_dataloader=train_dataloader,
                    hessian_structure=hessian_structure,
                )
            # trainer.fit_sigma_noise_la(model=la, log_sigma=-1.0)
            la_nll = nll_bayesian(model=la, dataloader=test_dataloader)
            log.info(
                f"LA on weights={subset_of_weights} and with hessian_structure={hessian_structure} achieved nll={la_nll}"
            )


if __name__ == "__main__":
    main()
