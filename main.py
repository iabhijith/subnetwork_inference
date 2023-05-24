import hydra
import numpy as np
import logging
import os
import random

import torch
import copy
import pickle

from laplace import Laplace
from enum import Enum, auto
from pathlib import Path
from collections import defaultdict
from laplace.utils import LargestVarianceDiagLaplaceSubnetMask
from strategies.pruning import OBDSubnetMask

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="configuration", config_name="uci", version_base=None)
def main(config: ExperimentConfig) -> None:
    set_seed(config.seed)
    os.makedirs(
        os.path.join(config.trainer.checkpoint_path, f"version_{config.version}"),
        exist_ok=True,
    )

    data = UCIData(config.data.path)
    meta_data = data.get_metadata()
    results = defaultdict(list)
    for split_index in range(meta_data["wine_gap"]["n_splits"]):
        train_dataloader, val_dataloader, test_dataloader = data.get_dataloaders(
            dataset=config.data.name,
            batch_size=config.trainer.batch_size,
            seed=config.data.seed,
            val_size=config.data.val_size,
            split_index=split_index,
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

        map_model, sigma = trainer.train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )
        log.info(f"Using sigma={sigma}")

        nll, err, count = trainer.evaluate(
            model=map_model, sigma=sigma, dataloader=test_dataloader
        )
        log.info(f"Test NLL={nll}, Test Err={err}, Test Count={count}")
        results["map"].append(nll)

        model_copy = copy.deepcopy(map_model)

        la, prior_precision = trainer.train_la_posthoc(
            model=model_copy,
            dataloader=train_dataloader,
            subset_of_weights="last_layer",
            hessian_structure="full",
            sigma_noise=sigma,
            prior_mean=config.trainer.la.prior_mean,
            val_dataloader=val_dataloader,
        )
        nll = trainer.evaluate_la(la, test_dataloader)
        results["last_layer_full"].append(nll)

        model_copy = copy.deepcopy(map_model)
        la, prior_precision = trainer.train_la_posthoc(
            model=model_copy,
            dataloader=train_dataloader,
            subset_of_weights="last_layer",
            hessian_structure="diag",
            sigma_noise=sigma,
            prior_mean=config.trainer.la.prior_mean,
            val_dataloader=val_dataloader,
        )
        nll = trainer.evaluate_la(la, test_dataloader)
        results["last_layer_diag"].append(nll)

        model_copy = copy.deepcopy(map_model)
        la, prior_precision = trainer.train_la_posthoc(
            model=model_copy,
            dataloader=train_dataloader,
            subset_of_weights="all",
            hessian_structure="diag",
            sigma_noise=sigma,
            prior_mean=config.trainer.la.prior_mean,
            val_dataloader=val_dataloader,
        )
        nll = trainer.evaluate_la(la, test_dataloader)
        results["all_diag"].append(nll)

        model_copy = copy.deepcopy(map_model)
        la, prior_precision = trainer.train_la_posthoc(
            model=model_copy,
            dataloader=train_dataloader,
            subset_of_weights="all",
            hessian_structure="kron",
            sigma_noise=sigma,
            prior_mean=config.trainer.la.prior_mean,
            val_dataloader=val_dataloader,
        )
        nll = trainer.evaluate_la(la, test_dataloader)
        results["all_kron"].append(nll)

        for n_params_subnet in [600, 1200, 1800]:
            model_for_diag = copy.deepcopy(map_model)
            diag_laplace_model = Laplace(
                model=model_for_diag,
                likelihood="regression",
                subset_of_weights="all",
                hessian_structure="diag",
                sigma_noise=sigma,
                prior_mean=config.trainer.la.prior_mean,
            )

            subnetwork_mask = LargestVarianceDiagLaplaceSubnetMask(
                model_for_diag,
                n_params_subnet=n_params_subnet,
                diag_laplace_model=diag_laplace_model,
            )
            subnetwork_mask.select(train_loader=train_dataloader)
            subnetwork_indices = subnetwork_mask.indices
            model_copy = copy.deepcopy(map_model)
            la, prior_precision = trainer.train_la_posthoc(
                model=model_copy,
                dataloader=train_dataloader,
                subset_of_weights="subnetwork",
                hessian_structure="full",
                sigma_noise=sigma,
                prior_mean=config.trainer.la.prior_mean,
                subnetwork_indices=subnetwork_indices,
                val_dataloader=val_dataloader,
            )

            nll = trainer.evaluate_la(la, test_dataloader)
            results[f"subnetwork_{n_params_subnet}"].append(nll)

    for key, value in results.items():
        log.info(f"{key} nll:{np.array(value).mean()}")

        # model_copy = copy.deepcopy(map_model)
        # subnetwork_mask = LargestVarianceDiagLaplaceSubnetMask(
        #     model_copy, n_params_subnet=128, diag_laplace_model=
        # )
        # subnetwork_mask.select(train_loader=train_dataloader)
        # subnetwork_indices = subnetwork_mask.indices

        # for subset_of_weights in config.trainer.la.subset_of_weights:
        #     for hessian_structure in config.trainer.la.hessian_structure:
        #         model_copy = copy.deepcopy(map_model)
        #         if model_type == ModelType.LA_POSTHOC:
        #             la = trainer.train_la_posthoc(
        #                 model=model_copy,
        #                 dataloader=train_dataloader,
        #                 subset_of_weights=subset_of_weights,
        #                 hessian_structure=hessian_structure,
        #                 sigma_noise=log_sigma.exp,
        #                 prior_precision=prior_precision,
        #                 prior_mean=config.trainer.la.prior_mean,
        #                 subnetwork_indices=subnetwork_indices,
        #             )
        #         else:
        #             la = trainer.train_la_marglik(
        #                 model=model_copy,
        #                 train_dataloader=train_dataloader,
        #                 hessian_structure=hessian_structure,
        #             )
        #         # trainer.fit_sigma_noise_la(model=la, log_sigma=-1.0)
        #         la_nll = nll_bayesian(model=la, dataloader=test_dataloader)
        #         log.info(
        #             f"LA on weights={subset_of_weights} and with hessian_structure={hessian_structure} achieved nll={la_nll}"
        #         )


if __name__ == "__main__":
    main()
