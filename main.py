import hydra
import numpy as np
import logging
import os
import random

import torch
import copy
import json

from laplace import Laplace
from enum import Enum, auto
from pathlib import Path
from collections import defaultdict
from laplace.utils import LargestVarianceDiagLaplaceSubnetMask, LargestMagnitudeSubnetMask
from strategies.pruning import OBDSubnetMask, SPRSubnetMask, MNSubnetMask
from strategies.kfe import KronckerFactoredEigenSubnetMask

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from configuration.config import ExperimentConfig
from models.nets import create_mlp
from data.uci_datasets import UCIData
from regression_trainer import ModelTrainer
from metrics import nll_bayesian, nll_map

cs = ConfigStore.instance()
cs.store(name="experiment_config", node=ExperimentConfig)

log = logging.getLogger(name="main")


class ModelType(Enum):
    MAP = auto()
    LA_POSTHOC = auto()


class Strategy(Enum):
    OBD = auto()
    KFE = auto()
    LVD = auto()
    SPR = auto()
    MN = auto()
    LMS = auto()



def model_paths(config: ExperimentConfig):
    """Get the model path and model meta path for the given configuration.
    Parameters
    ----------
    config : ExperimentConfig
        The experiment configuration.
    Returns
    -------
    model_file_path : str
        The model file path.
    model_metadata_file_path : str
        The model metadata file path.
    """
    checkpoint_path = Path(config.trainer.checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model_file_name = f"{config.data.name}_{config.data.split}_{config.data.split_index}_{ModelType.MAP.name}"
    model_metadata_file_name = model_file_name + f"_version_{config.seed}.json"
    model_file_name += f"_version_{config.seed}.pt"
    return checkpoint_path.joinpath(model_file_name), checkpoint_path.joinpath(
        model_metadata_file_name
    )


def results_file(config: ExperimentConfig):
    """Get the results file name for the given configuration.
    Parameters
    ----------
    config : ExperimentConfig
        The experiment configuration.
    Returns
    -------
    results_file_name : str
        The results file name.
    """
    results_path = Path(config.trainer.results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    results_file_name = f"{config.data.name}_{config.data.split}_{config.data.split_index}_{config.trainer.model_type}"
    if config.trainer.model_type == ModelType.LA_POSTHOC.name:
        results_file_name += f"_{config.trainer.la.subset_of_weights}_{config.trainer.la.hessian_structure}"
        if config.trainer.la.subset_of_weights == "subnetwork":
            results_file_name += f"_{config.trainer.la.selection_strategy}_{config.trainer.la.subset_size}"

    results_file_name += f"_version_{config.seed}.json"
    return results_path.joinpath(results_file_name)


def save_model(model, filepath):
    """Save the model to the given file path.
    Parameters
    ----------
    model : torch.nn.Module
        The model to save.
    filepath : str
        The file path to save the model to.
    """
    torch.save(model, filepath)


def load_model(filepath):
    """Load the model from the given file path.
    Parameters
    ----------
    filepath : str
        The file path to load the model from.
    Returns
    -------
    model : torch.nn.Module
        The loaded model.
    """
    model = torch.load(filepath)
    return model


def save_json(d, filepath):
    """Save the dictionary to the given file path.
    Parameters
    ----------
    d : dict
        The dictionary to save.
    filepath : str
        The file path to save the dictionary to.
    """
    with open(filepath, "w") as f:
        json.dump(d, f)


def load_json(filepath):
    """Load the dictionary from the given file path.
    Parameters
    ----------
    filepath : str
        The file path to load the dictionary from.
    Returns
    -------
    d : dict
        The loaded dictionary.
    """
    with open(filepath, "r") as f:
        d = json.load(f)
    return d


def set_seed(seed: int) -> None:
    """Set the seed for reproducibility.
    Parameters
    ----------
    seed : int
        The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the device to use.
    Returns
    -------
    device : torch.device
        The device to use.
    """
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def initialize_results(config: ExperimentConfig):
    """Initialize the results dictionary.
    Parameters
    ----------
    config : ExperimentConfig
        The experiment configuration.
    Returns
    -------
    results : dict
        The results dictionary.
    """
    results = {
        "dataset": config.data.name,
        "seed": config.seed,
        "split": config.data.split,
        "split_index": config.data.split_index,
        "model_type": config.trainer.model_type,
    }
    if config.trainer.model_type == ModelType.LA_POSTHOC.name:
        results["subset_of_weights"] = config.trainer.la.subset_of_weights
        results["hessian_structure"] = config.trainer.la.hessian_structure
        if config.trainer.la.subset_of_weights == "subnetwork":
            results["selection_strategy"] = config.trainer.la.selection_strategy
            results["subset_size"] = config.trainer.la.subset_size
        else:
            results["selection_strategy"] = None
            results["subset_size"] = None
    else:
        results["subset_size"] = 0
        results["subset_of_weights"] = None
        results["hessian_structure"] = None
        results["selection_strategy"] = None

    return results


@hydra.main(config_path="configuration", config_name="uci", version_base=None)
def main(config: ExperimentConfig) -> None:
    set_seed(config.seed)
    prior_precisions = np.logspace(0.1, 1, num=5, base=10).tolist()[:-1]  + np.logspace(1, 2, num=10, base=10).tolist()

    device = get_device()
    log.info(f"Using device: {device}")

    model_file_path, model_meta_path = model_paths(config)
    results_file_path = results_file(config)
    results = initialize_results(config)

    model_type = (
        ModelType.MAP
        if config.trainer.model_type == ModelType.MAP.name
        else ModelType.LA_POSTHOC
    )

    if model_type == ModelType.LA_POSTHOC.name and not model_file_path.exists():
        raise Exception(
            f"Model file {model_file_path} does not exist. Please train the MAP models first."
        )

    data = UCIData(config.data.path)
    meta_data = data.get_metadata()

    train_dataloader, val_dataloader, test_dataloader = data.get_dataloaders(
        dataset=config.data.name,
        batch_size=config.trainer.batch_size,
        seed=config.data.seed,
        val_size=config.data.val_size,
        split_index=config.data.split_index,
        gap=(config.data.split == "GAP"),
    )

    trainer = ModelTrainer(config.trainer, device=device)
    log.info(f"Training {model_type.name} model")
    if model_type == ModelType.MAP:
        model = create_mlp(
            input_size=meta_data[config.data.name]["input_dim"],
            hidden_sizes=config.model.hidden_sizes,
            output_size=meta_data[config.data.name]["output_dim"],
        )
        model = model.to(device=device, dtype=torch.float64)
        map_model, sigma = trainer.train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )
        save_model(map_model, model_file_path)
        hyperparams = {"sigma": sigma}
        model_for_tuning = copy.deepcopy(map_model)
        la, prior_precision_diag = trainer.train_la_posthoc(
                                    model=model_for_tuning,
                                    dataloader=train_dataloader,
                                    subset_of_weights="all",
                                    hessian_structure="diag",
                                    sigma_noise=sigma,
                                    prior_mean=config.trainer.la.prior_mean,
                                    val_dataloader=val_dataloader,
                                    prior_precisions=prior_precisions
                                    )
        
        hyperparams["prior_precision_diag"] = prior_precision_diag
        model_for_tuning = copy.deepcopy(map_model)
        la, prior_precision_kron = trainer.train_la_posthoc(
                                    model=model_for_tuning,
                                    dataloader=train_dataloader,
                                    subset_of_weights="all",
                                    hessian_structure="kron",
                                    sigma_noise=sigma,
                                    prior_mean=config.trainer.la.prior_mean,
                                    val_dataloader=val_dataloader,
                                    prior_precisions=prior_precisions
                                    )
        hyperparams["prior_precision_kron"] = prior_precision_kron
        save_json(hyperparams, model_meta_path)

        nll, err = trainer.evaluate(
            model=map_model, sigma=sigma, dataloader=test_dataloader
        )
        log.info(f"Test NLL={nll}, Test RMSE={err}")
        results["nll"] = nll
    else:
        map_model = load_model(model_file_path)
        hyperparams = load_json(model_meta_path)
        sigma = hyperparams["sigma"]
        prior_precision_diag = hyperparams["prior_precision_diag"]
        prior_precision_kron =  hyperparams["prior_precision_kron"]

        if config.trainer.la.subset_of_weights == "subnetwork":
            model_for_selection = copy.deepcopy(map_model)
            log.info(
                f"Using {config.trainer.la.selection_strategy} strategy for subnetwork selection"
            )
            if config.trainer.la.selection_strategy == Strategy.KFE.name:
                laplace_model_for_selection = Laplace(
                    model=model_for_selection,
                    likelihood="regression",
                    subset_of_weights="all",
                    hessian_structure="kron",
                    sigma_noise=sigma,
                    prior_mean=config.trainer.la.prior_mean,
                    prior_precision=prior_precision_kron
                )

                subnetwork_mask = KronckerFactoredEigenSubnetMask(
                    model_for_selection,
                    n_params_subnet=config.trainer.la.subset_size,
                    kron_laplace_model=laplace_model_for_selection,
                )
            elif config.trainer.la.selection_strategy == Strategy.OBD.name:
                laplace_model_for_selection = Laplace(
                    model=model_for_selection,
                    likelihood="regression",
                    subset_of_weights="all",
                    hessian_structure="diag",
                    sigma_noise=sigma,
                    prior_mean=config.trainer.la.prior_mean,
                    prior_precision=prior_precision_diag
                )

                subnetwork_mask = OBDSubnetMask(
                    model_for_selection,
                    n_params_subnet=config.trainer.la.subset_size,
                    diag_laplace_model=laplace_model_for_selection,
                )
            elif config.trainer.la.selection_strategy == Strategy.SPR.name:
                laplace_model_for_selection = Laplace(
                    model=model_for_selection,
                    likelihood="regression",
                    subset_of_weights="all",
                    hessian_structure="diag",
                    sigma_noise=sigma,
                    prior_mean=config.trainer.la.prior_mean,
                    prior_precision=prior_precision_diag
                )

                subnetwork_mask = SPRSubnetMask(
                    model_for_selection,
                    n_params_subnet=config.trainer.la.subset_size,
                    diag_laplace_model=laplace_model_for_selection,
                )
            elif config.trainer.la.selection_strategy == Strategy.MN.name:
                laplace_model_for_selection = Laplace(
                    model=model_for_selection,
                    likelihood="regression",
                    subset_of_weights="all",
                    hessian_structure="diag",
                    sigma_noise=sigma,
                    prior_mean=config.trainer.la.prior_mean,
                    prior_precision=prior_precision_diag
                )

                subnetwork_mask = MNSubnetMask(
                    model_for_selection,
                    n_params_subnet=config.trainer.la.subset_size,
                    diag_laplace_model=laplace_model_for_selection,
                )
            elif config.trainer.la.selection_strategy == Strategy.LMS.name:
                subnetwork_mask = LargestMagnitudeSubnetMask(
                    model_for_selection,
                    n_params_subnet=config.trainer.la.subset_size,
                )
            else:
                laplace_model_for_selection = Laplace(
                    model=model_for_selection,
                    likelihood="regression",
                    subset_of_weights="all",
                    hessian_structure="diag",
                    sigma_noise=sigma,
                    prior_mean=config.trainer.la.prior_mean,
                    prior_precision=prior_precision_diag
                )

                subnetwork_mask = LargestVarianceDiagLaplaceSubnetMask(
                    model_for_selection,
                    n_params_subnet=config.trainer.la.subset_size,
                    diag_laplace_model=laplace_model_for_selection,
                )

            subnetwork_indices = subnetwork_mask.select(train_loader=train_dataloader)
            subnetwork_indices = torch.tensor(
                subnetwork_indices.cpu().numpy(), dtype=torch.long
            )

            model_copy = copy.deepcopy(map_model)
            la, prior_precision = trainer.train_la_posthoc(
                model=model_copy,
                dataloader=train_dataloader,
                subset_of_weights=config.trainer.la.subset_of_weights,
                hessian_structure="full",
                sigma_noise=sigma,
                prior_mean=config.trainer.la.prior_mean,
                subnetwork_indices=subnetwork_indices,
                val_dataloader=val_dataloader,
                prior_precisions=prior_precisions
            )
            nll = trainer.evaluate_la(la, test_dataloader)
            results["nll"] = nll
            log.info(f"Test NLL={nll}")
        else:
            model_copy = copy.deepcopy(map_model)
            la, prior_precision = trainer.train_la_posthoc(
                model=model_copy,
                dataloader=train_dataloader,
                subset_of_weights=config.trainer.la.subset_of_weights,
                hessian_structure=config.trainer.la.hessian_structure,
                sigma_noise=sigma,
                prior_mean=config.trainer.la.prior_mean,
                val_dataloader=val_dataloader,
                prior_precisions=prior_precisions
            )
            nll = trainer.evaluate_la(la, test_dataloader)
            results["nll"] = nll
            log.info(f"Test NLL={nll}")

    save_json(results, results_file_path)


if __name__ == "__main__":
    main()
