import torch
import random
import os
import json
import hydra
import numpy as np
import logging
import copy
import torch.distributions as dists

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from enum import Enum, auto

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


from laplace import Laplace
from laplace.utils import LargestVarianceDiagLaplaceSubnetMask
from strategies.pruning import OBDSubnetMask
from strategies.kfe import KronckerFactoredEigenSubnetMask
from laplace.curvature import BackPackGGN
from backpack import backpack, extend


from configuration.mnist import ExperimentConfig
from data.image_datasets import get_image_loader
from models.resnets import resnet18, resnet34, resnet50, resnet101
from data.image_datasets import get_image_loader
from image_trainer import train_map, validate_map

cs = ConfigStore.instance()
cs.store(name="experiment_config", node=ExperimentConfig)

log = logging.getLogger(name="mnist")

class ModelType(Enum):
    MAP = auto()
    LA_POSTHOC = auto()


class Strategy(Enum):
    OBD = auto()
    KFE = auto()
    LVD = auto()


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
def create_model(architecture, n_channels, n_classes, p_drop):
    if architecture == 'resnet18':
        model_class = resnet18
    elif architecture == 'resnet34':
        model_class = resnet34
    elif architecture == 'resnet50':
        model_class = resnet50
    elif architecture == 'resnet101':
        model_class = resnet101
    else:
        raise NotImplementedError(f"{architecture} not implemented!")

    model = model_class(num_classes=n_classes,
                        zero_init_residual=True,
                        initial_conv='1x3',
                        input_chanels=n_channels,
                        p_drop=p_drop)

    return model

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def _get_config_file(model_path, model_name):
    return os.path.join(model_path, model_name + ".config")

def _get_model_file(model_path, model_name):
    return os.path.join(model_path, model_name + ".tar")

def save_model(model, model_path, model_name):
    config_dict = model.config
    os.makedirs(model_path, exist_ok=True)
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    with open(config_file, "w") as f:
        json.dump(config_dict, f)
    torch.save(model.state_dict(), model_file)

def load_model(model_path, model_name, net=None):
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)

    assert os.path.isfile(config_file), f"Could not find the config file {config_file}"
    assert os.path.isfile(model_file), f"Could not find the model file {model_file}"

    with open(config_file, "r") as f:
        config_dict = json.load(f)
    if net is None: 
        net = create_model(
            architecture=model_name,
            n_channels=config_dict["input_chanels"],
            n_classes=config_dict["num_classes"],
            p_drop=config_dict["p_drop"]
        )
    net.load_state_dict(torch.load(model_file))
    return net

@hydra.main(config_path="configuration", config_name="mnist", version_base=None)
def main(config: ExperimentConfig) -> None:
    log.info(f"Running experiment with configuration: \n{OmegaConf.to_yaml(config)}")

    set_seed(config.seed) 
    device = get_device()

    log.info(f"Running on device {device}")

    train_loader, test_loader, info = get_image_loader(dataset=config.data.name,
                                                       batch_size=config.trainer.batch_size,
                                                       gpu=True,
                                                       distributed=False,
                                                       workers=config.data.workers,
                                                       data_path=config.data.path)
    
    # model = create_model(architecture=config.model.architecture,
    #                      n_channels=info["n_channels"],
    #                      n_classes=info["n_classes"],
    #                      p_drop=config.model.p_drop)
    
    # model = train_map(model=model,
    #           data_loader=train_loader,
    #           device=device,
    #           epochs=config.trainer.epochs,
    #           lr=config.trainer.lr,
    #           momentum=config.trainer.momentum,
    #           weight_decay=config.trainer.weight_decay,
    #           milestones=config.trainer.milestones,
    #           gamma=config.trainer.gamma)
    
    # save_model(model=model,
    #            model_path=config.trainer.checkpoint_path,
    #            model_name=config.model.architecture)
    
    map_model = load_model(model_path=config.trainer.checkpoint_path,
               model_name=config.model.architecture)
    
    map_model = map_model.to(device=device)
    
    l, ca = validate_map(model=map_model, data_loader=test_loader, device=device)
    log.info(f"Test loss: {l}, classification accuracy: {ca}")

    train_loader, test_loader, info = get_image_loader(dataset=config.data.name,
                                                       batch_size=4,
                                                       gpu=True,
                                                       distributed=False,
                                                       workers=config.data.workers,
                                                       data_path=config.data.path)
    if config.trainer.la.subset_of_weights == "subnetwork":
        model_for_selection = copy.deepcopy(map_model)
        log.info(
            f"Using {config.trainer.la.selection_strategy} strategy for subnetwork selection"
        )
        if config.trainer.la.selection_strategy == Strategy.KFE.name:
            laplace_model_for_selection = Laplace(
                model=model_for_selection,
                likelihood="classification",
                subset_of_weights="all",
                hessian_structure="kron",
                sigma_noise=config.trainer.la.sigma_noise,
                prior_mean=config.trainer.la.prior_mean,
            )

            subnetwork_mask = KronckerFactoredEigenSubnetMask(
                model_for_selection,
                n_params_subnet=config.trainer.la.subset_size,
                kron_laplace_model=laplace_model_for_selection,
            )
        elif config.trainer.la.selection_strategy == Strategy.OBD.name:
            laplace_model_for_selection = Laplace(
                model=model_for_selection,
                likelihood="classification",
                subset_of_weights="all",
                hessian_structure="diag",
                sigma_noise=config.trainer.la.sigma_noise,
                prior_mean=config.trainer.la.prior_mean,
            )

            subnetwork_mask = OBDSubnetMask(
                model_for_selection,
                n_params_subnet=config.trainer.la.subset_size,
                diag_laplace_model=laplace_model_for_selection,
            )
        else:
            laplace_model_for_selection = Laplace(
                model=model_for_selection,
                likelihood="regression",
                subset_of_weights="all",
                hessian_structure="diag",
                sigma_noise=config.trainer.la.sigma_noise,
                prior_mean=config.trainer.la.prior_mean,
            )

            subnetwork_mask = LargestVarianceDiagLaplaceSubnetMask(
                model_for_selection,
                n_params_subnet=config.trainer.la.subset_size,
                diag_laplace_model=laplace_model_for_selection,
            )

        subnetwork_indices = subnetwork_mask.select(train_loader=train_loader)
        
        model_copy = copy.deepcopy(map_model)
        # la, prior_precision = trainer.train_la_posthoc(
        #     model=model_copy,
        #     dataloader=train_dataloader,
        #     subset_of_weights=config.trainer.la.subset_of_weights,
        #     hessian_structure="full",
        #     sigma_noise=sigma,
        #     prior_mean=config.trainer.la.prior_mean,
        #     subnetwork_indices=subnetwork_indices,
        #     val_dataloader=val_dataloader,
        # )
        # nll = trainer.evaluate_la(la, test_dataloader)
        # results["nll"] = nll
        # log.info(f"Test NLL={nll}")
    else:
        model_copy = copy.deepcopy(map_model)
        model_copy = extend(model_copy, use_converter=True)
        
        la = Laplace(model_copy, 'classification',
             subset_of_weights='all',
             hessian_structure='diag',
             )
        la.fit(train_loader)
        targets = torch.cat([y for x, y in test_loader], dim=0)
        targets = targets.to(device, non_blocking=True)
        probs_laplace = predict(test_loader, la, device=device, laplace=True)
        acc_laplace = (probs_laplace.argmax(-1) == targets).float().cpu().numpy().mean()
        nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).cpu().numpy().mean()

        print(f'[Laplace] Acc.: {acc_laplace:.1%}; NLL: {nll_laplace:.3}')


@torch.no_grad()
def predict(dataloader, model, device, laplace=False):
    py = []

    for x, _ in dataloader:
        x = x.to(device, non_blocking=True)
        if laplace:
            py.append(model(x))
        else:
            py.append(torch.softmax(model(x), dim=-1))

    return torch.cat(py)
   
    

if __name__ == "__main__":
    main()
    
    