from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    architecture: str
    p_drop: float

@dataclass
class LAConfig:
    subset_of_weights: str
    hessian_structure: str
    sigma_noise: float
    prior_precision: float
    prior_mean: float
    selection_strategy: str
    subset_size: int


@dataclass
class TrainerConfig:
    epochs: int
    lr: float
    momentum: float
    weight_decay: float
    milestones: List[int]
    gamma: float
    batch_size: int
    checkpoint_path: str
    model_type: str
    la: LAConfig
   


@dataclass
class DataConfig:
    seed: int
    name: str
    path: str
    workers:int
    val_size: float


@dataclass
class ExperimentConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    trainer: TrainerConfig
