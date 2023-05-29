from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    hidden_sizes: List[int]


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
    patience: int
    lr: float
    batch_size: int
    checkpoint_path: str
    model_type: str
    la: LAConfig


@dataclass
class DataConfig:
    seed: int
    path: str
    name: str
    val_size: float
    split_index: int
    gap: bool


@dataclass
class ExperimentConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    trainer: TrainerConfig
