from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    input_size: int
    hidden_sizes: List[int]
    output_size: int


@dataclass
class LAConfig:
    posthoc: bool
    subset_of_weights: List[str]
    hessian_structure: List[str]
    sigma_noise: float
    prior_precision: float
    prior_mean: float
    epochs: int
    lr: float


@dataclass
class TuningConfig:
    epochs: int
    lr: float


@dataclass
class TrainerConfig:
    epochs: int
    lr: float
    batch_size: int
    checkpoint_path: str
    la: LAConfig
    map_tuning: TuningConfig
    la_tuning: TuningConfig


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
    version: int
    data: DataConfig
    model: ModelConfig
    trainer: TrainerConfig
