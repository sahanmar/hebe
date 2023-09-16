from dataclasses import dataclass
from enum import Enum


class AcquisitionFunctions(Enum):
    random = "random"


@dataclass(frozen=True)
class NNParametersConfig:
    input_size: int = 2
    hidden_size: int = 100
    dropout: float = 0.5
    learning_rate: float = 0.001
    training_epochs: int = 1000


@dataclass(frozen=True)
class ActiveLearningConfig:
    num_of_instances_to_sample: int = 20
    acquisition_function: AcquisitionFunctions = AcquisitionFunctions.random


@dataclass
class Config:
    nn_parameters: NNParametersConfig = NNParametersConfig()
    active_learning: ActiveLearningConfig = ActiveLearningConfig()
