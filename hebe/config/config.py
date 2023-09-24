from dataclasses import dataclass
from enum import Enum


class AcquisitionFunctions(Enum):
    random = "random"
    entropy = "entropy"


class TrainingType(Enum):
    cold_start = "cold_start"
    warm_start = "warm_start"
    hot_start = "hot_start"


@dataclass(frozen=True)
class MCDropoutConfig:
    number_of_samples = 10


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
    acquisition_function: AcquisitionFunctions = AcquisitionFunctions.entropy


@dataclass(frozen=True)
class SimulationConfig:
    loops: int = 2  # num of loops for statistical validation
    iterations: int = 5  # active learning iterations
    training_type: TrainingType = TrainingType.hot_start


@dataclass
class Config:
    nn_parameters: NNParametersConfig = NNParametersConfig()
    active_learning: ActiveLearningConfig = ActiveLearningConfig()
    simulation: SimulationConfig = SimulationConfig()
    mc_dropout: MCDropoutConfig = MCDropoutConfig()
