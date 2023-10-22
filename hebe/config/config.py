from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class AcquisitionFunctions(Enum):
    random = "random"
    entropy = "entropy"
    hac_entropy = "hac_entropy"


class TrainingType(Enum):
    cold_start = "cold_start"
    warm_start = "warm_start"
    hot_start = "hot_start"


class Dateset(Enum):
    moons = "moons"
    chess = "chess"


@dataclass(frozen=True)
class MCDropoutConfig:
    use_uncertainty_in_prediction: bool = False
    number_of_samples: int = 10


@dataclass
class VadamConfig:
    number_of_samples: int = 10
    optimizer_switch_ratio: float = 0.1
    training_set_size: int = 50
    betas: Tuple[float, float] = (0.9, 0.999)
    std: float = 0.1


@dataclass
class EnsemblesConfig:
    number_of_samples: int = 10


@dataclass(frozen=True)
class NNParametersConfig:
    input_size: int = 2
    hidden_size: int = 150
    dropout: float = 0.5
    learning_rate: float = 0.001
    training_epochs: int = 2000


@dataclass(frozen=True)
class ActiveLearningConfig:
    num_of_instances_to_sample: int = 20
    acquisition_function: AcquisitionFunctions = (
        AcquisitionFunctions.hac_entropy
    )


@dataclass(frozen=True)
class SimulationConfig:
    loops: int = 2  # num of loops for statistical validation
    iterations: int = 15  # active learning iterations
    training_type: TrainingType = TrainingType.hot_start
    data: Dateset = Dateset.chess


@dataclass
class Config:
    nn_parameters: NNParametersConfig = NNParametersConfig()
    active_learning: ActiveLearningConfig = ActiveLearningConfig()
    simulation: SimulationConfig = SimulationConfig()
    mc_dropout: MCDropoutConfig = MCDropoutConfig()
    vadam: VadamConfig = VadamConfig()
    ensembles: EnsemblesConfig = EnsemblesConfig()
