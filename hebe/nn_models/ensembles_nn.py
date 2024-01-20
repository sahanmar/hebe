import torch
from torch.utils.data import DataLoader

from hebe.config import (
    AcquisitionFunctions,
    ActiveLearningConfig,
    EnsemblesConfig,
    NNParametersConfig,
)
from hebe.nn_models.feed_forward_nn import ACQUISITION_FUNCTIONS_MAP, Classifier
from hebe.nn_models.utils import ModelType


class EnsemblesClassifier:
    def __init__(
        self,
        nn_config: NNParametersConfig,
        active_learning_config: ActiveLearningConfig,
        ensembles_config: EnsemblesConfig,
    ):
        self.type = ModelType.ENSEMBLES
        self.ensembles_config = ensembles_config
        self.nn_config = nn_config
        self.active_learning_config = active_learning_config

        self.model: list[Classifier] = [
            Classifier(self.nn_config, self.active_learning_config)
            for i in range(self.ensembles_config.number_of_samples)
        ]

    def train(self, dataloader: DataLoader) -> None:
        for i, ensemble in enumerate(self.model):
            print(f"Training {i}-th ensemble...")
            ensemble.train(dataloader)

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        predictions = torch.stack(
            [ensemble.predict(input_data) for ensemble in self.model], dim=1
        )
        return torch.mean(predictions, dim=1)

    def sample_indices_from_unlabeled_data(
        self,
        input_data: torch.Tensor,
        num_of_instances_to_sample: int | None = None,
    ) -> torch.Tensor:
        """
        Returns sampled instances indices in the original dataset
        """

        acq_func = ACQUISITION_FUNCTIONS_MAP[
            self.active_learning_config.acquisition_function
        ]

        if (
            self.active_learning_config.acquisition_function
            is AcquisitionFunctions.random
        ):
            acq_func(
                input_data,
                num_of_instances_to_sample
                or self.active_learning_config.num_of_instances_to_sample,
            )

        predictions = self.predict(input_data)

        if (
            self.active_learning_config.acquisition_function
            is AcquisitionFunctions.entropy
        ):
            return acq_func(
                predictions,
                num_of_instances_to_sample
                or self.active_learning_config.num_of_instances_to_sample,
            )
        elif (
            self.active_learning_config.acquisition_function
            is AcquisitionFunctions.hac_entropy
        ):
            return acq_func(
                predictions,
                input_data,
                num_of_instances_to_sample
                or self.active_learning_config.num_of_instances_to_sample,
            )

        raise ValueError("Acquisition function is not specified...")

    def reset_cold_start(self) -> None:
        """
        Parameters random reinitialization.
        """
        for ensemble in self.model:
            ensemble.reset_cold_start()
