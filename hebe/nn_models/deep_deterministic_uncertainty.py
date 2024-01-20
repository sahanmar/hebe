import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

from hebe.config import (
    AcquisitionFunctions,
    ActiveLearningConfig,
    NNParametersConfig,
)
from hebe.nn_models.feed_forward_nn import Classifier
from hebe.nn_models.utils import ModelType


class DeepDeterministicUncertainty(Classifier):
    def __init__(
        self,
        nn_config: NNParametersConfig,
        active_learning_config: ActiveLearningConfig,
    ):
        super().__init__(nn_config, active_learning_config)

        self.type = ModelType.DEEP_DETERMINISTIC_UNCERTAINTY
        self.gmm_0: GaussianMixture | None = None
        self.gmm_1: GaussianMixture | None = None

    def train(self, dataloader: DataLoader) -> None:
        avg_loss_per_epoch: list[float] = []
        for epoch in range(self.training_epochs):
            losses = self.train_one_epoch(dataloader)
            avg_loss_per_epoch.append(np.mean(losses))

        self.model.eval()
        with torch.no_grad():
            self.gmm(
                torch.cat([x for x, _ in dataloader], dim=0),
                torch.cat([y for _, y in dataloader], dim=0),
            )

        print(f"First loss: {avg_loss_per_epoch[0]}")
        print(f"Last loss: {avg_loss_per_epoch[-1]}")

    def gmm(self, instances: torch.Tensor, labels: torch.Tensor) -> None:
        zero_class = instances[labels.squeeze() == 0]
        first_class = instances[labels.squeeze() == 1]

        self.gmm_0 = GaussianMixture(1).fit(
            self.model.hidden_layer_predict(zero_class)
        )
        self.gmm_1 = GaussianMixture(1).fit(
            self.model.hidden_layer_predict(first_class)
        )

    def sample_indices_from_unlabeled_data(
        self,
        input_data: torch.Tensor,
        num_of_instances_to_sample: int | None = None,
    ) -> torch.Tensor:
        """
        Returns sampled instances indices in the original dataset
        """

        def get_gmm_predictions(input_data: torch.Tensor) -> torch.Tensor:
            self.model.eval()
            with torch.no_grad():
                if not self.gmm_0 or not self.gmm_1:
                    raise ValueError("Please train the model first...")
                gmm_0_predictions = torch.tensor(
                    self.gmm_0.predict(
                        self.model.hidden_layer_predict(input_data)
                    )
                ).unsqueeze(1)
                gmm_1_predictions = torch.tensor(
                    self.gmm_1.predict(
                        self.model.hidden_layer_predict(input_data)
                    )
                ).unsqueeze(1)
                normalization = gmm_0_predictions + gmm_1_predictions

                return torch.cat(
                    [
                        gmm_0_predictions / normalization,
                        gmm_1_predictions / normalization,
                    ],
                    dim=1,
                )

        if (
            self.active_learning_config.acquisition_function
            is AcquisitionFunctions.random
        ):
            return self.acquisition_funtion(
                input_data,
                num_of_instances_to_sample or self.num_of_instances_to_sample,
            )

        predictions = self.predict(get_gmm_predictions(input_data))

        if (
            self.active_learning_config.acquisition_function
            is AcquisitionFunctions.entropy
        ):
            return self.acquisition_funtion(
                predictions,
                num_of_instances_to_sample or self.num_of_instances_to_sample,
            )
        elif (
            self.active_learning_config.acquisition_function
            is AcquisitionFunctions.hac_entropy
        ):
            return self.acquisition_funtion(
                predictions,
                input_data,
                num_of_instances_to_sample or self.num_of_instances_to_sample,
            )

        raise ValueError("Acquisition function is not specified...")
