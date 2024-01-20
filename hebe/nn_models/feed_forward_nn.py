from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hebe.acquisitions_functions import (
    hac_sampling,
    maximal_entropy_sampling,
    random_indices_sampling,
)
from hebe.config import (
    AcquisitionFunctions,
    ActiveLearningConfig,
    NNParametersConfig,
)
from hebe.nn_models.utils import ModelType

ACQUISITION_FUNCTIONS_MAP: dict[
    AcquisitionFunctions, Callable[..., torch.Tensor]
] = {
    AcquisitionFunctions.random: random_indices_sampling,
    AcquisitionFunctions.entropy: maximal_entropy_sampling,
    AcquisitionFunctions.hac_entropy: hac_sampling,
}


class FeedForwardNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        dropout: float,
    ):
        super(FeedForwardNN, self).__init__()

        self.dropout_ratio = dropout
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.build_the_model()

    def build_the_model(self) -> None:
        self.hidden = nn.Linear(self.input_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=self.dropout_ratio, inplace=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        first_layer = self.sigmoid(self.dropout(self.hidden(input)))
        second_layer = self.sigmoid(self.output(first_layer))
        return second_layer

    def hidden_layer_predict(self, input: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.dropout(self.hidden(input)))


class Classifier:
    def __init__(
        self,
        nn_config: NNParametersConfig,
        active_learning_config: ActiveLearningConfig,
    ):
        self.type = ModelType.FEED_FORWARD_NN
        self.input_size = nn_config.input_size
        self.hidden_size = nn_config.hidden_size
        self.num_classes = 1
        self.dropout_ratio = nn_config.dropout
        self.learning_rate = nn_config.learning_rate
        self.training_epochs = nn_config.training_epochs

        self.active_learning_config = active_learning_config
        self.num_of_instances_to_sample = (
            self.active_learning_config.num_of_instances_to_sample
        )
        self.acquisition_funtion = ACQUISITION_FUNCTIONS_MAP[
            self.active_learning_config.acquisition_function
        ]

        self.model: FeedForwardNN = FeedForwardNN(
            self.input_size,
            self.hidden_size,
            self.num_classes,
            self.dropout_ratio,
        )
        self.criterion: nn.BCELoss = nn.BCELoss()
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.model.parameters(), self.learning_rate
        )

    def train_one_epoch(self, dataloader: DataLoader) -> list[float]:
        losses: list[float] = []
        for batch_id, (x_train, y_train) in enumerate(dataloader):
            y_pred = self.model(x_train)

            loss = self.criterion(y_pred, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.detach().item())

        return losses

    def train(self, dataloader: DataLoader) -> None:
        avg_loss_per_epoch: list[float] = []
        for epoch in range(self.training_epochs):
            losses = self.train_one_epoch(dataloader)
            avg_loss_per_epoch.append(np.mean(losses))

        print(f"First loss: {avg_loss_per_epoch[0]}")
        print(f"Last loss: {avg_loss_per_epoch[-1]}")

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        # Set the model to evaluation mode (important for dropout)
        self.model.eval()

        # Forward pass to get predictions
        with torch.no_grad():
            predictions = self.model(input_data)

        return predictions

    def sample_indices_from_unlabeled_data(
        self,
        input_data: torch.Tensor,
        num_of_instances_to_sample: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Returns sampled instances indices in the original dataset
        """

        if (
            self.active_learning_config.acquisition_function
            is AcquisitionFunctions.random
        ):
            return self.acquisition_funtion(
                input_data,
                num_of_instances_to_sample or self.num_of_instances_to_sample,
            )

        predictions = self.predict(input_data)

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

    def reset_cold_start(self) -> None:
        """
        Parameters random reinitialization.
        """
        del self.model
        del self.optimizer

        self.model: FeedForwardNN = FeedForwardNN(  # type: ignore
            self.input_size,
            self.hidden_size,
            self.num_classes,
            self.dropout_ratio,
        )
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(  # type: ignore
            self.model.parameters(), self.learning_rate
        )
