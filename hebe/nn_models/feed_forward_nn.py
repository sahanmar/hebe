import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import Callable, Optional
from hebe.config.config import (
    NNParametersConfig,
    ActiveLearningConfig,
    AcquisitionFunctions,
)


def random_indices_sampling(
    data: torch.Tensor, num_of_instances_to_sample: int, seed=None
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)

    if num_of_instances_to_sample > data.size(0):
        raise ValueError(
            "Number of instances to sample (n) cannot be greater than the size of the input tensor (K)."
        )

    indices = torch.randperm(data.size(0))[:num_of_instances_to_sample]

    return indices


ACQUISITION_FUNCTIONS_MAP = {AcquisitionFunctions.random: random_indices_sampling}


class FeedForwardNN(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_classes: int, dropout: int
    ):
        super(FeedForwardNN, self).__init__()

        self.dropout_ratio = dropout
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.build_the_model()

    def build_the_model(self) -> "FeedForwardNN":
        self.hidden = nn.Linear(self.input_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=self.dropout_ratio, inplace=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        first_layer = self.sigmoid(self.dropout(self.hidden(input)))
        second_layer = self.sigmoid(self.output(first_layer))
        return second_layer


class Classifier:
    def __init__(
        self,
        # TODO Generalize it into config
        nn_config: NNParametersConfig,
        active_learning_config: ActiveLearningConfig
        # acquisition_funtion: Callable[
        #     [torch.Tensor], torch.Tensor
        # ] = random_indices_sampling,
    ):
        self.input_size = nn_config.input_size
        self.hidden_size = nn_config.hidden_size
        self.num_classes = 1
        self.dropout_ratio = nn_config.dropout
        self.learning_rate = nn_config.learning_rate
        self.training_epochs = nn_config.training_epochs
        self.num_of_instances_to_sample = (
            active_learning_config.num_of_instances_to_sample
        )
        self.acquisition_funtion: Callable[
            [torch.Tensor], torch.Tensor
        ] = ACQUISITION_FUNCTIONS_MAP[active_learning_config.acquisition_function]

        self.model: FeedForwardNN = FeedForwardNN(
            self.input_size,
            self.hidden_size,
            self.num_classes,
            self.dropout_ratio,
        )
        self.criterion: nn.BCELoss() = nn.BCELoss()
        self.optimizer: torch.optim.Adam = torch.optim.Adam(
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
        self, input_data: torch.Tensor, num_of_instances_to_sample: Optional[int] = None
    ) -> torch.Tensor:
        """
        Returns sampled instances indices in the original dataset
        """
        if num_of_instances_to_sample is None:
            return self.acquisition_funtion(input_data, self.num_of_instances_to_sample)
        return self.acquisition_funtion(input_data, num_of_instances_to_sample)

    def reset_cold_start(self) -> None:
        """
        Parameters random reinitialization.
        """
        self.model: FeedForwardNN = FeedForwardNN(
            self.input_size,
            self.hidden_size,
            self.num_classes,
            self.dropout_ratio,
        )
        self.optimizer: torch.optim.Adam = torch.optim.Adam(
            self.model.parameters(), self.learning_rate
        )
