import numpy as np
import torch
from torch.utils.data import DataLoader

from hebe.config import ActiveLearningConfig, NNParametersConfig, VadamConfig
from hebe.nn_models.feed_forward_nn import Classifier, FeedForwardNN
from hebe.nn_models.vadam.vadam_optimizer import Vadam
from hebe.nn_models.utils import ModelType


class VadamClassifier(Classifier):
    def __init__(
        self,
        nn_config: NNParametersConfig,
        active_learning_config: ActiveLearningConfig,
        vadam_config: VadamConfig,
    ):
        super().__init__(nn_config, active_learning_config)

        self.type = ModelType.VADAM
        self.vadam_config: VadamConfig = vadam_config
        self.switch_optimizer: bool = False
        self.training_set_size: int | None = None

        self.nn_config = nn_config
        self.optimizer_1: torch.optim.Adam = torch.optim.Adam(
            self.model.parameters(), self.learning_rate
        )
        self.optimizer_2: Vadam = Vadam(
            self.model.parameters(),
            self.vadam_config.training_set_size,
            self.learning_rate,
            betas=self.vadam_config.betas,
            std=self.vadam_config.std,
        )

        del self.optimizer

    def train(self, dataloader: DataLoader) -> None:
        avg_loss_per_epoch: list[float] = []
        for epoch in range(self.training_epochs):
            if (
                1 - (epoch / self.training_epochs)
                <= self.vadam_config.optimizer_switch_ratio
            ):
                self.switch_optimizer = True
            else:
                self.switch_optimizer = False

            losses = self.train_one_epoch(dataloader)
            avg_loss_per_epoch.append(np.mean(losses))

        print(f"First loss: {avg_loss_per_epoch[0]}")
        print(f"Last loss: {avg_loss_per_epoch[-1]}")

    def train_one_epoch(self, dataloader: DataLoader) -> list[float]:
        losses: list[float] = []

        for batch_id, (x_train, y_train) in enumerate(dataloader):
            self.training_set_size = x_train.shape[0]

            def closure():
                self.optimizer_1.zero_grad()
                self.optimizer_2.zero_grad()
                y_pred = self.model(x_train)
                loss = self.criterion(y_pred, y_train)
                loss.backward()
                return loss

            loss = (
                self.optimizer_1.step(closure)
                if not self.switch_optimizer
                else self.optimizer_2.step(closure)
            )

            if loss:
                losses.append(loss.detach().item())

        return losses

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        # Forward pass to get predictions
        with torch.no_grad():
            if self.switch_optimizer:
                multiple_predictions = self.optimizer_2.get_mc_predictions(
                    self.model, input_data, self.vadam_config.number_of_samples
                )

                stacked_predictions = torch.stack(multiple_predictions, dim=0)

                predictions = torch.mean(stacked_predictions, dim=0)
            else:
                predictions = self.model(predictions)

        return predictions

    def reset_cold_start(self) -> None:
        """
        Parameters random reinitialization.
        """
        del self.model
        del self.optimizer_1
        del self.optimizer_2

        self.model: FeedForwardNN = FeedForwardNN(  # type: ignore
            self.input_size,
            self.hidden_size,
            self.num_classes,
            self.dropout_ratio,
        )
        self.optimizer_1: torch.optim.Adam = torch.optim.Adam(  # type: ignore
            self.model.parameters(), self.learning_rate
        )
        self.optimizer_2: Vadam = Vadam(  # type: ignore
            self.model.parameters(),
            self.training_set_size
            if self.training_set_size
            else self.vadam_config.training_set_size,
            self.learning_rate,
            betas=self.vadam_config.betas,
            std=self.vadam_config.std,
        )
