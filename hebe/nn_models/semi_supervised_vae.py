import torch
import torch.nn as nn
import numpy as np

from torch.distributions import Normal
from hebe.nn_models.feed_forward_nn import FeedForwardNN
from hebe.config import (
    ActiveLearningConfig,
    NNParametersConfig,
    VAEConfig,
    AcquisitionFunctions,
)
from torch.utils.data import DataLoader
from typing import Tuple
from itertools import chain
from hebe.nn_models.utils import ModelType
from hebe.nn_models.feed_forward_nn import ACQUISITION_FUNCTIONS_MAP

# Semi-supervised Encoder


class Encoder(nn.Module):
    def __init__(self, vae_config: VAEConfig) -> None:
        """
        Encoder for semi-supervised VAE
        """
        super(Encoder, self).__init__()

        self.mu = nn.Linear(
            vae_config.encoder_in_features, vae_config.encoder_out_features
        )
        self.rho = nn.Linear(
            vae_config.encoder_in_features, vae_config.encoder_out_features
        )
        self.softplus = nn.Softplus()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xy = torch.cat((x, y), dim=-1)
        mu = self.mu(xy)
        sigma = self.softplus(self.rho(xy)) + 1e-10  # improves stability
        eps = torch.randn_like(sigma)
        return mu + sigma * eps, mu, sigma


# Semi-supervised Decoder


class Decoder(nn.Module):
    def __init__(self, vae_config: VAEConfig) -> None:
        """
        Decoder for semi-supervised VAE
        """
        super(Decoder, self).__init__()
        self.mu = nn.Linear(
            in_features=vae_config.decoder_in_features,
            out_features=vae_config.decoder_out_features,
        )
        self.rho = nn.Linear(
            in_features=vae_config.decoder_in_features,
            out_features=vae_config.decoder_out_features,
        )
        self.softplus = nn.Softplus()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xy = torch.cat((x, y), dim=-1)
        mu = self.mu(xy)
        sigma = self.softplus(self.rho(xy)) + 1e-4  # improves stability
        return mu, sigma


# Semi-supervised Variational Autoencoder


class Model:
    def __init__(
        self,
        vae_config: VAEConfig,
        nn_config: NNParametersConfig,
    ):
        self.vae_config = vae_config
        self.nn_config = nn_config

        # VAE Econder and Decoder
        self.encoder: nn.Module = Encoder(self.vae_config)
        self.decoder: nn.Module = Decoder(self.vae_config)

        # Classifier
        self.classifier: FeedForwardNN = FeedForwardNN(
            self.nn_config.input_size,
            self.nn_config.hidden_size,
            1,
            self.nn_config.dropout,
        )


class SSVAE:
    def __init__(
        self,
        nn_config: NNParametersConfig,
        active_learning_config: ActiveLearningConfig,
        vae_config: VAEConfig,
    ):
        self.type = ModelType.SSVAE
        self.vae_config = vae_config
        self.nn_config = nn_config
        self.learning_rate = nn_config.learning_rate
        self.training_epochs = nn_config.training_epochs

        self.model = Model(self.vae_config, self.nn_config)

        # active learning
        self.active_learning_config = active_learning_config
        self.num_of_instances_to_sample = (
            self.active_learning_config.num_of_instances_to_sample
        )
        self.acquisition_funtion = ACQUISITION_FUNCTIONS_MAP[
            self.active_learning_config.acquisition_function
        ]

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            chain(
                self.model.encoder.parameters(),
                self.model.decoder.parameters(),
                self.model.classifier.parameters(),
            ),
            self.learning_rate,
        )
        self.clf_criterion: nn.BCELoss = nn.BCELoss()
        self.num_labels = self.nn_config.num_labels

    def elbo_xy(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        # log uniform prior
        logpy = torch.log(1 / self.num_labels * torch.ones(x.size(dim=0)))
        # kld
        z_l, mu_z, sigma_z = self.model.encoder(x, y)
        kld_z = -0.5 * torch.mean(
            torch.sum(
                input=(
                    1 + torch.log(sigma_z.pow(2)) - mu_z.pow(2) - sigma_z.pow(2)
                ),
                dim=1,
            )
            * torch.ones(x.size(dim=0))
        )
        # likelihood
        mu_x, sigma_x = self.model.decoder(z_l, y)
        logpx = torch.sum(Normal(mu_x, scale=sigma_x).log_prob(x), dim=-1)

        return logpx - kld_z + logpy

    def elbo_x(self, x: torch.Tensor) -> torch.Tensor:
        y_pred = self.model.classifier(x)
        y_pred = torch.cat([1 - y_pred, y_pred], dim=1)
        likelihoods = []
        for i in range(self.num_labels):
            y = i * torch.ones(x.size(dim=0), 1).to(torch.int64)
            likelihoods.append(self.elbo_xy(x, y))

        likelihood = torch.stack(likelihoods, dim=1)
        likelihood = torch.sum(
            y_pred * likelihood - y_pred * torch.log(y_pred), dim=1
        )
        return torch.mean(likelihood, dim=0)

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        # Set the model to evaluation mode (important for dropout)
        self.model.classifier.eval()

        # Forward pass to get predictions
        with torch.no_grad():
            predictions = self.model.classifier(input_data)

        return predictions

    def criterion(
        self, data: torch.Tensor | tuple[torch.Tensor, torch.Tensor], alpha=1
    ) -> torch.Tensor:
        if len(data) > 1:
            x, y = data
            l_xy = self.elbo_xy(x, y)
            clf_loss = self.clf_criterion(self.model.classifier(x), y)
            loss = -torch.mean(l_xy) + alpha * clf_loss
        else:
            x = data[0]
            loss = -self.elbo_x(x)
        return loss

    def train_one_epoch(
        self, labaled_dataloader: DataLoader, unlabaled_dataloader: DataLoader
    ) -> list[float]:
        losses: list[float] = []
        for data in chain(labaled_dataloader, unlabaled_dataloader):
            loss = self.criterion(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.detach().item())

        return losses

    def train(
        self, labaled_dataloader: DataLoader, unlabaled_dataloader: DataLoader
    ) -> None:
        avg_loss_per_epoch: list[float] = []
        for _ in range(self.training_epochs):
            losses = self.train_one_epoch(
                labaled_dataloader, unlabaled_dataloader
            )
            avg_loss_per_epoch.append(np.mean(losses))

        print(f"First loss: {avg_loss_per_epoch[0]}")
        print(f"Last loss: {avg_loss_per_epoch[-1]}")

    def sample_indices_from_unlabeled_data(
        self,
        input_data: torch.Tensor,
        num_of_instances_to_sample: int | None = None,
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

        self.model = Model(self.vae_config, self.nn_config)  # type: ignore

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(  # type: ignore
            chain(
                self.model.encoder.parameters(),
                self.model.decoder.parameters(),
                self.model.classifier.parameters(),
            ),
            self.learning_rate,
        )
