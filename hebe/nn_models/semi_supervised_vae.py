import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Normal
from hebe.nn_models.feed_forward_nn import FeedForwardNN
from hebe.config import ActiveLearningConfig, NNParametersConfig, VAEConfig


def kl_div(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Function that computes variation autoencoder KL divergence

    mu and sigma are the same size
    """

    return -0.5 * torch.sum(
        1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)
    )


# Semi-supervised Encoder


class Encoder(nn.Module):
    def __init__(self, vae_config: VAEConfig):
        """
        Encoder for semi-supervised VAE
        """
        super(Encoder, self).__init__()

        self.mu = nn.Linear(vae_config.in_features, vae_config.out_features)
        self.rho = nn.Linear(vae_config.in_features, vae_config.out_features)
        self.softplus = nn.Softplus()

    def forward(self, x, y):
        xy = torch.cat((x, y), dim=-1)
        mu = self.mu(xy)
        sigma = self.softplus(self.rho(xy)) + 1e-10  # improves stability
        eps = torch.randn_like(sigma)
        return mu + sigma * eps, mu, sigma


# Semi-supervised Decoder


class Decoder(nn.Module):
    def __init__(self, vae_config: VAEConfig):
        """
        Decoder for semi-supervised VAE
        """
        super(Decoder, self).__init__()
        self.mu = nn.Linear(
            in_features=vae_config.in_features, out_features=vae_config.mu_out
        )
        self.rho = nn.Linear(
            in_features=vae_config.in_features,
            out_features=vae_config.sigma_out,
        )
        self.softplus = nn.Softplus()

    def forward(self, x, y):
        xy = torch.cat((x, y), dim=-1)
        mu = self.mu(xy)
        sigma = self.softplus(self.rho(xy)) + 1e-10  # improves stability
        return mu, sigma


# Semi-supervised Variational Autoencoder


class SSVAE(nn.Module):
    def __init__(
        self,
        nn_config: NNParametersConfig,
        active_learning_config: ActiveLearningConfig,
        vae_config: VAEConfig,
    ):
        super(SSVAE, self).__init__()

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

        # active learning setup
        self.active_learning_config = active_learning_config
        self.num_of_instances_to_sample = (
            self.active_learning_config.num_of_instances_to_sample
        )

        # TODO add this to config
        self.num_labels = 2
        self.batch_size = 1
        self.dataset_size = 1000

    def likelihood(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mu_x: torch.Tensor,
        sigma_x: torch.Tensor,
        mu_z: torch.Tensor,
        sigma_z: torch.Tensor,
    ) -> torch.Tensor:
        # uniform prior
        prior_y = 1 / 2 * torch.ones(x.size(dim=0))
        logpy = F.nll_loss(prior_y, y)

        kld_z = -0.5 * torch.mean(
            torch.sum(
                1 + torch.log(sigma_z.pow(2)) - mu_z.pow(2) - sigma_z.pow(2),
                axis=1,
            )
        )

        logpx = Normal(mu_x, scale=sigma_x).log_prob(x)
        likelihood = logpx + logpy - kld_z
        return likelihood

    # def prior_likelihood(self):
    #     likelihood = 0
    #     vars = self.trainable_vars()
    #     for var in vars:
    #         likelihood += tf.reduce_sum(logpdf.std_gaussian(var))
    #     return likelihood

    def forward(self, x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        alpha = 0.1 * self.batch_size

        # Classification
        y_pred = self.classifier(x)

        if y:
            self.loss_clf = torch.sum(
                F.nll_loss(y_pred, labels=self.y, reduce=False), dim=0
            )  # <- must be a scalar!

            ##### Labeled Data Encoder Decoder #####

            z_l, mu_z_l, sigma_z_l = self.encoder(self.x, self.y)
            mu_x_l, sigma_x_l = self.decoder(z_l, y)

            # loss of labelled data, refered as L(x, y)
            likelihood_l = self.likelihood(
                x, y, mu_x_l, sigma_x_l, mu_z_l, sigma_z_l
            )

            self.loss_l = -torch.sum(likelihood_l, dim=0)

            self.loss = (self.loss_l + alpha * self.loss_clf) / self.batch_size

        else:
            ##### Unlabled Data Encoder Decoder #####

            likelihood_u_list = []
            for i in range(self.num_labels):
                y_us = i * torch.ones(x.size(dim=0))
                y_us = F.one_hot(y_us, num_classes=self.num_labels)

                z_u, mu_z_u, sigma_z_u = self.encoder(self.x, y_us)
                mu_x_u, sigma_x_u = self.decoder(z_u, y_us, reuse=True)

                _likelihood_u = self.likelihood(
                    x, y_us, mu_x_u, sigma_x_u, mu_z_u, sigma_z_u
                )
                likelihood_u_list.append(_likelihood_u)

            likelihood_u = torch.stack(likelihood_u_list, dim=1)

            # add the H(q(y|x))
            likelihood_u = torch.sum(
                y_pred * likelihood_u - y_pred * torch.log(y_pred), dim=1
            )

            self.loss_u = -torch.sum(likelihood_u, dim=0)

            self.loss = self.loss_u / self.batch_size

        prior_weight = 1.0 / (self.dataset_size)
        # self.loss_prior = -self.prior_likelihood() <- Finished here
        # self.loss += prior_weight * self.loss_prior

        return torch.argmax(y_pred, 1)
