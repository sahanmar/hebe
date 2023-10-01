import torch

from hebe.config import (AcquisitionFunctions, ActiveLearningConfig,
                         MCDropoutConfig, NNParametersConfig)
from hebe.nn_models.feed_forward_nn import Classifier


class MCDropoutClassifier(Classifier):
    def __init__(
        self,
        nn_config: NNParametersConfig,
        active_learning_config: ActiveLearningConfig,
        mc_dropout_config: MCDropoutConfig,
    ):
        super().__init__(nn_config, active_learning_config)

        self.mc_dropout_config = mc_dropout_config

    def predict(
        self, input_data: torch.Tensor, data_sampling: bool = False
    ) -> torch.Tensor:
        # Set the model to evaluation mode (important for dropout)
        if not data_sampling:
            self.model.eval()
        else:
            self.model.train()

        # Forward pass to get predictions
        with torch.no_grad():
            predictions = self.model(input_data)

        return predictions

    def sample_indices_from_unlabeled_data(
        self,
        input_data: torch.Tensor,
        num_of_instances_to_sample: int | None = None,
    ) -> torch.Tensor:
        """
        Returns sampled instances indices in the original dataset
        """

        if self.acquisition_funtion is AcquisitionFunctions.random:
            predictions = input_data
        else:
            predictions = torch.stack(
                [
                    self.predict(input_data, data_sampling=True)
                    for i in range(self.mc_dropout_config.number_of_samples)
                ],
                dim=1,
            )
            mean_predictions = torch.mean(predictions, dim=1)
            predictions = mean_predictions / mean_predictions.sum(
                dim=1, keepdim=True
            )

        return self.acquisition_funtion(
            predictions,
            num_of_instances_to_sample or self.num_of_instances_to_sample,
        )
