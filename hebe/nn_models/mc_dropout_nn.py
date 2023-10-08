import torch

from hebe.config import (
    AcquisitionFunctions,
    ActiveLearningConfig,
    MCDropoutConfig,
    NNParametersConfig,
)
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

    def predict_inner(
        self, input_data: torch.Tensor, use_uncertainty_in_prediction: bool
    ) -> torch.Tensor:
        # Set the model to evaluation mode (important for dropout)
        if not use_uncertainty_in_prediction:
            self.model.eval()
        else:
            self.model.train()

        # Forward pass to get predictions
        with torch.no_grad():
            predictions = self.model(input_data)

        return predictions

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.predict_inner(
            input_data, self.mc_dropout_config.use_uncertainty_in_prediction
        )

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
                    self.predict_inner(input_data, True)
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
