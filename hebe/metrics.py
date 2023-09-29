import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def calculate_roc_auc(
    predictions: np.ndarray | torch.Tensor,
    ground_truth: np.ndarray | torch.Tensor,
) -> float:
    """
    Calculate the Area Under the ROC Curve (ROC AUC) for binary classification.

    Args:
        predictions (torch.Tensor): Predicted probabilities or scores for the
        positive class.
        ground_truth (torch.Tensor): True binary labels (0 or 1).

    Returns:
        float: The ROC AUC score.
    """
    # Ensure predictions and ground_truth are NumPy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(ground_truth, predictions)

    return roc_auc
