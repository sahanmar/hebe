from typing import Tuple

import pytest
import torch

from hebe.acquisitions_functions import hac_sampling


@pytest.fixture
def instaces_w_predictions() -> Tuple[torch.Tensor, torch.Tensor]:
    "simulating a line from 0 to 20 where class separation is at 10"

    instances = torch.Tensor([[i, 1] for i in range(20)])
    predictions = torch.Tensor(
        [i / 20 for i in range(1, 11)]
        + [0.5 - (i - 10) / 20 for i in range(10, 20)]
    ).unsqueeze(1)
    return instances, predictions


def test_hac_sampling(
    instaces_w_predictions: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    instances, predictions = instaces_w_predictions
    indices = hac_sampling(predictions, instances, 2)
    assert torch.all(indices == torch.Tensor([10, 9]))
