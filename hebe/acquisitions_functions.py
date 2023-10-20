import torch


def data_sanity_check(
    num_of_instances_to_sample: int, data: torch.Tensor
) -> None:
    if num_of_instances_to_sample > data.size(0):
        raise ValueError(
            "Number of instances to sample (n) cannot be greater"
            "than the size of the input tensor (K)."
        )


def random_indices_sampling(
    data: torch.Tensor,
    num_of_instances_to_sample: int,
    seed: int | None = None,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)

    data_sanity_check(num_of_instances_to_sample, data)

    indices = torch.randperm(data.size(0))[:num_of_instances_to_sample]

    return indices


def maximal_entropy_sampling(
    data: torch.Tensor,
    num_of_instances_to_sample: int,
    seed: int | None = None,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)

    data_sanity_check(num_of_instances_to_sample, data)

    if data.size()[1] == 1:
        probabilities = torch.cat((data, 1 - data), dim=1)
    else:
        probabilities = data

    # Calculate entropy for each data point
    entropy = -torch.sum(probabilities * torch.log(probabilities), dim=1)

    # Sort data points by descending entropy
    sorted_indices = torch.argsort(entropy, descending=True)

    # Select the top `num_of_instances_to_sample` indices
    selected_indices = sorted_indices[:num_of_instances_to_sample]

    return selected_indices
