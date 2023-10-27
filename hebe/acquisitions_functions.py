import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering


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


def _get_cluster_sizes(key: int, cluster_tree: dict[int, np.ndarray]) -> int:
    cluster_size = 0
    for leaf in cluster_tree[key]:
        if leaf in cluster_tree:
            cluster_size += _get_cluster_sizes(leaf, cluster_tree)
        else:
            cluster_size += 1

    return cluster_size


def hac_sampling(
    predictions: torch.Tensor,
    instances: torch.Tensor,
    num_of_instances_to_sample: int,
    seed: int | None = None,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)

    data_sanity_check(num_of_instances_to_sample, predictions)

    high_entropy_values = maximal_entropy_sampling(
        predictions, 5 * num_of_instances_to_sample
    ).tolist()

    model = AgglomerativeClustering(linkage="average").fit(instances.cpu())
    cluster_tree = dict(enumerate(model.children_, model.n_leaves_))

    cluster_criterion = []
    high_entropy_clusters = {}  # type: ignore
    for high_entropy_val_idx in high_entropy_values:
        for cluster, components in cluster_tree.items():
            if (
                high_entropy_val_idx in components
                and _get_cluster_sizes(cluster, cluster_tree) > 1
            ):
                high_entropy_clusters.setdefault(cluster, []).append(
                    high_entropy_val_idx
                )
                if cluster not in cluster_criterion:
                    cluster_criterion.append(cluster)

    ids_2_return: list[int] = []
    for cluster_id in cluster_criterion:
        if len(ids_2_return) == num_of_instances_to_sample:
            break
        if (
            cluster_id in high_entropy_clusters
            and high_entropy_clusters[cluster_id]
        ):
            index_to_add = high_entropy_clusters[cluster_id].pop(0)
            ids_2_return.append(index_to_add)

    if len(ids_2_return) > num_of_instances_to_sample:
        print(
            f"Smth went wrong, only {len(ids_2_return)} values sampled instead of {num_of_instances_to_sample}"
        )

    return torch.Tensor(ids_2_return)
