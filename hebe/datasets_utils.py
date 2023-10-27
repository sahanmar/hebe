from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


def create_moons_data() -> (
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
    X = scale(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.99)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

    return X_train, Y_train, X_test, Y_test


def create_chess_deck_data() -> (
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    # config inits
    gird_size = 4
    cells_seed = list(range(gird_size**2))
    instances_per_cell = 1000 // gird_size**2
    cell_size = 6 / gird_size
    var = (cell_size / 5) ** 2

    # create the dataset
    grid: list[list[np.ndarray]] = []
    labels: list[list[np.ndarray]] = []
    counter = 0

    x_axis, y_axis = -2.2, -2.2
    for i in range(gird_size):
        flag = i % 2
        local_values: list[np.ndarray] = []
        local_labels: list[np.ndarray] = []
        y_axis = -2.2
        for j in range(gird_size):
            np.random.seed(cells_seed[counter])
            local_values.append(
                np.random.multivariate_normal(
                    mean=[x_axis, y_axis],
                    cov=[[var, 0], [0, var]],
                    size=instances_per_cell,
                )
            )
            local_labels.append(
                np.zeros((instances_per_cell, 1))
                if (j + flag) % 2 == 0
                else np.ones((instances_per_cell, 1))
            )
            y_axis += cell_size
            counter += 1
        x_axis += cell_size
        grid.append(local_values)
        labels.append(local_labels)

    X = np.concatenate(
        [np.concatenate(inner_list, axis=0) for inner_list in grid],
        axis=0,
    )

    Y = np.concatenate(
        [np.concatenate(inner_list, axis=0) for inner_list in labels],
        axis=0,
    )

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.99)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

    return X_train, Y_train, X_test, Y_test


def simulate_grid() -> Tuple[np.ndarray, torch.Tensor]:
    grid = np.mgrid[-3:3:100j, -3:3:100j]  # type: ignore
    # Reshape the grid into a list of (x, y) pairs
    points = grid.reshape(2, -1).transpose()

    # Convert the scaled_points to a PyTorch tensor
    return grid, torch.tensor(points, dtype=torch.float32)


def extend_training_data(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    test_to_train_data_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The function extends training data given test_to_train_data_indices.
    x_train and y_train get the labeled data from x_test and y_test based on
    test_to_train_data_indices.
    """
    # Get the indices to transfer from test to train
    train_indices = test_to_train_data_indices.cpu().numpy()

    # Update x_train and y_train in-place without allocating new memory
    x_train = torch.cat((x_train, x_test[train_indices]), dim=0)
    y_train = torch.cat((y_train, y_test[train_indices]), dim=0)

    # Get the complement indices for x_test and y_test
    complement_indices = torch.tensor(
        list(set(range(x_test.size(0))) - set(train_indices))
    )

    # Create new tensors for x_test and y_test without the transferred data
    x_test = x_test[complement_indices]
    y_test = y_test[complement_indices]

    return x_train, y_train, x_test, y_test


def plot_data_uncertainty_grid(
    predictions: torch.Tensor,
    sampled_data: torch.Tensor,
    grid: np.ndarray,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
) -> None:
    # Set up the colormap
    cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot the uncertainty grid
    contour = ax.contourf(
        grid[1], grid[0], predictions.reshape(100, 100).T, cmap=cmap
    )

    # Cat instances and labels
    x = torch.cat((x_train, x_test), dim=0)
    y = torch.cat((y_train, y_test), dim=0)

    # Scatter plot for test data
    class_1 = (y == 0).squeeze(1)
    class_2 = (y == 1).squeeze(1)
    ax.scatter(x[class_1, 0], x[class_1, 1])
    ax.scatter(x[class_2, 0], x[class_2, 1], color="r")
    ax.scatter(
        x_test[sampled_data.cpu().numpy()][:, 0],
        x_test[sampled_data.cpu().numpy()][:, 1],
        marker="*",
        color="k",
        s=100,
    )

    # Add a colorbar
    cbar = plt.colorbar(contour, ax=ax)

    # Set axis limits and labels
    ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="X", ylabel="Y")
    cbar.ax.set_ylabel(
        "Posterior predictive mean probability of class label = 0"
    )

    plt.show()
