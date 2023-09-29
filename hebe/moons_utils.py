import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple
from sklearn.datasets import make_moons
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split


def create_moons_data() -> (
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
    X = scale(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.95)

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


def extend_moons_training_data(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    test_to_train_data_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The function extends moon training data given test_to_train_data_indices.
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


def plot_moons_uncertainty_grid(
    predictions: torch.Tensor,
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

    # Add a colorbar
    cbar = plt.colorbar(contour, ax=ax)

    # Set axis limits and labels
    ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="X", ylabel="Y")
    cbar.ax.set_ylabel(
        "Posterior predictive mean probability of class label = 0"
    )

    plt.show()
