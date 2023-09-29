import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from hebe.config import SimulationConfig
from hebe.nn_models import Classifier
from torch.utils.data import TensorDataset, DataLoader
from hebe.moons_utils import (
    extend_moons_training_data,
    plot_moons_uncertainty_grid,
    create_moons_data,
    simulate_grid,
)
from hebe.metrics import calculate_roc_auc
from datetime import datetime
from hebe.config import TrainingType
from paths import DATA


def log_simulation_data(
    simulation_config: SimulationConfig, auc_history: list[list[float]]
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = DATA / f"simulation_data_{timestamp}.json"
    DATA.mkdir(parents=True, exist_ok=True)

    data = {
        "simulation_config": {
            "loops": simulation_config.loops,
            "iterations": simulation_config.iterations,
            "training_type": simulation_config.training_type.value,
            # Add other simulation config data here
        },
        "auc_history": auc_history,
    }

    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def plot_auc(auc_history: list[list[float]]) -> None:
    plt.plot(np.mean(auc_history, axis=0), marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC Evolution")
    plt.grid(True)
    plt.show()


def create_train_dataloader(
    x_train: torch.Tensor, y_train: torch.Tensor, batch_size: int | None = None
) -> DataLoader:
    """
    x_train - training instances
    y_train - training labels
    """
    if not batch_size:
        batch_size = len(x_train)
    train_dataset = TensorDataset(x_train, y_train)
    return DataLoader(train_dataset, batch_size=batch_size)


def active_learning_moon_simulation(
    simulation_config: SimulationConfig,
    model: Classifier,
) -> None:
    x_train, y_train, x_test, y_test = create_moons_data()
    grid, nn_input_grid = simulate_grid()
    auc_history = []

    for loop in range(simulation_config.loops):
        loop_auc_values = []
        for iter in range(simulation_config.iterations):
            # create training data structure
            train_dataloader = create_train_dataloader(x_train, y_train)
            # train and predict on visualisation data
            model.train(train_dataloader)
            predictions = model.predict(nn_input_grid)
            test_predictions = model.predict(x_test)
            auc = calculate_roc_auc(test_predictions, y_test)
            loop_auc_values.append(auc)
            # visualise
            plot_moons_uncertainty_grid(
                predictions, grid, x_train, y_train, x_test, y_test
            )
            # perform new data sampling with a training data change
            samped_indices = model.sample_indices_from_unlabeled_data(x_test)
            (x_train, y_train, x_test, y_test) = extend_moons_training_data(
                x_train,
                y_train,
                x_test,
                y_test,
                samped_indices,
            )
            # Reinitialize the model
            if simulation_config.training_type is TrainingType.cold_start:
                model.reset_cold_start()

            print(
                f"Iteration {iter} if finished with AUC = {round(auc,3)} ...",
                "\n",
            )

        auc_history.append(loop_auc_values)
        model.reset_cold_start()

    plot_auc(auc_history)
    log_simulation_data(simulation_config, auc_history)
