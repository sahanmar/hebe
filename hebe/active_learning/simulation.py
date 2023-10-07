import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from hebe.config import SimulationConfig, TrainingType, Dateset
from hebe.metrics import calculate_roc_auc
from hebe.datasets_utils import (
    create_moons_data,
    extend_training_data,
    plot_data_uncertainty_grid,
    simulate_grid,
    create_chess_deck_data,
)
from hebe.nn_models import Classifier
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
    auc_history = []

    for loop in range(simulation_config.loops):
        x_train, y_train, x_test, y_test = (
            create_moons_data()
            if simulation_config.data == Dateset.moons
            else create_chess_deck_data()
        )
        grid, nn_input_grid = simulate_grid()
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
            plot_data_uncertainty_grid(
                predictions, grid, x_train, y_train, x_test, y_test
            )
            # perform new data sampling with a training data change
            samped_indices = model.sample_indices_from_unlabeled_data(x_test)
            (x_train, y_train, x_test, y_test) = extend_training_data(
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
