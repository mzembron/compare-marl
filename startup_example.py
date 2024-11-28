from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

from vmas import render_interactively

import torch

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your CUDA installation.")
        exit(1)
    else:
        print("CUDA is available.")

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = "cuda"
    experiment_config.train_device = "cuda"
    experiment_config.buffer_device = "cuda"

    # Loads from "benchmarl/conf/task/vmas/balance.yaml"
    task = VmasTask.BALANCE.get_from_yaml()
    task.render_mode = "human"  # Ensure the task is set to render in human mode

    # Loads from "benchmarl/conf/algorithm/mappo.yaml"
    algorithm_config = MappoConfig.get_from_yaml()

    # Loads from "benchmarl/conf/model/layers/mlp.yaml"
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )
    experiment.run()

    # Render the task interactively
    render_interactively(task)