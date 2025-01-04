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

    # If installed with conda, the benchmarl sources are under:
    # /home/shared/miniconda3/envs/<env-name>/lib/python<version>/site-packages/benchmarl
    # eg. /home/shared/miniconda3/envs/marl-py-3-11/lib/python3.11/site-packages/benchmarl

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = "cuda"
    experiment_config.train_device = "cuda"
    experiment_config.buffer_device = "cuda"
    experiment_config.checkpoint_at_end = True
    experiment_config.checkpoint_interval = 50 * experiment_config.off_policy_collected_frames_per_batch
    # experiment_config.render = True

    # Loads from "benchmarl/conf/task/vmas/balance.yaml"
    task = VmasTask.BALANCE.get_from_yaml()
    task.render_mode = "human"  # Ensure the task is set to render in human mode
    # task.max_steps(300)

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
    # render_interactively(task)