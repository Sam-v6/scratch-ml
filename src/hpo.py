# Base imports
import os
import time
import logging

# ML imports
import json

# MLflow
import mlflow

# Ray Tune
import ray
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune import Tuner, RunConfig, TuneConfig, FailureConfig
from ray.tune.schedulers import ASHAScheduler

# Local imports
from train import transform_data, train_trial
from paths import SCRATCH_HOME

def run_HPO(data_ref, seed):
    ######################################################################
    # Start parent HPO, MLflow session
    ######################################################################
    log_dir = SCRATCH_HOME / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    mlflow_db_path = log_dir / "mlflow.db"
    mlflow_tracking_uri = f"sqlite:///{mlflow_db_path}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment = "scratch"

    ######################################################################
    # Define search space and scheduler
    ######################################################################
    lstm_params = {
        # Model shape
        "hidden_size": tune.choice([32, 64, 128, 256]),
        "num_layers": tune.choice([1, 2]),
        "dropout": tune.choice([0.0, 0.1, 0.2]) ,

        # Training
        "batch_size": tune.choice([32, 64, 128]),

        # Epochs / checkpointing
        "epochs": 50,
        "checkpoint_freq": 5,
    }

    params=lstm_params

    # Async Successive Halfing Scheduler (ASHA)
    # Instead of running all trials for all epochs, it allocates more resources to promising ones and kills of bad ones early, these trials are pruned after grace period according to our reduction factor
    scheduler = ASHAScheduler(
        max_t=params["epochs"],                     # Max amount of "things" on our whatever our scale is (since we call tune.report once per epoch this max epochs per trial)
        grace_period=params["checkpoint_freq"]+1,   # Allow for x epochs each trial until we kill it
        reduction_factor=2,                         # ASHA keeps about 50% of the top trials each time it prunes
    )
    
    ######################################################################
    # Build tuner; pass MLflow context and PARENT RUN ID to workers via env vars
    ######################################################################
    # Allows each training run to get training data from shared object store and random seed
    trainable = tune.with_parameters(
        train_trial,            # The function that we want
        data_ref=data_ref,      # The data we are passing in
        base_seed=base_seed,    # Random seed we want everything to go from
        RAY_HPO=True            # Indicating this is a HPO run where MLflow logging will work differently
        )
    
    tuner = Tuner(
        tune.with_resources(trainable, resources={"cpu": 4, "gpu": 1}),  # Gives 4 CPU and one GPU per trial, GPU will bottlenecking trials here at 8, fractional GPU is possible for inference or if we manage mem explicitly which sounds like a nightmare
        param_space=params,
        tune_config=TuneConfig( 
            metric="val_rmse",
            mode="min",                 # Minimize RMSE
            scheduler=scheduler,
            num_samples=25,            # total trials
        ),
        run_config=RunConfig(
            name="lstm_hpo",
            storage_path=os.path.abspath("./log/ray_results"),
            failure_config=FailureConfig(fail_fast=True),
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=mlflow_tracking_uri,
                    experiment_name=experiment,
                    save_artifact=True,
                )
            ],
        ),
    )

    ######################################################################
    # Execute HPO
    ######################################################################
    results = tuner.fit()
    best = results.get_best_result(metric="val_rmse", mode="min")
    print("Best config:", best.config)

    return best

if __name__ == "__main__":

    # TODO: https://docs.ray.io/en/latest/train/user-guides/hyperparameter-optimization.html#train-tune
    # Right now I'm just using Tune and doing a sweep such that each trial get 4 CPU and 1 GPU to train on
    # We could do Tune --> Train --> Workers such that one trial then gets picked up Train such that we can use multiple workers to train for that (multuple GPUs, etc)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Set seed the get data and store it common share
    ray.init()
    base_seed = int(42)
    x_train, y_train, x_val, y_val, scaler = transform_data(base_seed)
    data_ref = ray.put((x_train, y_train, x_val, y_val, scaler))              # Puts data into Ray's object store which each trial can access

    # Run hyperparameter optimization
    best = run_HPO(
        data_ref,
        base_seed,
        )

    # Save best configuration to JSON
    config_path = SCRATCH_HOME / "log" / "best_config.json"
    with open(config_path, 'w') as f:
        json.dump(best.config, f, indent=4)

    # Re train a model with the best config
    with open(config_path, 'r') as f:
        best_config = json.load(f)

    train_trial(
        config=best_config,
        data_ref=(x_train, y_train, x_val, y_val, scaler),
        base_seed=base_seed,
        RAY_HPO=False,
    )

