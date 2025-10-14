# Base imports
import os
import time
import logging

# Common imports
import numpy as np
from pathlib import Path

# Plotting
import matplotlib

# ML imports
import torch
import joblib # Save pkl

# MLflow
import mlflow
from contextlib import nullcontext

# Ray Tune
import ray
from ray import train, tune, air
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train import Checkpoint
from ray.tune import Tuner, RunConfig, TuneConfig, FailureConfig
from ray.tune.schedulers import ASHAScheduler

# Local imports
from train import transform_data, create_dataloaders, create_lstm_model, train_epoch, validate_epoch
import json

def _mlflow_run_context():
    """Start an MLflow run only if none is active (plays nice with Ray's MLflowCallback)."""
    if mlflow.active_run() is None:
        return mlflow.start_run()
    else:
        return nullcontext()


def train_trial(config, data_ref, base_seed, RAY_HPO=False):
    """Train one model that can be used with our without Ray Tune."""

    #########################################################################
    # Setup MLflow if this is not a Ray orchestrated trial
    #########################################################################
    if not RAY_HPO:
        mlflow.set_tracking_uri("file:./log/mlruns")
        scratch_experiment = mlflow.set_experiment("scratch")

    #########################################################################
    # Set seeds
    #########################################################################
    np.random.seed(base_seed)
    #torch.manual_seed(base_seed)
    g = torch.Generator()         # Creates a generator that fixes the shuffle in torch Dataloader
    g.manual_seed(base_seed)

    #########################################################################
    # Create dataloaders
    #########################################################################
    x_train, y_train, x_val, y_val, scaler = data_ref   # Pull out data from Ray store
    train_loader, val_loader = create_dataloaders(
        x_train,
        y_train,
        x_val,
        y_val,
        int(config["batch_size"]),
        g,
        )
   
    ######################################################################
    # Create model, loss, optimizer
    ######################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # this is the Ray-assigned GPU (Ray sets CUDA_VISIBLE_DEVICES)
    criterion = torch.nn.MSELoss()
    
    # LSTM model
    model = create_lstm_model(
        device=device,
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        )

    # Adapative moment estimation, makes sure we step opposite smoothed gradient and shrink/grow step based on how noisy each model parameter's gradient has been
    optimizer = torch.optim.Adam(model.parameters())

    ######################################################################
    # Training Loop
    ######################################################################
    epochs = int(config["epochs"])
    train_rmse_hist, val_rmse_hist = [], []
    
    # Initiate the MLflow run context
    with _mlflow_run_context():
        for epoch in range(1, epochs + 1):
            
            # Training & Validation
            train_loss = train_epoch(model, device, criterion, optimizer, train_loader)
            val_loss = validate_epoch(model, device, criterion, optimizer, val_loader)
        
            # Create performance metrics
            train_rmse = float(np.sqrt(train_loss))
            val_rmse   = float(np.sqrt(val_loss))
            train_rmse_hist.append(train_rmse)
            val_rmse_hist.append(val_rmse)
        
            # We want to report metrics every epoch regardless if we are checkpointing
            metrics = {
                "val_rmse": float(val_rmse_hist[-1]),
                "train_rmse": float(train_rmse_hist[-1]),
                "epoch": int(epoch),
            }

            ######################################################################
            # Tune logging (with MLflow callaback this is all mirrored there as well)
            # NOTE: By calling tune.report here effectively once per epoch, that becomes our time scale!
            ######################################################################
            if RAY_HPO:
                # Report metrics and save checkpoint if applicable (checkpoint every n epochs and don't have redudant checkpoints if using workers via train)
                checkpoint = None
                should_checkpoint = epoch % config.get("checkpoint_freq", 1) == 0

                # NOTE: In standard DDP training, where the model is the same across all ranks, only the global rank 0 worker needs to save and report the checkpoint
                if should_checkpoint: # add in tune.get_context().get_world_rank() == 0 when workers implemented

                    # Create the checkpoint dir
                    session   = tune.get_context()
                    trial_dir = Path(session.get_trial_dir())
                    ckpt_dir = trial_dir / f"ckpt_e{epoch:04d}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)

                    # Save the model
                    torch.save(model.state_dict(), ckpt_dir / "model.pt")

                    # Save loss plot
                    matplotlib.use("Agg") # Matplotlib runs headless
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    plt.plot(range(1, epoch + 1), train_rmse_hist, label="train_rmse")
                    plt.plot(range(1, epoch + 1), val_rmse_hist,   label="val_rmse")
                    plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.title("Training/Validation RMSE")
                    plt.legend(); plt.tight_layout()
                    plt.savefig(ckpt_dir / "loss_curve.png", dpi=150)
                    plt.close(fig)

                    # Save scaler
                    joblib.dump(scaler, ckpt_dir / "standard_scaler.pkl")

                    # Create checkpoint
                    checkpoint = Checkpoint.from_directory(str(ckpt_dir))

                # Report to Ray (MLflow mirroring is handled by Ray's MLflowCallback)
                tune.report(metrics, checkpoint=checkpoint)
            else:
                # Standalone: log to MLflow each epoch if a run is active
                if mlflow.active_run() is not None:
                    mlflow.log_metrics(metrics, step=epoch)

            # Status for screen
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch:02d} | train RMSE: {train_rmse:.6f} | val RMSE: {val_rmse:.6f}")

        #########################################################################
        # If this is not Ray orchestrated, log mlflow params, register model, and log onnx artifact
        #########################################################################
        if not RAY_HPO and mlflow.active_run() is not None:

            # Param logging
            mlflow.log_params(config)
            
            ####################################################
            # Build an MLflow signature using a small CPU batch
            ####################################################
            model.eval()

            # Choose a small slice for signature + input example
            B = int(config.get("batch_size", 64))
            x_signature_tensor = x_train[:int(config.get("batch_size", 64))]  # torch.Tensor
            x_example_tensor  = x_train[:10]                                  # torch.Tensor

            # Run the model forward with tensors
            with torch.no_grad():
                y_pred_np = model(x_signature_tensor.to(device)).detach().cpu().numpy()

            # Convert inputs to NumPy for MLflow
            x_signature_np = x_signature_tensor.detach().cpu().numpy()        # np.ndarray
            x_example_np = x_example_tensor.detach().cpu().numpy()            # np.ndarray

            # Signature expects numpy (or other) hence why we converted
            signature = mlflow.models.infer_signature(x_signature_np, y_pred_np)

            # Log the trained model (CPU for portability)
            mlflow.pytorch.log_model(
                pytorch_model=model.to("cpu"),
                name="model",
                signature=signature,
                input_example=x_example_np,
            )

            ####################################################
            # ONNX export
            ####################################################
            # Use a batch=1 example for LSTM safety
            example_input_t = x_example_tensor[:1].detach().cpu().float()

            onnx_path = os.path.join(os.getenv("SCRATCH_HOME", "."), "log", "model.onnx")
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

            torch.onnx.export(
                model.to("cpu").eval(),
                example_input_t,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=int(config.get("onnx_opset", 18)),  # guess
                dynamo=True,
            )
            mlflow.log_artifact(onnx_path)
            
def run_HPO(data_ref, seed):
    ######################################################################
    # Start parent HPO, MLflow session
    ######################################################################
    mlflow_tracking_uri = f"file:{os.path.abspath('./log/mlruns')}"  # absolute path
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
    config_path = os.path.join(os.getenv('SCRATCH_HOME'), 'log', 'best_config.json')
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

