#!/bin/bash

# Set envs
export SCRATCH_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export MLFLOW_TRACKING_URI="file:/home/sevani/repos/scratch-ml/log/mlruns"
export MLFLOW_REGISTRY_URI="file:/home/sevani/repos/scratch-ml/log/mlruns"   # safe to set too