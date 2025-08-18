#!/bin/bash
# Initialize conda
source /opt/conda/etc/profile.d/conda.sh

# Activate your environment
conda activate myenv

# Build a timestamp, e.g. 20250622_103045
timestamp=$(date +"%Y%m%d_%H%M%S")

# Run the training script in the background, logging stdout/stderr to a timestamped file
nohup python -u torch-mini.py config-model1.json > batch_scripts/run_logs/run_${timestamp}.log 2>&1 &

# Detach the job so it keeps running after you log out
disown

echo "Training started; logs -> batch_scripts/run_logs/run_${timestamp}.log"
