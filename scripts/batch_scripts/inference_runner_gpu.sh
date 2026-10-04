#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --job-name=i1_h200
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --output=batch_scripts/run_logs/inference_runner/inf.%j.out
#SBATCH --error=batch_scripts/run_logs/inference_runner/inf.%j.out

lscpu
nvidia-smi
module load anaconda3
source activate myenv

python re-evaluation.py
