#!/bin/bash
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --job-name=m5_h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --output=batch_scripts/run_logs/mini/model_5.%j.out
#SBATCH --error=batch_scripts/run_logs/mini/model_5.%j.out

lscpu
nvidia-smi
module load anaconda3
source activate myenv

python -u torch-mini.py config-model5.json
