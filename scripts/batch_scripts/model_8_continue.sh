#!/bin/bash
#SBATCH --partition=courses-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --job-name=m8_v100_cont
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --output=batch_scripts/run_logs/m8/model_8.%j.out
#SBATCH --error=batch_scripts/run_logs/m8/model_8.%j.out

lscpu
nvidia-smi
module load anaconda3
source activate myenv

python torch-NN.py config-model8_continue.json
