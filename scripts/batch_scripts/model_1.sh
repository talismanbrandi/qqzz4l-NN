#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --job-name=m1_v100
#SBATCH --ntasks=1
#SBATCH --output=batch_scripts/run_logs/m1/model_1.%j.out
#SBATCH --error=batch_scripts/run_logs/m1/model_1.%j.out

lscpu
nvidia-smi
module load anaconda3/2022.05
module load OpenJDK/19.0.1
source activate myenv

python torch-NN.py config-model1.json
