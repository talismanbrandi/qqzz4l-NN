#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=i2_cpu
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=batch_scripts/run_logs/inference_runner/inf.%j.out
#SBATCH --error=batch_scripts/run_logs/inference_runner/inf.%j.out

lscpu
module load anaconda3
module load OpenJDK/22.0.2
source activate myenv

python re-evaluation.py
