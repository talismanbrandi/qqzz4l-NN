#!/bin/bash
#SBATCH --nodes=1                       # number of nodes requested
#SBATCH --ntasks=1                      # number of tasks (default: 1)
#SBATCH --ntasks-per-node=1             # number of tasks per node (default: whole node)
#SBATCH --partition=maxgpu              # partition to run in (all or maxwell)
#SBATCH --job-name=DNN-9847e0b7             # job name
#SBATCH --output=DNN-TF-9847e0b7-%N-%j.out  # output file name
#SBATCH --error=DNN-TF-9847e0b7-%N-%j.err   # error file name
#SBATCH --time=96:00:00                 # runtime requested
#SBATCH --mail-user=ayan.paul@desy.de   # notification email
#SBATCH --mail-type=END,FAIL            # notification type
#SBATCH --export=ALL
#SBATCH --constraint=A100
export LD_PRELOAD=""

# load module
module load python/3.10
module load cuda/11.4

# run the application:
source ../.hpreg/bin/activate
python3.10 torch-NN.py config-9847e0b7.json
deactivate
rm config-9847e0b7.json
