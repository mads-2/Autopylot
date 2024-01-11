#!/bin/bash

#SBATCH -p elipierilab
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -J autopilot_test
#SBATCH --mem=75G
#SBATCH -t 30:00:00
#SBATCH --constraint=[h100]
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

#Load necessary modules

#srun run.sh
module load tc/23.08

python autopilot.py -i test_input.yaml
