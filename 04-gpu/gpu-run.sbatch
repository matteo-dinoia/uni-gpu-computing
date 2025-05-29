#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=result.out
#SBATCH --error=result.err
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#module load cuda/12.1

./compiled.exec
