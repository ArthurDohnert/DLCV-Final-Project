#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Uso (de dentro da pasta do repositório no PCAD):  mkdir -p logs && sbatch scripts/slurm/gpu_test.sbatch
# Se a tupi estiver lotada, troque a partição para poti (RTX 4070, 12 GB).
hostname; nvidia-smi
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate sat3d
python src/env_check.py
