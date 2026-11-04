#!/bin/bash
#SBATCH --job-name=inspect_data
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Uso:  sbatch scripts/slurm/inspect_data.sbatch
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate sat3d
python src/data/inspect_track2.py --data "$HOME/data/dfc2019/track2" --out outputs/inspect
