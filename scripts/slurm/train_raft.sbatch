#!/bin/bash
#SBATCH --job-name=raft
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --time=23:30:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Caminho B com RAFT-Stereo pré-treinada. Antes, uma vez: bash scripts/pcad/get_raft.sh
# Uso (na pasta do repositório):  sbatch -p poti scripts/slurm/train_raft.sbatch
# Na tupi (24 GB) dá para usar lotes maiores:  sbatch scripts/slurm/train_raft.sbatch --batch 4
# Rodar de novo com o mesmo --name continua do último checkpoint.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate sat3d
DATA="$HOME/data/dfc2019/track2"
if [ -n "$SCRATCH" ] && mkdir -p "$SCRATCH/track2" 2>/dev/null; then
  rsync -a --include='*_LEFT_RGB.tif' --include='*_RIGHT_RGB.tif' --include='*_LEFT_DSP.tif' --include='*_LEFT_AGL.tif' \
    --exclude='*' "$DATA/" "$SCRATCH/track2/" && DATA="$SCRATCH/track2"
fi
echo "Nó $(hostname) | dados em $DATA"; nvidia-smi --query-gpu=name,memory.total --format=csv
python -m src.train_raft --data "$DATA" --epochs 20 --crop 256 512 --batch 2 --workers 8 --name raft "$@" &
PID=$!
trap 'echo "Aviso do SLURM: salvando checkpoint"; kill -USR1 $PID' USR1
wait $PID; wait $PID
