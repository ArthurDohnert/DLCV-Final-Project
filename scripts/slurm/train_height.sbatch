#!/bin/bash
#SBATCH --job-name=unet_r34
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --time=23:30:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Caminho A no PCAD. Uso (na pasta do repositório):
#   sbatch scripts/slurm/train_height.sbatch                    # padrão: U-Net ResNet34
#   sbatch scripts/slurm/train_height.sbatch --name unet_effb4 --encoder efficientnet-b4
# Rodar de novo com o mesmo --name continua do último checkpoint.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate sat3d
DATA="$HOME/data/dfc2019/track2"
# copia só o necessário para o disco local do nó (mais rápido que ler do HOME pela rede)
if [ -n "$SCRATCH" ] && mkdir -p "$SCRATCH/track2" 2>/dev/null; then
  rsync -a --include='*_LEFT_RGB.tif' --include='*_LEFT_AGL.tif' --exclude='*' "$DATA/" "$SCRATCH/track2/" && DATA="$SCRATCH/track2"
fi
echo "Nó $(hostname) | dados em $DATA"; nvidia-smi --query-gpu=name,memory.total --format=csv
python -m src.train_height --data "$DATA" --epochs 40 --crop 512 --batch 8 --workers 8 --name unet_r34 "$@" &
PID=$!
trap 'echo "Aviso do SLURM: salvando checkpoint"; kill -USR1 $PID' USR1
wait $PID; wait $PID
