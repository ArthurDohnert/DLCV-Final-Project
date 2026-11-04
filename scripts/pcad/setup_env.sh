#!/bin/bash
# Passo 5 do guia: instala Miniforge + ambiente "sat3d" no seu HOME do PCAD.
# Roda uma vez, no front-end. Demora uns 10-15 min. Ocupa uns 6-8 GB.
set -euo pipefail
if [ ! -d "$HOME/miniforge3" ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge_$USER.sh
  bash /tmp/miniforge_$USER.sh -b -p "$HOME/miniforge3"
  rm /tmp/miniforge_$USER.sh
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"
if ! conda env list | grep -q '^sat3d '; then
  conda create -y -n sat3d python=3.11
fi
conda activate sat3d
# PyTorch com CUDA 12.1 (funciona na RTX 4090 da tupi e na 4070 da poti)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
pip install -r "$REPO_DIR/environment.txt"
# baixa os pesos ImageNet já agora (caso os nós de computação não tenham internet)
python -c "import segmentation_models_pytorch as smp; smp.Unet('resnet34', encoder_weights='imagenet'); smp.Unet('efficientnet-b4', encoder_weights='imagenet'); print('pesos ImageNet em cache')"
python -c "import torch, segmentation_models_pytorch, tifffile; print('OK torch', torch.__version__)"
echo "Ambiente pronto. Para usar: source ~/miniforge3/etc/profile.d/conda.sh && conda activate sat3d"
