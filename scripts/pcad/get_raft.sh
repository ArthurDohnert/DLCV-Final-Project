#!/bin/bash
# Baixa o código oficial da RAFT-Stereo e os pesos pré-treinados para third_party/RAFT-Stereo.
# Roda uma vez, no front-end do PCAD (precisa de internet), dentro da pasta do repositório.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
D="$REPO_DIR/third_party/RAFT-Stereo"
[ -d "$D/core" ] || git clone --depth 1 https://github.com/princeton-vl/RAFT-Stereo "$D"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate sat3d
pip install -q opt_einsum==3.4.0
if [ ! -f "$D/models/raftstereo-sceneflow.pth" ]; then
  # mesmo link do download_models.sh oficial
  wget -q --show-progress -O "$D/models.zip" \
    "https://www.dropbox.com/scl/fi/5khx1bhz84dapi8vtwapg/models.zip?rlkey=ggddrn1du1iiq6mgc2dsdpmwi&dl=1"
  python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$D/models.zip" "$D"
  mkdir -p "$D/models"
  find "$D" -maxdepth 2 -name 'raftstereo-*.pth' -not -path "$D/models/*" -exec mv {} "$D/models/" \;
  rm -f "$D/models.zip"
fi
ls -lh "$D/models"
python -c "import torch, sys; sys.path.insert(0, '$REPO_DIR'); from src.models.raft import RaftDisparity; \
m = RaftDisparity(); m.load_pretrained('$D/models/raftstereo-sceneflow.pth'); print('RAFT-Stereo OK: pesos SceneFlow carregados')"
