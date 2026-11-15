#!/bin/bash
# Passo 7 do guia: extrai os .tif dos zips da Track 2 para ~/data/dfc2019/track2.
# O PCAD não tem o programa unzip, então a extração é feita em Python (ambiente sat3d).
# Pode rodar de novo depois de baixar outro zip: o que já foi extraído é pulado.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate sat3d
python "$REPO_DIR/scripts/local/extract_data.py" --zips "$HOME/data/dfc2019/zips" --out "$HOME/data/dfc2019/track2"
echo "Total de .tif: $(find "$HOME/data/dfc2019/track2" -maxdepth 1 -name '*.tif' | wc -l)"
