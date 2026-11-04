#!/bin/bash
# Passo 7 do guia: descompacta os zips da Track 2 numa pasta única.
set -euo pipefail
Z="$HOME/data/dfc2019/zips"; OUT="$HOME/data/dfc2019/track2"
mkdir -p "$OUT"
for f in "$Z"/*.zip; do
  echo "Descompactando $(basename "$f")..."
  unzip -n -q "$f" -d "$OUT"
done
# os zips trazem subpastas; junta todos os .tif numa pasta só
find "$OUT" -mindepth 2 -name '*.tif' -exec mv -n {} "$OUT"/ \;
find "$OUT" -mindepth 1 -type d -empty -delete
echo "Total de .tif: $(find "$OUT" -maxdepth 1 -name "*.tif" | wc -l)"
find "$OUT" -maxdepth 1 -name "*.tif" | head -12
