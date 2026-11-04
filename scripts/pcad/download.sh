#!/bin/bash
# Passo 6 do guia: baixa UM arquivo do IEEE DataPort direto no PCAD.
# Uso:  bash scripts/pcad/download.sh <nome_do_arquivo.zip> '<link copiado do navegador>'
# Coloque o link entre aspas simples. O link expira rápido: cole logo depois de copiar.
set -u
NAME="$1"; URL="$2"
DEST="$HOME/data/dfc2019/zips"
mkdir -p "$DEST" "$HOME/data/dfc2019/logs"
cd "$DEST"
wget -c --tries=20 --waitretry=10 --timeout=60 -O "$NAME" "$URL"
STATUS=$?
if [ $STATUS -eq 0 ] && unzip -tq "$NAME" >/dev/null 2>&1; then
  echo "OK: $NAME baixado e íntegro ($(du -h "$NAME" | cut -f1))"
else
  echo "FALHOU ($STATUS): $NAME. Se o erro foi 403, o link expirou: copie um novo e rode de novo (o -c continua de onde parou)."
fi
