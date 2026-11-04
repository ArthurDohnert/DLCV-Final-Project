#!/bin/bash
# Passo 3 do guia: roda no front-end do PCAD (gppd-hpc), logo depois do primeiro login.
# Mostra espaço em disco, cota, partições com GPU e se a internet funciona.
echo "== Usuário e máquina =="; whoami; hostname
echo; echo "== Espaço no HOME (onde os dados vão morar) =="; df -h "$HOME"
echo; echo "== Cota (pode não existir) =="; quota -s 2>/dev/null || echo "comando quota indisponível"
echo; echo "== Já ocupado no HOME =="; du -sh "$HOME" 2>/dev/null
echo; echo "== Partições de GPU que interessam =="
sinfo -p tupi,poti,beagle -o "%P %N %G %t %l" 2>/dev/null || sinfo
echo; echo "== Internet a partir do front-end =="
curl -sI https://pypi.org | head -1 || echo "SEM INTERNET no front-end"
echo; echo "== Ferramentas =="; which tmux screen wget unzip git python3 2>/dev/null
