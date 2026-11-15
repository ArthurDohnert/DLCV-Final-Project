#!/bin/bash
# Submete N jobs encadeados: cada um só começa quando o anterior termina (limite de 24 h do PCAD).
# Uso: bash scripts/slurm/chain.sh <arquivo.sbatch> <N> [argumentos extras do treino]
F="$1"; N="$2"; shift 2
mkdir -p logs
JID=$(sbatch --parsable "$F" "$@") || exit 1
echo "job 1: $JID"
for i in $(seq 2 "$N"); do
  JID=$(sbatch --parsable --dependency=afterany:"$JID" "$F" "$@") || exit 1
  echo "job $i: $JID (espera o anterior)"
done
