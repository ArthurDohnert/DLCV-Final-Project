# PCAD — passo a passo inicial

Fonte: https://gppd-hpc.inf.ufrgs.br (conferido em 10/11/2026; a página não mudou desde a queda do /home).
Comandos com `PS>` rodam no PowerShell do Windows; os demais, no terminal do PCAD.

## 0. Antes de tudo: mandar o código para o GitHub (Windows)

    PS> cd C:\dev\DLCV-Final-Project
    PS> git add .
    PS> git commit -m "Caminho B, avaliação justa, visualizador 3D, proposta e scripts do PCAD"
    PS> git push

## 1. Entrar no PCAD

Atalho em `C:\Users\<você>\.ssh\config` (só uma vez):

    Host pcad
        HostName gppd-hpc.inf.ufrgs.br
        User SEU_USUARIO
        IdentityFile ~/.ssh/id_ed25519
        ServerAliveInterval 60

    PS> ssh pcad

Se travar sem resposta, ligue a VPN da UFRGS e tente de novo. O "fingerprint" que aparece na primeira vez: responda `yes`.

## 2. Ver o que a sua conta pode usar

    sinfo -s                          # todas as partições e quantos nós estão livres (idle)
    sinfo -p tupi,poti,beagle         # as que mais interessam
    echo $SCRATCH; df -h $HOME        # disco local dos nós e espaço no HOME

GPUs úteis: `tupi` (RTX 4090, 24 GB), `poti` (RTX 4070, 12 GB), `beagle` (2x GTX 1080 Ti).
Partições de projeto (cidia, saude, tsubasa, blaise, marcs, apolo, grace) só com autorização do coordenador
do projeto. Se a do seu orientador for uma delas, troque `--partition` nos .sbatch.
Se uma partição não for permitida, o `sbatch` recusa na hora com erro de "partition".

## 3. Baixar o repositório

    git clone https://github.com/ArthurDohnert/DLCV-Final-Project.git ~/DLCV-Final-Project
    cd ~/DLCV-Final-Project
    mkdir -p logs

## 4. tmux: para não perder nada se a conexão cair

    tmux new -s sat3d         # abre uma sessão
    Ctrl+b, depois d          # sai sem fechar (o que estiver rodando continua)
    tmux attach -t sat3d      # volta
    Ctrl+b, depois c          # nova aba;  Ctrl+b, depois n  # próxima aba

## 5. Instalar o ambiente Python (uma vez, ~15 min)

    bash scripts/pcad/setup_env.sh

Termina com `OK torch 2.5.1...`. Também baixa os pesos ImageNet, caso os nós de GPU não tenham internet.

## 6. Baixar os dados direto no PCAD (~33 GB)

No navegador do notebook, no IEEE DataPort (DFC2019): clique no arquivo, `Ctrl+J`, botão direito no
download > Copiar link, cancele o download. No PCAD, na hora (o link expira):

    bash scripts/pcad/download.sh Train-Track2-Truth.zip 'LINK'
    bash scripts/pcad/download.sh Train-Track2-RGB-1.zip 'LINK'
    bash scripts/pcad/download.sh Train-Track2-RGB-2.zip 'LINK'
    bash scripts/pcad/download.sh Train-Track2-RGB-3.zip 'LINK'
    bash scripts/pcad/download.sh Train-Track2-RGB-4.zip 'LINK'

Erro 403 = link expirou: copie outro e rode o mesmo comando (continua de onde parou).
Depois: `bash scripts/pcad/unzip_data.sh`

## 7. Testar a GPU e inspecionar os dados

    sbatch scripts/slurm/gpu_test.sbatch          # 20 min na tupi
    sbatch scripts/slurm/inspect_data.sbatch      # recria o split com todos os tiles (JAX + OMA)
    squeue -u $USER                               # PD = na fila, R = rodando
    cat logs/gpu_test_*.out logs/inspect_data_*.out

## 8. Treinar

    sbatch scripts/slurm/train_height.sbatch                       # Caminho A
    bash scripts/slurm/chain.sh scripts/slurm/train_stereo.sbatch 2  # Caminho B, 2 jobs de 24 h encadeados

Cada job copia os dados para o `$SCRATCH` do nó, salva checkpoint a cada época e, 10 min antes do
limite de 24 h, salva e sai. O job seguinte continua do checkpoint.

Acompanhar e cancelar:

    tail -f logs/psmnet_<JOBID>.out
    scancel <JOBID>

## Regras que importam

- Não treine no front-end (`gppd-hpc`): só edite, baixe, instale e submeta jobs.
- Não há backup do HOME nem do SCRATCH, e o SCRATCH pode ser apagado a qualquer momento.
  Código no GitHub; checkpoints e resultados importantes, copie para o notebook:
  `PS> scp -r pcad:~/DLCV-Final-Project/runs/psmnet C:\dev\DLCV-Final-Project\runs\`
- Avisos do PCAD chegam pela lista hpc-users-l@inf.ufrgs.br (seu e-mail da UFRGS).
