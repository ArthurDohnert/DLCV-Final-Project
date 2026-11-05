# Semana 1 — Setup do PCAD e dos dados (passo a passo)

Ao final: ambiente funcionando no PCAD, GPU testada, Track 2 baixada direto no PCAD e inspecionada, split por tile salvo.

## Decisões que economizam 150 GB

- **Não baixe nenhum arquivo "Multispectral images".** Usamos só RGB. Isso corta ~125 GB.
- **Não baixe a Track 1.** Os zips de RGB da Track 2 já trazem `LEFT_AGL` (altura acima do solo) para a imagem esquerda de cada par; o zip Reference traz as disparidades (`LEFT_DSP`). Então o Caminho A treina com `LEFT_RGB -> LEFT_AGL` e o Caminho B com `LEFT_RGB + RIGHT_RGB -> LEFT_DSP`, **nas mesmas imagens**. A comparação A x B fica perfeita.
- **Não baixe "Validation data".** Ela não tem verdade de solo (o gabarito ficou com a organização do concurso). Nosso val/teste sai dos tiles de treino, separados por tile.
- Total: ~33 GB (Track 2 RGB 1/4 a 4/4 + Track 2 Reference). Para começar hoje, só ~5 GB.
- Fatos do código oficial (`pubgeo/dfc2019`): disparidade **pode ser negativa**, pixels sem verdade valem **-999**.

## Rodando no notebook (enquanto o PCAD está fora do ar)

Com `Train-Track2-Truth.zip` e `Train-Track2-RGB-2.zip` na raiz do projeto, no PowerShell:

```powershell
cd C:\dev\DLCV-Final-Project
powershell -ExecutionPolicy Bypass -File scripts\local\setup_venv.ps1   # cria .venv (Python 3.11.9 + PyTorch CUDA)
.venv\Scripts\python scripts\local\extract_data.py                     # extrai só o necessário para data\dfc2019\track2
.venv\Scripts\python src\data\inspect_track2.py --data data\dfc2019\track2
```

Resultado da primeira inspeção (RGB 2/4, 4/11/2026): 454 pares, todos de Jacksonville, em 9 tiles; imagens 1024×1024; disparidade inteira, de −75 a 128 px (99,8% entre −28 e 30); 20% dos pixels sem verdade (−999); altura de 0 a 209 m (99,9% abaixo de 29 m).

## Quem faz o quê

| Passo | Quem | Por quê |
|---|---|---|
| Escrever scripts, código, configs | Claude (direto nesta pasta) | já tenho acesso à pasta do repositório |
| SSH no PCAD, rodar comandos lá | você | a conta e a chave SSH são suas; meu ambiente não alcança a rede da UFRGS |
| Login no IEEE DataPort e copiar links | você | exige seu login |
| Ler logs e erros, ajustar código | Claude | cole a saída aqui no chat |

---

## Passo 1 — Atalho de SSH no Windows (5 min)

No PowerShell:

```powershell
notepad $HOME\.ssh\config
```

Cole (troque `SEU_USUARIO` e o nome da chave, se for diferente de `id_ed25519`):

```
Host pcad
    HostName gppd-hpc.inf.ufrgs.br
    User SEU_USUARIO
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

Teste: `ssh pcad`. Se travar sem responder (timeout), ligue a VPN da UFRGS e tente de novo. A documentação do PCAD não exige VPN, mas a rede pode bloquear acesso de fora.

## Passo 2 — Mandar os scripts para o GitHub (no Windows, 2 min)

```powershell
cd C:\dev\DLCV-Final-Project
git add .
git commit -m "Semana 1: scripts de setup do PCAD e inspeção da Track 2"
git push
```

## Passo 3 — Primeiro login e checagem da conta (5 min)

```bash
ssh pcad
git clone https://github.com/ArthurDohnert/DLCV-Final-Project.git ~/DLCV-Final-Project
cd ~/DLCV-Final-Project
bash scripts/pcad/check_account.sh
```

Se o repositório for privado, o clone pede senha: use um token do GitHub (Settings > Developer settings > Fine-grained tokens, só leitura deste repo) como senha.

**Cole a saída do check_account.sh no chat.** O que importa: espaço livre no HOME (precisamos de ~80 GB: 33 GB de zips + 33 GB descompactados; depois apagamos os zips) e se a partição `tupi` aparece.
Se o espaço não der, peça cota na lista `hpc-users-l@inf.ufrgs.br`.

## Passo 4 — Use tmux para tudo que demora

Downloads e instalação continuam rodando mesmo se sua internet cair.

```bash
tmux new -s sat3d        # cria a sessão
# Ctrl+b e depois d      -> sai sem matar nada
tmux attach -t sat3d     # volta para ela
# Ctrl+b e depois c      -> nova janela; Ctrl+b e depois n -> próxima janela
```

Se `tmux` não existir, use `screen -S sat3d` (volta com `screen -r sat3d`).

## Passo 5 — Instalar o ambiente Python (15 min, roda sozinho)

Dentro do tmux:

```bash
cd ~/DLCV-Final-Project
bash scripts/pcad/setup_env.sh
```

Termina com `OK torch ...`. Enquanto roda, faça o Passo 6 em outra janela do tmux.

## Passo 6 — Baixar os dados direto no PCAD

Os links do IEEE DataPort exigem login e expiram em pouco tempo. O truque é pegar o link no navegador e colar no PCAD na hora:

1. No navegador do notebook, entre no IEEE DataPort e abra a página do DFC2019.
2. Clique no arquivo (ex.: `Track 2 / Training data / Reference`). O download começa.
3. Abra a lista de downloads (`Ctrl+J`), clique com o botão direito no download > **Copiar link**. Depois **cancele** o download no notebook.
4. No PCAD, numa janela do tmux, rode na hora (link entre aspas simples):

```bash
cd ~/DLCV-Final-Project
bash scripts/pcad/download.sh Train-Track2-Truth.zip 'COLE_O_LINK_AQUI'
```

Ordem (os dois primeiros bastam para começar hoje):

| # | Arquivo no DataPort | Nome para salvar | Tamanho |
|---|---|---|---|
| 1 | Track 2 / Training data / Reference | Train-Track2-Truth.zip | 1.68 GB |
| 2 | Track 2 / Training data / RGB images 2/4 | Train-Track2-RGB-2.zip | 3.42 GB |
| 3 | Track 2 / Training data / RGB images 4/4 | Train-Track2-RGB-4.zip | 4.98 GB |
| 4 | Track 2 / Training data / RGB images 3/4 | Train-Track2-RGB-3.zip | 10.4 GB |
| 5 | Track 2 / Training data / RGB images 1/4 | Train-Track2-RGB-1.zip | 12.52 GB |

Pode rodar vários ao mesmo tempo, um por janela do tmux.

Problemas comuns:
- **Erro 403 / "Forbidden"**: o link expirou. Copie um novo e rode o mesmo comando; ele continua de onde parou.
- **O arquivo baixado é uma página HTML pequena**: o link exigia cookies. Plano B: no navegador, aperte F12 > aba Network, clique no arquivo, clique com o botão direito na requisição do download > Copy > **Copy as cURL (bash)**, cole no PCAD e acrescente no final ` -o ~/data/dfc2019/zips/NOME.zip`.

## Passo 7 — Descompactar (quando os arquivos 1 e 2 terminarem)

```bash
bash scripts/pcad/unzip_data.sh
```

## Passo 8 — Testar a GPU (job de 20 min na tupi)

```bash
cd ~/DLCV-Final-Project && mkdir -p logs
sbatch scripts/slurm/gpu_test.sbatch
squeue -u $USER                 # acompanha: PD = na fila, R = rodando
cat logs/gpu_test_*.out         # quando terminar
```

Deve aparecer `RTX 4090`, o tempo por iteração e `TUDO OK`. Se ficar muito tempo em `PD`, troque `--partition=tupi` por `poti` no arquivo.

## Passo 9 — Inspecionar os dados e criar o split (job de CPU)

```bash
sbatch scripts/slurm/inspect_data.sbatch
cat logs/inspect_data_*.out
```

Traga a figura para o notebook (no PowerShell do Windows):

```powershell
mkdir C:\dev\DLCV-Final-Project\outputs -Force
scp "pcad:~/DLCV-Final-Project/outputs/inspect/*" C:\dev\DLCV-Final-Project\outputs\
```

**Cole no chat a saída dos logs `gpu_test` e `inspect_data`.** Com eles eu configuro o dataloader, o intervalo de disparidade da PSMNet e o teste de overfit.

## Regras do PCAD que importam

- Front-end (`gppd-hpc`): só para editar, baixar, instalar e submeter. Nada de treinar nele.
- `$HOME` fica no front-end e é visto por todos os nós (via rede, mais lento). `$SCRATCH` é o disco local de cada nó: rápido, mas temporário e apagável a qualquer momento. Nos treinos, copiamos os dados para `$SCRATCH` no início do job.
- Limite de 24h por job na maioria das partições. Partição `shared` é para testes rápidos.
- Não há backup. Código no GitHub; resultados importantes, copie para o notebook.
- GPUs úteis: `tupi` (RTX 4090, 24 GB), `poti` (RTX 4070, 12 GB). Evite `draco` (Tesla K20m, antiga demais para o PyTorch atual).

## Termos de uso do dataset

Nunca suba os dados para o GitHub (o `.gitignore` já bloqueia). No relatório, cite o DFC2019 e inclua o agradecimento exigido pela organização. Publicação científica precisa de aprovação prévia deles; trabalho de disciplina e post mostrando resultados não são publicação científica, mas não distribua os arquivos.
