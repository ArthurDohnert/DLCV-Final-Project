# Cria a venv .venv do projeto no Windows com Python 3.11.9 (via pyenv-win)
# e instala PyTorch com CUDA 12.1 + os pacotes de requirements.txt, nas versões fixadas.
#
# Uso, na raiz do projeto (PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts\local\setup_venv.ps1
# O "-ExecutionPolicy Bypass" vale só para esta execução; não muda nada no sistema.

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
$PyVersion = "3.11.9"

if (-not (Get-Command pyenv -ErrorAction SilentlyContinue)) {
    throw "pyenv não encontrado no PATH. Abra um terminal novo ou confira a instalação do pyenv-win."
}
if (-not ((pyenv versions --bare) -match "^$PyVersion$")) {
    Write-Host "Instalando Python $PyVersion pelo pyenv (alguns minutos)..."
    pyenv install $PyVersion
}
pyenv local $PyVersion
$Python = (pyenv which python).Trim()
Write-Host "Python base: $Python"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $Python -m venv .venv
}
$VenvPy = ".venv\Scripts\python.exe"
& $VenvPy -m pip install --upgrade pip
& $VenvPy -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
& $VenvPy -m pip install -r requirements.txt
& $VenvPy -c "import torch, segmentation_models_pytorch, tifffile; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem GPU')"
Write-Host ""
Write-Host "Pronto. Para ativar a venv:  .venv\Scripts\Activate.ps1"
