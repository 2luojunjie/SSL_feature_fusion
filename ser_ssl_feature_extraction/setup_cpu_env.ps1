$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "Creating conda environment: ser_ssl_cpu"
conda create -n ser_ssl_cpu python=3.10 pip -y

Write-Host "Installing CPU dependencies from requirements-cpu.txt"
conda run -n ser_ssl_cpu python -m pip install -r requirements-cpu.txt

Write-Host "Activating environment and verifying installed packages"
conda run -n ser_ssl_cpu python -c "import torch, torchaudio, transformers, pandas, yaml, soundfile; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('torchaudio=', torchaudio.__version__); print('transformers=', transformers.__version__)"

Write-Host ""
Write-Host "Environment is ready."
Write-Host "Activate it with:"
Write-Host "  conda activate ser_ssl_cpu"
Write-Host ""
Write-Host "Run feature extraction with:"
Write-Host "  python -m src.extract_features --config configs/feature_extract.yaml"
