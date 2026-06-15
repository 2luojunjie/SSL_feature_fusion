#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Creating conda environment: ser_ssl_gpu"
conda create -n ser_ssl_gpu python=3.10 pip -y

echo "Installing GPU dependencies from requirements-gpu-cu121.txt"
conda run -n ser_ssl_gpu python -m pip install -r requirements-gpu-cu121.txt

echo "Verifying installed packages"
conda run -n ser_ssl_gpu python - <<'PY'
import torch
import torchaudio
import transformers

print("torch=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
print("cuda_version=", torch.version.cuda)
print("gpu_count=", torch.cuda.device_count())
print("torchaudio=", torchaudio.__version__)
print("transformers=", transformers.__version__)
PY

echo ""
echo "Environment is ready."
echo "Activate it with:"
echo "  conda activate ser_ssl_gpu"
echo ""
echo "Run GPU feature extraction with:"
echo "  python -m src.extract_features --config configs/feature_extract_emotion2vec_gpu.yaml"
