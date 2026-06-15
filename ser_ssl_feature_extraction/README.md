# SER SSL Feature Extraction

This project extracts and caches self-supervised speech features for speech emotion recognition (SER). It supports WavLM through Hugging Face Transformers and provides an adapter for emotion2vec through FunASR or ModelScope.

The project only handles SSL feature extraction, saving, and loading. It does not implement a CNN-Transformer classifier or any training loop.

## Project Layout

```text
ser_ssl_feature_extraction/
├── configs/feature_extract.yaml
├── configs/feature_extract_wavlm_cpu.yaml
├── configs/feature_extract_wavlm_gpu.yaml
├── configs/feature_extract_emotion2vec_cpu.yaml
├── configs/feature_extract_emotion2vec_gpu.yaml
├── data/
├── features/
├── src/
│   ├── audio_utils.py
│   ├── dataset_parser.py
│   ├── ssl_extractors.py
│   ├── extract_features.py
│   └── feature_dataset.py
├── requirements.txt
├── requirements-cpu.txt
├── requirements-gpu-cu121.txt
└── README.md
```

## Install

```bash
cd ser_ssl_feature_extraction
pip install -r requirements-cpu.txt
```

`requirements-cpu.txt` pins dependency versions and uses the Tsinghua PyPI mirror by default. PyTorch and torchaudio use the official PyTorch CPU wheel index as an extra source because CPU-specific wheels such as `torch==2.5.1+cpu` are published there.

For CUDA 12.1 servers, use `requirements-gpu-cu121.txt`. It pins the same project dependencies but installs `torch==2.5.1+cu121` and `torchaudio==2.5.1+cu121` from the official PyTorch CUDA 12.1 wheel index.

For a CPU-only Miniconda environment on Windows:

```powershell
cd D:\MyCode\SER\SSL_feature_fusion\emotion2vec\ser_ssl_feature_extraction
powershell -ExecutionPolicy Bypass -File .\setup_cpu_env.ps1
conda activate ser_ssl_cpu
python -m src.extract_features --config configs/feature_extract.yaml
```

If you prefer manual commands:

```powershell
conda create -n ser_ssl_cpu python=3.10 pip -y
conda activate ser_ssl_cpu
pip install -r requirements-cpu.txt
```

For a GPU Miniconda environment on a Linux server with a CUDA 12.1-compatible NVIDIA driver:

```bash
cd /path/to/ser_ssl_feature_extraction
bash setup_gpu_env.sh
conda activate ser_ssl_gpu
python -m src.extract_features --config configs/feature_extract_emotion2vec_gpu.yaml
```

Manual GPU commands:

```bash
conda create -n ser_ssl_gpu python=3.10 pip -y
conda activate ser_ssl_gpu
pip install -r requirements-gpu-cu121.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

If your server driver does not support CUDA 12.1 wheels, replace the PyTorch wheel index and package suffix in `requirements-gpu-cu121.txt` with the CUDA version supported by your server, for example `cu118`.

## Prepare RAVDESS

Download and extract RAVDESS manually, then place it under:

```text
ser_ssl_feature_extraction/data/RAVDESS/
```

Actor subfolders are fine because the parser recursively scans `.wav` files.

## Configure

Edit `configs/feature_extract.yaml`.

Important fields:

- `dataset.name`: currently `ravdess`
- `dataset.root_dir`: dataset root directory
- `audio.sample_rate`: default `16000`
- `audio.max_duration`: crop or pad duration in seconds, or `null`
- `ssl.model_type`: `wavlm` or `emotion2vec`
- `ssl.model_name`: model id or local path
- `output.feature_dir`: output cache directory
- `output.overwrite`: whether to regenerate existing `.pt` files

Ready-to-use config templates:

- `configs/feature_extract_wavlm_cpu.yaml`
- `configs/feature_extract_wavlm_gpu.yaml`
- `configs/feature_extract_emotion2vec_cpu.yaml`
- `configs/feature_extract_emotion2vec_gpu.yaml`

## Extract WavLM Features

The default config uses:

```yaml
ssl:
  model_type: wavlm
  model_name: microsoft/wavlm-base-plus
  layer: mean_last4
  device: cpu
```

Run:

```bash
python -m src.extract_features --config configs/feature_extract.yaml
```

Or choose a template directly:

```bash
python -m src.extract_features --config configs/feature_extract_wavlm_cpu.yaml
python -m src.extract_features --config configs/feature_extract_wavlm_gpu.yaml
```

If CUDA is unavailable, the extractor automatically falls back to CPU and prints a message.

## Extract emotion2vec Features

Change the SSL section:

```yaml
ssl:
  model_type: emotion2vec
  model_name: emotion2vec/emotion2vec_plus_seed
  device: cuda
```

Then run:

```bash
python -m src.extract_features --config configs/feature_extract.yaml
```

Or choose a template directly:

```bash
python -m src.extract_features --config configs/feature_extract_emotion2vec_cpu.yaml
python -m src.extract_features --config configs/feature_extract_emotion2vec_gpu.yaml
```

emotion2vec releases may expose different APIs. The environment-specific logic is isolated in `src/ssl_extractors.py` inside `Emotion2VecExtractor`. If your local model returns a different key or requires a different pipeline task, adjust that class only.

## Output Format

Each audio file is saved as one `.pt` file:

```python
{
    "feature": torch.Tensor,  # [T, D] or [1, D]
    "label": int,
    "label_name": str,
    "path": str,
    "speaker": str,
    "sr": int,
    "ssl_model": str,
}
```

The script also writes `metadata.csv`:

```text
feature_path,audio_path,label,label_name,speaker,num_frames,feature_dim
```

Failed audio files are recorded in `failed.txt` without stopping the whole extraction run.

## Load Features For Later Models

```python
from src.feature_dataset import SSLFeatureDataset, ssl_feature_collate_fn
from torch.utils.data import DataLoader

dataset = SSLFeatureDataset("features/ravdess_wavlm_mean_last4/metadata.csv")
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=ssl_feature_collate_fn)

for features, lengths, labels, attention_mask in loader:
    print(features.shape)
    print(lengths)
    print(labels)
    print(attention_mask.shape)
    break
```

`features` has shape `[B, T_max, D]`, `lengths` stores the valid frame counts, and `attention_mask` marks valid frames with `1` and padded frames with `0`.

## Minimal Run

```bash
cd ser_ssl_feature_extraction
pip install -r requirements-cpu.txt
python -m src.extract_features --config configs/feature_extract.yaml
```

## Common Issues

CUDA is unavailable: use `ssl.device: cpu`, or install a CUDA-compatible PyTorch build. The WavLM extractor falls back to CPU automatically.

Audio sampling rates differ: `load_audio` resamples every file to `audio.sample_rate`, which is `16000` by default.

emotion2vec loading fails: install `funasr` or `modelscope`, check the model id or local model path, and adjust `Emotion2VecExtractor` if your installed release uses a different output key.

Feature lengths differ: this is normal for frame-level SSL features. Use `ssl_feature_collate_fn` to pad features and create `attention_mask`.

Windows path issues: use forward slashes in YAML when possible, or quote paths such as `"D:/datasets/RAVDESS"`.
