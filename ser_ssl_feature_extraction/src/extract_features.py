"""Command line entry point for batch SSL feature extraction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from .audio_utils import load_audio
from .dataset_parser import parse_dataset
from .ssl_extractors import build_extractor


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return config


def feature_filename(audio_path: str) -> str:
    """Create a stable cache filename from an absolute audio path."""
    path = Path(audio_path)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}_{digest}.pt"


def configure_huggingface_cache(cache_dir: str | None) -> None:
    """Use a project-local Hugging Face cache unless the user already set one."""
    if not cache_dir:
        return

    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_path))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_path / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def write_metadata(metadata_path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["feature_path", "audio_path", "label", "label_name", "speaker", "num_frames", "feature_dim"]
    with metadata_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SSL speech features for SER datasets.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    audio_cfg = config.get("audio", {})
    ssl_cfg = config.get("ssl", {})
    output_cfg = config.get("output", {})

    dataset_name = str(dataset_cfg.get("name", "ravdess"))
    root_dir = str(dataset_cfg.get("root_dir", "./data/RAVDESS"))
    feature_dir = Path(str(output_cfg.get("feature_dir", "./features"))).expanduser().resolve()
    save_format = str(output_cfg.get("save_format", "pt")).lower()
    overwrite = bool(output_cfg.get("overwrite", False))

    if save_format != "pt":
        raise ValueError("Only save_format='pt' is currently supported.")

    feature_dir.mkdir(parents=True, exist_ok=True)
    failed_path = feature_dir / "failed.txt"
    metadata_path = feature_dir / "metadata.csv"

    samples = parse_dataset(dataset_name, root_dir)
    if not samples:
        raise RuntimeError(f"No valid samples found for dataset '{dataset_name}' under {root_dir}")

    configure_huggingface_cache(ssl_cfg.get("cache_dir"))
    extractor = build_extractor(ssl_cfg)
    metadata_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for sample in tqdm(samples, desc="Extracting SSL features"):
        audio_path = sample["path"]
        out_path = feature_dir / feature_filename(audio_path)

        try:
            if out_path.exists() and not overwrite:
                saved = torch.load(out_path, map_location="cpu", weights_only=True)
                feature = saved["feature"]
            else:
                wav, sr = load_audio(
                    audio_path,
                    target_sr=int(audio_cfg.get("sample_rate", 16000)),
                    max_duration=audio_cfg.get("max_duration", 6.0),
                    normalize=bool(audio_cfg.get("normalize", True)),
                )
                feature = extractor.extract(wav, sr)
                torch.save(
                    {
                        "feature": feature,
                        "label": int(sample["label"]),
                        "label_name": sample["label_name"],
                        "path": audio_path,
                        "speaker": sample["speaker"],
                        "sr": sr,
                        "ssl_model": str(ssl_cfg.get("model_name", "")),
                    },
                    out_path,
                )

            if feature.ndim != 2:
                raise RuntimeError(f"Feature must have shape [T, D], got {tuple(feature.shape)}")

            metadata_rows.append(
                {
                    "feature_path": str(out_path),
                    "audio_path": audio_path,
                    "label": int(sample["label"]),
                    "label_name": sample["label_name"],
                    "speaker": sample["speaker"],
                    "num_frames": int(feature.shape[0]),
                    "feature_dim": int(feature.shape[1]),
                }
            )
        except Exception as exc:
            message = f"{audio_path}\t{type(exc).__name__}: {exc}"
            failures.append(message)
            tqdm.write(f"Failed: {message}")

    write_metadata(metadata_path, metadata_rows)
    if failures:
        failed_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
    elif failed_path.exists():
        failed_path.unlink()

    print(f"Done. Saved {len(metadata_rows)} feature files to {feature_dir}")
    print(f"Metadata: {metadata_path}")
    if failures:
        print(f"Failures: {failed_path}")


if __name__ == "__main__":
    main()
