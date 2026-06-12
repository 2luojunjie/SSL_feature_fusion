"""PyTorch dataset for cached SSL speech features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SSLFeatureDataset(Dataset):
    """Load cached `.pt` feature files listed in `metadata.csv`."""

    def __init__(self, metadata_csv: str) -> None:
        self.metadata_csv = Path(metadata_csv).expanduser().resolve()
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV does not exist: {self.metadata_csv}")
        self.table = pd.read_csv(self.metadata_csv)
        if "feature_path" not in self.table.columns:
            raise ValueError(f"metadata.csv must contain a 'feature_path' column: {self.metadata_csv}")

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.table.iloc[idx]
        feature_path = Path(str(row["feature_path"])).expanduser()
        if not feature_path.is_absolute():
            feature_path = (self.metadata_csv.parent / feature_path).resolve()
        saved = torch.load(feature_path, map_location="cpu", weights_only=True)
        feature = saved["feature"].to(torch.float32)
        label = int(saved.get("label", row.get("label")))
        if feature.ndim != 2:
            raise RuntimeError(f"Expected feature shape [T, D], got {tuple(feature.shape)} in {feature_path}")
        return feature, label


def ssl_feature_collate_fn(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length SSL features in a batch."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    features, labels = zip(*batch)
    lengths = torch.tensor([feat.shape[0] for feat in features], dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = int(features[0].shape[1])

    padded: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for feat in features:
        if feat.ndim != 2:
            raise RuntimeError(f"Expected feature shape [T, D], got {tuple(feat.shape)}")
        if feat.shape[1] != feat_dim:
            raise RuntimeError("All features in a batch must have the same feature_dim.")
        pad_len = max_len - feat.shape[0]
        padded.append(F.pad(feat, (0, 0, 0, pad_len)))
        masks.append(torch.cat([torch.ones(feat.shape[0]), torch.zeros(pad_len)]))

    features_padded = torch.stack(padded, dim=0)
    attention_mask = torch.stack(masks, dim=0).to(torch.long)
    return features_padded, lengths, labels_tensor, attention_mask
