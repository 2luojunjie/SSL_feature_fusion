"""Dataset parsers for speech emotion recognition corpora."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any


RAVDESS_EMOTIONS: dict[str, tuple[int, str]] = {
    "01": (0, "neutral"),
    "02": (1, "calm"),
    "03": (2, "happy"),
    "04": (3, "sad"),
    "05": (4, "angry"),
    "06": (5, "fearful"),
    "07": (6, "disgust"),
    "08": (7, "surprised"),
}


def parse_ravdess(root_dir: str) -> list[dict[str, Any]]:
    """Parse RAVDESS `.wav` files.

    RAVDESS filenames follow:
    `modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav`.
    """
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"RAVDESS root directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"RAVDESS root path is not a directory: {root}")

    samples: list[dict[str, Any]] = []
    for wav_path in sorted(root.rglob("*.wav"), key=lambda p: str(p.resolve())):
        parts = wav_path.stem.split("-")
        if len(parts) != 7:
            warnings.warn(f"Skip unrecognized RAVDESS filename: {wav_path}", stacklevel=2)
            continue

        emotion_code = parts[2]
        actor = parts[6]
        if emotion_code not in RAVDESS_EMOTIONS:
            warnings.warn(f"Skip file with unknown RAVDESS emotion code '{emotion_code}': {wav_path}", stacklevel=2)
            continue

        label, label_name = RAVDESS_EMOTIONS[emotion_code]
        samples.append(
            {
                "path": str(wav_path.resolve()),
                "label": label,
                "label_name": label_name,
                "speaker": actor,
            }
        )

    samples.sort(key=lambda item: item["path"])
    return samples


def parse_dataset(name: str, root_dir: str) -> list[dict[str, Any]]:
    """Dispatch to a dataset parser by name."""
    normalized = name.strip().lower()
    if normalized == "ravdess":
        return parse_ravdess(root_dir)
    raise ValueError(f"Unsupported dataset '{name}'. Currently supported: ravdess")
