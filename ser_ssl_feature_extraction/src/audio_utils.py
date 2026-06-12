"""Audio loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio


def load_audio(
    path: str,
    target_sr: int = 16000,
    max_duration: float | None = 6.0,
    normalize: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load a wav file as a mono 1D tensor.

    Args:
        path: Audio file path.
        target_sr: Target sampling rate.
        max_duration: If set, crop or zero-pad to this duration in seconds.
        normalize: Whether to peak-normalize the waveform.

    Returns:
        A tuple of `(waveform, sample_rate)`, where waveform has shape `[num_samples]`.

    Raises:
        RuntimeError: If the file cannot be read or preprocessed.
    """
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    try:
        wav, sr = torchaudio.load(str(audio_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio '{audio_path}': {exc}") from exc

    if wav.ndim != 2:
        raise RuntimeError(f"Expected torchaudio waveform shape [C, T], got {tuple(wav.shape)}")

    try:
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
            sr = target_sr

        wav_1d = wav.squeeze(0).to(torch.float32)

        if max_duration is not None:
            if max_duration <= 0:
                raise ValueError(f"max_duration must be positive or None, got {max_duration}")
            target_len = int(round(target_sr * max_duration))
            if wav_1d.numel() > target_len:
                wav_1d = wav_1d[:target_len]
            elif wav_1d.numel() < target_len:
                wav_1d = F.pad(wav_1d, (0, target_len - wav_1d.numel()))

        if normalize:
            peak = wav_1d.abs().max()
            if peak > 0:
                wav_1d = wav_1d / peak

        return wav_1d.contiguous(), sr
    except Exception as exc:
        raise RuntimeError(f"Failed to preprocess audio '{audio_path}': {exc}") from exc
