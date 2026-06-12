"""Self-supervised speech model feature extractors."""

from __future__ import annotations

import logging
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


class BaseSSLExtractor:
    """Common interface for SSL speech feature extractors."""

    def extract(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract features from a 1D waveform and return `[T, D]`."""
        raise NotImplementedError


def resolve_device(device: str) -> torch.device:
    """Return a usable torch device, falling back to CPU when needed."""
    requested = device.lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    try:
        return torch.device(device)
    except Exception:
        print(f"Invalid device '{device}'. Falling back to CPU.")
        return torch.device("cpu")


class WavLMExtractor(BaseSSLExtractor):
    """Hugging Face WavLM frame-level feature extractor."""

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        device: str = "cuda",
        layer: str = "last",
        freeze: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        if layer not in {"last", "mean_last4"}:
            raise ValueError("WavLM layer must be 'last' or 'mean_last4'.")

        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise ImportError("Please install transformers to use WavLMExtractor.") from exc

        self.model_name = model_name
        self.layer = layer
        self.device = resolve_device(device)
        resolved_cache_dir = str(Path(cache_dir).expanduser().resolve()) if cache_dir else None
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir=resolved_cache_dir)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            cache_dir=resolved_cache_dir,
        ).to(self.device)
        self.model.eval()

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def extract(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if wav.ndim != 1:
            raise ValueError(f"WavLMExtractor expects a 1D waveform, got shape {tuple(wav.shape)}")

        inputs = self.feature_extractor(
            wav.detach().cpu().numpy(),
            sampling_rate=sr,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            if self.layer == "last":
                features = outputs.last_hidden_state.squeeze(0)
            else:
                hidden_states = outputs.hidden_states
                if hidden_states is None or len(hidden_states) < 4:
                    raise RuntimeError("Model did not return enough hidden states for mean_last4.")
                features = torch.stack(hidden_states[-4:], dim=0).mean(dim=0).squeeze(0)

        return features.detach().cpu().contiguous()


class Emotion2VecExtractor(BaseSSLExtractor):
    """Adapter for emotion2vec models published through FunASR/ModelScope.

    emotion2vec releases have changed interfaces across versions. The
    environment-specific calls are intentionally isolated in this class so they
    can be adjusted without touching the rest of the project.
    """

    def __init__(
        self,
        model_name: str = "emotion2vec/emotion2vec_plus_seed",
        device: str = "cuda",
        freeze: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.freeze = freeze
        self.backend = ""
        self.model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        funasr_spec = find_spec("funasr")
        if funasr_spec is not None:
            try:
                from funasr import AutoModel

                self.model = AutoModel(model=self.model_name, device=str(self.device))
                self.backend = "funasr"
                return
            except Exception as exc:
                raise RuntimeError(
                    "Found funasr, but failed to load emotion2vec. "
                    "Check model_name, network access, cache path, and FunASR version. "
                    f"Original error: {exc}"
                ) from exc

        modelscope_spec = find_spec("modelscope")
        if modelscope_spec is not None:
            try:
                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks

                self.model = pipeline(task=Tasks.emotion_recognition, model=self.model_name)
                self.backend = "modelscope"
                return
            except Exception as exc:
                raise RuntimeError(
                    "Found modelscope, but failed to load emotion2vec. "
                    "Check whether this model supports the ModelScope emotion pipeline. "
                    f"Original error: {exc}"
                ) from exc

        raise ImportError(
            "Emotion2VecExtractor requires either funasr or modelscope. "
            "Install one of them, for example: pip install funasr modelscope. "
            "If your local emotion2vec package exposes another API, update Emotion2VecExtractor._load_model()."
        )

    def extract(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if wav.ndim != 1:
            raise ValueError(f"Emotion2VecExtractor expects a 1D waveform, got shape {tuple(wav.shape)}")

        if self.backend == "funasr":
            return self._extract_with_funasr(wav, sr)
        if self.backend == "modelscope":
            return self._extract_with_modelscope(wav, sr)
        raise RuntimeError("Emotion2VecExtractor backend is not initialized.")

    def _extract_with_funasr(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            import torchaudio

            torchaudio.save(str(tmp_path), wav.detach().cpu().unsqueeze(0), sample_rate=sr)
            with torch.no_grad():
                result = self.model.generate(input=str(tmp_path), granularity="utterance", extract_embedding=True)
            feature = self._coerce_result_to_tensor(result)
            if feature.size(0) == 1:
                LOGGER.info("emotion2vec FunASR returned utterance-level embedding with shape [1, D].")
            return feature
        finally:
            tmp_path.unlink(missing_ok=True)

    def _extract_with_modelscope(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            import torchaudio

            torchaudio.save(str(tmp_path), wav.detach().cpu().unsqueeze(0), sample_rate=sr)
            with torch.no_grad():
                result = self.model(str(tmp_path))
            feature = self._coerce_result_to_tensor(result)
            if feature.size(0) == 1:
                LOGGER.info("emotion2vec ModelScope returned utterance-level embedding with shape [1, D].")
            return feature
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _coerce_result_to_tensor(result: Any) -> torch.Tensor:
        """Convert common FunASR/ModelScope outputs to a `[T, D]` tensor."""
        candidate = result
        if isinstance(result, list) and result:
            candidate = result[0]
        if isinstance(candidate, dict):
            for key in ("feats", "features", "embedding", "spk_embedding", "xvector", "scores"):
                if key in candidate:
                    candidate = candidate[key]
                    break

        tensor = torch.as_tensor(candidate, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim > 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        if tensor.ndim != 2:
            raise RuntimeError(f"Could not convert emotion2vec output to [T, D], got shape {tuple(tensor.shape)}")
        return tensor.detach().cpu().contiguous()


def build_extractor(config: dict[str, Any]) -> BaseSSLExtractor:
    """Create an SSL extractor from the `ssl` config section."""
    model_type = str(config.get("model_type", "wavlm")).lower()
    model_name = str(config.get("model_name", ""))
    device = str(config.get("device", "cuda"))
    cache_dir = config.get("cache_dir")

    if model_type == "wavlm":
        return WavLMExtractor(
            model_name=model_name or "microsoft/wavlm-base-plus",
            device=device,
            layer=str(config.get("layer", "last")),
            freeze=bool(config.get("freeze", True)),
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    if model_type in {"emotion2vec", "e2v"}:
        return Emotion2VecExtractor(
            model_name=model_name or "emotion2vec/emotion2vec_plus_seed",
            device=device,
            freeze=bool(config.get("freeze", True)),
        )
    raise ValueError(f"Unsupported SSL model_type '{model_type}'. Use 'wavlm' or 'emotion2vec'.")
