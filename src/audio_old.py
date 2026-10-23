
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16000


def prepare_audio(
    audio: np.ndarray,
    source_sr: int,
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Convert audio to mono float32 at the target sample rate."""

    audio = np.asarray(audio)

    if audio.size == 0:
        raise ValueError("Audio cannot be empty.")

    if audio.ndim not in (1, 2):
        raise ValueError(
            "Audio must have shape (samples,) or (samples, channels)."
        )

    if not np.issubdtype(audio.dtype, np.number):
        raise TypeError("Audio must contain numeric samples.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or infinite values.")

    if source_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive.")

    # Convert integer PCM samples to float32.
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = max(abs(info.min), info.max)
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)

    # Convert (samples, channels) to mono.
    if audio.ndim == 2:
        if audio.shape[1] == 0:
            raise ValueError("Audio must contain at least one channel.")

        audio = audio.mean(axis=1, dtype=np.float32)

    # Resample if the sample rates differ.
    if source_sr != target_sr:
        audio = librosa.resample(
            audio,
            orig_sr=source_sr,
            target_sr=target_sr,
        )

    return np.ascontiguousarray(audio, dtype=np.float32)


def load_audio(path: str | Path) -> np.ndarray:
    """Load a file as mono float32 audio at 16 kHz."""

    audio, source_sr = librosa.load(
        path,
        sr=None,
        mono=False,
        dtype=np.float32,
    )

    # librosa returns multichannel audio as (channels, samples).
    if audio.ndim == 2:
        audio = audio.T

    return prepare_audio(audio, source_sr=source_sr)
