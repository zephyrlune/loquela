from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

from scipy.fft import rfft
from scipy.signal import get_window

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_cnn_lstm_128_dropout_04_cnn_3.pt"
)

# Audio configuration — keep compatible with training
SAMPLE_RATE = 16000
FRAME_LENGTH = 0.025
FRAME_HOP = 0.020

N_FFT = 400
HOP_LENGTH = 320
N_MELS = 80

CHUNK_DURATION = 1.0
CHUNK_SAMPLES = int(
    SAMPLE_RATE * CHUNK_DURATION
)

# Choose GPU when available
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# phones
phones = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', '[SIL]', '[UNK]']

phone_to_id = {
    phone: i
    for i, phone in enumerate(phones)
}

id_to_phone = {
    i: phone
    for phone, i in phone_to_id.items()
}

NUM_CLASSES = len(phones)


# CNN-LSTM model

class CNNLSTM(nn.Module):

    def __init__(
        self,
        num_classes,
        hidden_size=128,
    ):
        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=(2, 1)
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=(2, 1)
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),
        )

        cnn_feature_size = 128 * 20

        self.lstm = nn.LSTM(
            input_size=cnn_feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # dropoout at 0.4
        self.dropout = nn.Dropout(
            p=0.4
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.cnn(x)

        x = x.permute(
            0, 3, 1, 2
        )

        x = x.flatten(
            start_dim=2
        )

        x, _ = self.lstm(x)

        x = self.dropout(x)

        x = self.classifier(x)

        return x

# load model

def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = CNNLSTM(num_classes=NUM_CLASSES)

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    assert model.classifier.out_features == NUM_CLASSES, (
        f"Model has {model.classifier.out_features} outputs, "
        f"but {NUM_CLASSES} phoneme classes were configured."
    )

    model = model.to(DEVICE)
    model.eval()

    return model


model = load_model()





# Audio → Log-Mel (old)

# def audio_to_log_mel(audio):
#     audio = np.asarray(audio, dtype=np.float32)

#     # Expect a mono audio buffer
#     if audio.ndim != 1:
#         raise ValueError(
#             "Expected mono audio with shape (samples,). "
#             f"Received shape: {audio.shape}"
#         )

#     if not np.isfinite(audio).all():
#         raise ValueError("Audio contains NaN or infinite values.")

#     # Pad or truncate to exactly one second
#     if len(audio) < CHUNK_SAMPLES:
#         audio = np.pad(
#             audio,
#             (0, CHUNK_SAMPLES - len(audio)),
#         )
#     else:
#         audio = audio[:CHUNK_SAMPLES]

#     log_mel = librosa.feature.melspectrogram(
#         y=audio,
#         sr=SAMPLE_RATE,
#         n_fft=N_FFT,
#         hop_length=HOP_LENGTH,
#         n_mels=N_MELS,
#         power=2.0,
#         center=False,
#     )

#     log_mel = librosa.power_to_db(
#         log_mel,
#         ref=np.max,
#     )

#     return log_mel.astype(np.float32, copy=False)

# real time feature caching

class RollingMelCache:
    """
    Maintains a rolling 1-second audio window and its mel-power features.

    Expected update size:
        640 samples = 40 ms at 16 kHz

    Feature shape:
        (80, 49)
    """

    def __init__(self, initial_audio=None):
        if initial_audio is None:
            initial_audio = np.zeros(
                CHUNK_SAMPLES,
                dtype=np.float32,
            )

        initial_audio = np.asarray(
            initial_audio,
            dtype=np.float32,
        )

        if initial_audio.ndim != 1:
            raise ValueError("Audio must be mono (1D).")

        if len(initial_audio) != CHUNK_SAMPLES:
            raise ValueError(
                f"Expected {CHUNK_SAMPLES} initial samples, "
                f"got {len(initial_audio)}."
            )

        if not np.isfinite(initial_audio).all():
            raise ValueError("Audio contains NaN or infinite values.")

        self.audio = initial_audio.copy()

        # Match librosa's default Hann window.
        self.window = get_window(
            "hann",
            N_FFT,
            fftbins=True,
        ).astype(np.float32)

        # Shape: (80, 201)
        self.mel_basis = librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            n_mels=N_MELS,
            dtype=np.float32,
        )

        # Initial calculation: all 49 frames.
        frame_starts = range(
            0,
            CHUNK_SAMPLES - N_FFT + 1,
            HOP_LENGTH,
        )

        self.mel_power = self._compute_frames(
            self.audio,
            frame_starts,
        )

    def _compute_frames(self, audio, frame_starts):
        """Calculate mel power for selected frame positions."""

        frame_starts = list(frame_starts)

        frames = np.stack([
            audio[start:start + N_FFT]
            for start in frame_starts
        ])

        # Apply the Hann window.
        windowed = frames * self.window[None, :]

        # Real FFT and power spectrum.
        spectrum = rfft(
            windowed,
            n=N_FFT,
            axis=-1,
        )

        power = np.abs(spectrum) ** 2

        # Convert frequency bins into mel bins.
        # Result shape: (80, number_of_frames)
        mel_power = self.mel_basis @ power.T

        return mel_power.astype(np.float32, copy=False)

    def update(self, new_samples):
        """
        Append exactly 640 new samples and return updated log-mel features.
        """

        new_samples = np.asarray(
            new_samples,
            dtype=np.float32,
        ).reshape(-1)

        expected_samples = 2 * HOP_LENGTH

        if len(new_samples) != expected_samples:
            raise ValueError(
                f"Expected {expected_samples} new samples "
                f"({expected_samples / SAMPLE_RATE:.3f} seconds), "
                f"got {len(new_samples)}."
            )

        if not np.isfinite(new_samples).all():
            raise ValueError("New audio contains NaN or infinite values.")

        # Join the old window and new audio.
        extended_audio = np.concatenate((
            self.audio,
            new_samples,
        ))

        shift = len(new_samples)

        # Calculate only the two newest frames.
        new_frame_starts = [
            shift + 47 * HOP_LENGTH,
            shift + 48 * HOP_LENGTH,
        ]

        new_mel_power = self._compute_frames(
            extended_audio,
            new_frame_starts,
        )

        # Drop the oldest two frames and append the new two.
        self.mel_power = np.concatenate((
            self.mel_power[:, 2:],
            new_mel_power,
        ), axis=1)

        # Keep the latest 1-second audio window.
        self.audio = extended_audio[shift:].copy()

        # Preserve original global-maximum dB normalization.
        log_mel = librosa.power_to_db(
            self.mel_power,
            ref=np.max,
        )

        return log_mel.astype(np.float32, copy=False)

# predict phoneme old

# def predict_latest_phoneme(audio):
#     log_mel = audio_to_log_mel(audio)

#     X = torch.from_numpy(log_mel).unsqueeze(0).to(DEVICE)

#     with torch.inference_mode():
#         logits = model(X)

#         if logits.ndim != 3 or logits.shape[-1] != NUM_CLASSES:
#             raise RuntimeError(
#                 f"Unexpected model output shape: {tuple(logits.shape)}"
#             )

#         latest_logits = logits[0, -1]
#         probabilities = torch.softmax(latest_logits, dim=-1)

#         confidence, prediction = probabilities.max(dim=-1)

#     phone = id_to_phone[prediction.item()]

#     return phone, confidence.item()


# predict phoneme new

def predict_latest_phoneme_from_features(log_mel):
    """
    Predict the latest phoneme from precomputed log-mel features.
    """

    log_mel = np.asarray(log_mel, dtype=np.float32)

    expected_shape = (N_MELS, 49)

    if log_mel.shape != expected_shape:
        raise ValueError(
            f"Expected features with shape {expected_shape}, "
            f"got {log_mel.shape}."
        )

    if not np.isfinite(log_mel).all():
        raise ValueError("Features contain NaN or infinite values.")

    X = torch.from_numpy(log_mel).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(X)

        if logits.ndim != 3 or logits.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected model output shape: {tuple(logits.shape)}"
            )

        if logits.shape[-1] != NUM_CLASSES:
            raise RuntimeError(
                f"Expected {NUM_CLASSES} output classes, "
                f"got {logits.shape[-1]}."
            )

        latest_logits = logits[0, -1]
        probabilities = torch.softmax(latest_logits, dim=-1)

        confidence, prediction = probabilities.max(dim=-1)

    phone = id_to_phone[prediction.item()]

    return phone, confidence.item()

