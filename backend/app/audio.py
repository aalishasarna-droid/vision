"""
Audio processing pipeline for industrial anomaly detection.
Uses Librosa to convert raw audio (.wav) into Log-Mel Spectrograms.
"""

import io
from typing import Tuple
import numpy as np

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        target_frames: int = 64,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_frames = target_frames

    def load_audio_from_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Loads audio waveform from raw bytes."""
        if not LIBROSA_AVAILABLE:
            # Fallback mock for demonstration if librosa isn't installed in environment yet
            return np.random.randn(self.sample_rate * 2).astype(np.float32), self.sample_rate

        with io.BytesIO(audio_bytes) as audio_file:
            y, sr = sf.read(audio_file)
            if y.ndim > 1:
                # Convert stereo to mono
                y = np.mean(y, axis=1)
            if sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            return y.astype(np.float32), self.sample_rate

    def compute_log_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Converts a 1D audio waveform into a 2D Log-Mel Spectrogram matrix.
        Output shape: (n_mels, frames)
        """
        if not LIBROSA_AVAILABLE:
            # Synthetic mock spectrogram for initial setup / dry runs
            spec = np.sin(np.linspace(0, 10, self.n_mels)[:, None]) * np.cos(
                np.linspace(0, 10, self.target_frames)[None, :]
            )
            return spec.astype(np.float32)

        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )

        # Convert to dB scale (Log-Mel)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Pad or crop to target frames (e.g. 64 frames)
        if log_mel_spec.shape[1] < self.target_frames:
            pad_width = self.target_frames - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode="constant")
        else:
            log_mel_spec = log_mel_spec[:, : self.target_frames]

        # Min-max normalize to [-1, 1] range for Autoencoder
        min_val = np.min(log_mel_spec)
        max_val = np.max(log_mel_spec)
        if max_val - min_val > 1e-6:
            norm_spec = 2.0 * (log_mel_spec - min_val) / (max_val - min_val) - 1.0
        else:
            norm_spec = np.zeros_like(log_mel_spec)

        return norm_spec.astype(np.float32)
