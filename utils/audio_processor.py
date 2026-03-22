# utils/audio_processor.py
"""
Audio preprocessing pipeline:
  - Load & normalize audio
  - Extract MFCC, Mel-spectrogram, Chroma, ZCR features
  - Speech-to-text transcription via Whisper (or fallback SpeechRecognition)
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from loguru import logger
from typing import Dict, Optional, Tuple
import torch

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import audio_cfg


class AudioProcessor:
    """Feature extraction and transcription from interview audio."""

    def __init__(self, audio_path: str):
        self.audio_path = Path(audio_path)
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        self.waveform, self.sr = librosa.load(
            str(self.audio_path), sr=audio_cfg.sample_rate, mono=True
        )
        # Trim silence at start/end
        self.waveform, _ = librosa.effects.trim(self.waveform, top_db=20)
        logger.info(f"Audio loaded: {len(self.waveform)/self.sr:.1f}s @ {self.sr}Hz")

    # ── Feature Extraction ────────────────────────────────────────────────────

    def extract_mfcc(self) -> np.ndarray:
        """Returns MFCC matrix of shape (n_mfcc, T)."""
        mfcc = librosa.feature.mfcc(
            y=self.waveform,
            sr=self.sr,
            n_mfcc=audio_cfg.n_mfcc,
            hop_length=audio_cfg.hop_length
        )
        return librosa.util.normalize(mfcc)

    def extract_mel_spectrogram(self) -> np.ndarray:
        """Returns log-mel spectrogram of shape (n_mels, T)."""
        mel = librosa.feature.melspectrogram(
            y=self.waveform,
            sr=self.sr,
            n_mels=audio_cfg.n_mels,
            hop_length=audio_cfg.hop_length
        )
        return librosa.power_to_db(mel, ref=np.max)

    def extract_chroma(self) -> np.ndarray:
        """Chroma features — captures tonal content."""
        return librosa.feature.chroma_stft(
            y=self.waveform, sr=self.sr, hop_length=audio_cfg.hop_length
        )

    def extract_zcr(self) -> np.ndarray:
        """Zero-crossing rate — proxy for speech activity."""
        return librosa.feature.zero_crossing_rate(
            self.waveform, hop_length=audio_cfg.hop_length
        )

    def extract_all_features(self, fixed_length: int = 300) -> np.ndarray:
        """
        Concatenate all features into a single matrix of shape (features, fixed_length).
        Pads or truncates time dimension to fixed_length.
        """
        mfcc    = self.extract_mfcc()           # (40, T)
        mel     = self.extract_mel_spectrogram() # (128, T)
        chroma  = self.extract_chroma()          # (12, T)
        zcr     = self.extract_zcr()             # (1, T)

        features = np.vstack([mfcc, mel, chroma, zcr])  # (181, T)

        # Pad or truncate
        T = features.shape[1]
        if T < fixed_length:
            features = np.pad(features, ((0, 0), (0, fixed_length - T)), mode="constant")
        else:
            features = features[:, :fixed_length]

        return features.astype(np.float32)

    def get_prosodic_features(self) -> Dict[str, float]:
        """
        Scalar prosodic features used for scoring:
          - speaking_rate, pitch_mean, pitch_std, energy_mean, pause_ratio
        """
        # Pitch (F0)
        f0, voiced_flag, _ = librosa.pyin(
            self.waveform, fmin=80, fmax=400, sr=self.sr
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
        pitch_mean = float(np.nanmean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        pitch_std  = float(np.nanstd(f0_voiced))  if len(f0_voiced) > 0 else 0.0

        # Energy
        rms = librosa.feature.rms(y=self.waveform)[0]
        energy_mean = float(np.mean(rms))

        # Pause ratio (frames below energy threshold)
        silence_threshold = 0.01
        pause_ratio = float(np.mean(rms < silence_threshold))

        # Speaking rate (rough estimate via onset detection)
        onsets = librosa.onset.onset_detect(y=self.waveform, sr=self.sr)
        speaking_rate = len(onsets) / (len(self.waveform) / self.sr)  # onsets/sec

        return {
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "energy_mean": energy_mean,
            "pause_ratio": pause_ratio,
            "speaking_rate": speaking_rate
        }

    # ── Speech-to-Text ────────────────────────────────────────────────────────

    def transcribe(self, use_whisper: bool = True) -> str:
        """
        Transcribe audio to text.
        Tries OpenAI Whisper first (best quality), falls back to SpeechRecognition.
        """
        if use_whisper:
            try:
                import whisper
                model = whisper.load_model("base")
                result = model.transcribe(str(self.audio_path))
                text = result["text"].strip()
                logger.info(f"Whisper transcription: {len(text)} chars")
                return text
            except ImportError:
                logger.warning("Whisper not installed. Falling back to SpeechRecognition.")

        # Fallback: Google SpeechRecognition
        try:
            import speech_recognition as sr_lib
            recognizer = sr_lib.Recognizer()
            with sr_lib.AudioFile(str(self.audio_path)) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            logger.info(f"SpeechRecognition transcription: {len(text)} chars")
            return text
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
