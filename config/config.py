# config/config.py
"""
Central configuration for the Multimodal Interview Analyzer.
Adjust these values to tune model behavior and paths.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Project Root ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)


# ── Video / Frame Settings ────────────────────────────────────────────────────
@dataclass
class VideoConfig:
    frame_rate: int = 5           # frames per second to sample
    face_size: tuple = (224, 224) # resize to this for CNN/ViT
    max_duration_sec: int = 300   # max interview clip length (5 min)


# ── Audio Settings ────────────────────────────────────────────────────────────
@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_mfcc: int = 40
    n_mels: int = 128
    hop_length: int = 512
    max_audio_len_sec: int = 300


# ── NLP / BERT Settings ───────────────────────────────────────────────────────
@dataclass
class NLPConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    batch_size: int = 8


# ── Facial Emotion Model ──────────────────────────────────────────────────────
@dataclass
class FacialConfig:
    backbone: str = "efficientnet_b0"   # or "vit_small_patch16_224"
    num_emotions: int = 7               # angry, disgust, fear, happy, neutral, sad, surprise
    emotion_labels: list = field(default_factory=lambda: [
        "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"
    ])
    pretrained: bool = True
    dropout: float = 0.3
    learning_rate: float = 1e-4
    epochs: int = 30
    batch_size: int = 32


# ── Speech Emotion Model ──────────────────────────────────────────────────────
@dataclass
class SpeechConfig:
    num_emotions: int = 6    # neutral, calm, happy, sad, angry, fearful
    emotion_labels: list = field(default_factory=lambda: [
        "Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful"
    ])
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.3
    learning_rate: float = 1e-3
    epochs: int = 40
    batch_size: int = 32


# ── Fusion / Output ───────────────────────────────────────────────────────────
@dataclass
class FusionConfig:
    fusion_method: str = "attention"   # "concat" | "attention" | "weighted"
    hidden_dim: int = 512
    dropout: float = 0.4
    learning_rate: float = 5e-5
    epochs: int = 20


# ── Score Weights (must sum to 1.0) ──────────────────────────────────────────
@dataclass
class ScoringConfig:
    facial_weight: float = 0.30
    speech_weight: float = 0.30
    nlp_weight: float = 0.40


# ── Instantiate Configs ───────────────────────────────────────────────────────
video_cfg  = VideoConfig()
audio_cfg  = AudioConfig()
nlp_cfg    = NLPConfig()
facial_cfg = FacialConfig()
speech_cfg = SpeechConfig()
fusion_cfg = FusionConfig()
score_cfg  = ScoringConfig()
