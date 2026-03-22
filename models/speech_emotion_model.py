# models/speech_emotion_model.py
"""
Speech Emotion Recognition using a CNN + Bi-GRU architecture.
Input: MFCC / Mel-spectrogram features from AudioProcessor
Output: 6-class emotion probabilities + prosodic communication scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, Tuple, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import speech_cfg, MODELS_DIR


class SpeechEmotionModel(nn.Module):
    """
    2D-CNN extracts local spectral patterns from Mel/MFCC features.
    Bi-GRU captures temporal dynamics.
    """

    def __init__(self, cfg=speech_cfg, input_channels: int = 1):
        super().__init__()
        self.cfg = cfg

        # ── CNN Feature Extractor ──────────────────────────────────────────
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),  # collapse frequency dim
        )

        # ── Bi-GRU Temporal Encoder ────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
            bidirectional=True
        )

        # ── Classification Head ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size * 2),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size * 2, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout / 2),
            nn.Linear(128, cfg.num_emotions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, freq, time) — mel spectrogram or MFCC
        Returns:
            logits: (B, num_emotions)
        """
        # CNN: (B,1,F,T) → (B,128,1,T') → (B,T',128)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.squeeze(2).permute(0, 2, 1)

        # GRU: (B,T',128) → (B,T',hidden*2)
        gru_out, _ = self.gru(cnn_out)

        # Use mean pooling over time
        pooled = gru_out.mean(dim=1)   # (B, hidden*2)

        return self.classifier(pooled)  # (B, num_emotions)


# ── High-Level Analyzer ───────────────────────────────────────────────────────

class SpeechEmotionAnalyzer:
    """
    Takes audio features from AudioProcessor, returns emotion predictions
    and communication quality scores.
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = SpeechEmotionModel().to(self.device)
        self.cfg    = speech_cfg

        if weights_path and Path(weights_path).exists():
            self.model.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
            logger.info(f"Speech model loaded from {weights_path}")
        else:
            logger.warning("No weights loaded — using random init (train first).")
        self.model.eval()

    def analyze(self, audio_processor) -> Dict:
        """
        Args:
            audio_processor: AudioProcessor instance
        Returns:
            dict with emotion prediction + communication scores
        """
        # Get feature matrix: (181, T)
        features = audio_processor.extract_all_features()  # (181, 300)

        # Use only mel component (rows 40:168) for the model, shape (128, 300)
        mel_features = features[40:168, :]
        tensor = torch.tensor(mel_features).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,128,300)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = F.softmax(logits, dim=-1).squeeze().cpu().tolist()

        emotion_probs = dict(zip(self.cfg.emotion_labels, probs))
        dominant_emotion = max(emotion_probs, key=emotion_probs.get)

        # Prosodic features for scoring
        prosodic = audio_processor.get_prosodic_features()

        # Communication score heuristics
        comm_score = self._compute_communication_score(emotion_probs, prosodic)

        return {
            "emotion_probabilities": {k: round(v, 4) for k, v in emotion_probs.items()},
            "dominant_emotion": dominant_emotion,
            "prosodic_features": prosodic,
            "communication_score": comm_score,
        }

    def _compute_communication_score(
        self, emotion_probs: Dict, prosodic: Dict
    ) -> float:
        """
        Heuristic scoring (0–100):
          - Positive emotional tone (Neutral + Calm + Happy → 40 pts)
          - Speaking rate in sweet spot (2–6 onsets/sec → 30 pts)
          - Pitch variability (non-monotone → 20 pts)
          - Low pause ratio (<30% → 10 pts)
        """
        score = 0.0

        # Tone
        positive = emotion_probs.get("Neutral", 0) + \
                   emotion_probs.get("Calm", 0) + \
                   emotion_probs.get("Happy", 0)
        score += min(40, positive * 40)

        # Speaking rate (2–6 syllables/sec is natural)
        rate = prosodic.get("speaking_rate", 0)
        if 2 <= rate <= 6:
            score += 30
        elif rate > 0:
            score += max(0, 30 - abs(rate - 4) * 5)

        # Pitch variation (higher std = more expressive)
        pitch_std = prosodic.get("pitch_std", 0)
        score += min(20, pitch_std / 10)

        # Pause ratio (less is better, up to 30%)
        pause = prosodic.get("pause_ratio", 1.0)
        score += max(0, 10 - pause * 30)

        return round(min(100, score), 1)


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_speech_model(train_loader, val_loader, cfg=speech_cfg):
    """
    Train on RAVDESS / CREMA-D / TESS datasets.
    
    Expected DataLoader output: (mel_tensor, label)
      - mel_tensor: (B, 1, 128, T)
      - label: (B,) int class index
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SpeechEmotionModel(cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = nn.CrossEntropyLoss()
    save_path = MODELS_DIR / "speech_emotion_best.pth"
    best_val_acc = 0.0

    for epoch in range(cfg.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)
        train_acc = correct / total

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                preds = model(features).argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total   += labels.size(0)
        val_acc = v_correct / v_total
        scheduler.step(1 - val_acc)

        logger.info(f"[Speech] Epoch {epoch+1}/{cfg.epochs} | "
                    f"Loss: {total_loss/len(train_loader):.4f} | "
                    f"Train: {train_acc:.3f} | Val: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ Best model saved ({val_acc:.3f})")

    return model
