# models/facial_emotion_model.py
"""
Facial Emotion Recognition using EfficientNet-B0 (or ViT backbone).
Trained on FER-2013 / AffectNet style data.

Architecture:
  - Pretrained CNN backbone (timm)
  - Global Average Pooling
  - Dropout + Dense head → 7-class softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2
from pathlib import Path
from loguru import logger
from typing import List, Dict, Tuple, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import facial_cfg, MODELS_DIR


class FacialEmotionModel(nn.Module):
    """EfficientNet-B0 backbone for 7-class facial emotion classification."""

    def __init__(self, cfg=facial_cfg):
        super().__init__()
        self.cfg = cfg

        # Load pretrained backbone (no classifier head)
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0  # remove original head
        )
        feat_dim = self.backbone.num_features

        # Custom classification head
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(cfg.dropout / 2),
            nn.Linear(256, cfg.num_emotions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalized face images
        Returns:
            logits: (B, num_emotions)
        """
        features = self.backbone(x)  # (B, feat_dim)
        return self.head(features)   # (B, num_emotions)

    def predict_emotion(self, x: torch.Tensor) -> Tuple[List[str], List[float]]:
        """Returns (predicted_labels, probabilities) for a batch."""
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs  = F.softmax(logits, dim=-1)
            preds  = probs.argmax(dim=-1)
        labels = [self.cfg.emotion_labels[i] for i in preds.cpu().tolist()]
        prob_list = probs.cpu().tolist()
        return labels, prob_list


# ── Inference Helper ──────────────────────────────────────────────────────────

class FacialEmotionAnalyzer:
    """
    High-level interface: takes a list of face crops (BGR numpy arrays),
    returns per-frame emotions and an aggregated summary.
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, weights_path: Optional[str] = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = FacialEmotionModel().to(self.device)

        if weights_path and Path(weights_path).exists():
            self.model.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
            logger.info(f"Facial model loaded from {weights_path}")
        else:
            logger.warning("No weights loaded — using random init (train first).")

        self.model.eval()

    def _preprocess(self, face_bgr: np.ndarray) -> torch.Tensor:
        """BGR numpy (H,W,3) → normalized tensor (3,H,W)."""
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.MEAN) / self.STD
        return torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)

    def analyze_sequence(
        self, face_crops: List[np.ndarray], batch_size: int = 16
    ) -> Dict:
        """
        Args:
            face_crops: list of BGR face images (already resized to face_size)
        Returns:
            dict with per_frame results + aggregate emotion distribution
        """
        if not face_crops:
            return {"error": "No face crops provided"}

        tensors = [self._preprocess(f) for f in face_crops]
        per_frame = []

        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i+batch_size]).to(self.device)
            labels, probs = self.model.predict_emotion(batch)
            for lbl, prob in zip(labels, probs):
                per_frame.append({
                    "emotion": lbl,
                    "probabilities": dict(zip(facial_cfg.emotion_labels, prob))
                })

        # Aggregate: mean probability per emotion across all frames
        agg = {e: 0.0 for e in facial_cfg.emotion_labels}
        for frame in per_frame:
            for emotion, p in frame["probabilities"].items():
                agg[emotion] += p
        total = len(per_frame)
        agg = {k: round(v / total, 4) for k, v in agg.items()}

        # Confidence score: % of frames showing positive emotions (Happy, Neutral)
        positive_ratio = sum(
            1 for f in per_frame if f["emotion"] in ["Happy", "Neutral"]
        ) / total

        return {
            "per_frame": per_frame,
            "aggregate": agg,
            "dominant_emotion": max(agg, key=agg.get),
            "emotional_stability_score": round(positive_ratio * 100, 1),
            "total_frames_analyzed": total
        }


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_facial_model(train_loader, val_loader, cfg=facial_cfg):
    """
    Standard training loop. Call with PyTorch DataLoaders for FER-2013/AffectNet.
    
    Dataset structure expected:
        data/raw/facial/train/<emotion_label>/<image>.jpg
        data/raw/facial/val/<emotion_label>/<image>.jpg
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = FacialEmotionModel(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    save_path    = MODELS_DIR / "facial_emotion_best.pth"

    for epoch in range(cfg.epochs):
        # ── Train ──
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)
        train_acc = correct / total

        # ── Validate ──
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total   += labels.size(0)
        val_acc = v_correct / v_total

        scheduler.step()
        logger.info(f"[Facial] Epoch {epoch+1}/{cfg.epochs} | "
                    f"Loss: {total_loss/len(train_loader):.4f} | "
                    f"Train: {train_acc:.3f} | Val: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ Best model saved ({val_acc:.3f})")

    return model
