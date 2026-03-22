# models/fusion_model.py
"""
Multimodal Fusion Layer.

Takes embeddings from all three modalities and produces:
  - Attention-weighted fusion
  - Final behavioral prediction scores
  - Suitability score (0-100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, Optional, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import fusion_cfg, score_cfg, MODELS_DIR


class CrossModalAttention(nn.Module):
    """
    Learns pairwise attention weights between modality pairs.
    This allows each modality to attend to relevant info from others.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm  = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query, key_value, key_value)
        return self.norm(query + attn_out)


class MultimodalFusionModel(nn.Module):
    """
    Fuses three modality embeddings via cross-modal attention + MLP.

    Inputs:
      - facial_emb:  (B, 256)  from FacialEmotionModel
      - speech_emb:  (B, 512)  from SpeechEmotionModel (hidden*2)
      - text_emb:    (B, 768)  from BERT [CLS]

    Output:
      - suitability logit (B, 1) mapped to 0-100
    """

    def __init__(self, cfg=fusion_cfg,
                 facial_dim: int = 256,
                 speech_dim: int = 512,
                 text_dim: int = 768):
        super().__init__()
        hidden = cfg.hidden_dim  # 512

        # Project each modality to same dimension
        self.facial_proj = nn.Linear(facial_dim, hidden)
        self.speech_proj = nn.Linear(speech_dim, hidden)
        self.text_proj   = nn.Linear(text_dim,   hidden)

        # Cross-modal attention (each attends to others)
        self.fa_sa = CrossModalAttention(hidden)   # facial ← speech
        self.fa_ta = CrossModalAttention(hidden)   # facial ← text
        self.sa_ta = CrossModalAttention(hidden)   # speech ← text
        self.ta_fa = CrossModalAttention(hidden)   # text   ← facial

        # Gating network: learn per-modality importance
        self.gate = nn.Sequential(
            nn.Linear(hidden * 3, 3),
            nn.Softmax(dim=-1)
        )

        # Final MLP head
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout / 2),
            nn.Linear(128, 1),
            nn.Sigmoid()   # output in [0,1], multiply by 100 for score
        )

    def forward(
        self,
        facial_emb: torch.Tensor,
        speech_emb: torch.Tensor,
        text_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            score (B,1) in [0,1]
            gates (B,3) — modality importance weights
        """
        # Project to common dim — add sequence dim for attention
        f = self.facial_proj(facial_emb).unsqueeze(1)   # (B,1,H)
        s = self.speech_proj(speech_emb).unsqueeze(1)
        t = self.text_proj(text_emb).unsqueeze(1)

        # Cross-modal attention
        f = self.fa_sa(f, s) + self.fa_ta(f, t)
        s = self.sa_ta(s, t)
        t = self.ta_fa(t, f)

        f = f.squeeze(1)   # (B,H)
        s = s.squeeze(1)
        t = t.squeeze(1)

        # Learned gating
        concat = torch.cat([f, s, t], dim=-1)   # (B, H*3)
        gates  = self.gate(concat)               # (B, 3)
        fused  = gates[:, 0:1] * f + gates[:, 1:2] * s + gates[:, 2:3] * t
        fused  = torch.cat([f, s, t], dim=-1)    # also keep full concat for head

        score = self.head(fused)
        return score, gates


# ── Rule-based Scorer (no training required) ──────────────────────────────────

class InterviewScorer:
    """
    Weighted combination of modality scores when fusion model is not trained.
    Produces the 4 required output scores.
    """

    def __init__(self, cfg=score_cfg):
        self.cfg = cfg

    def compute_scores(
        self,
        facial_result: Dict,
        speech_result: Dict,
        nlp_result: Dict
    ) -> Dict:
        """
        Combines all analysis results into 4 final interview scores.
        """
        # ── Emotional Stability (facial + speech) ──────────────────────────
        facial_stability = facial_result.get("emotional_stability_score", 50.0)
        speech_emotion   = speech_result.get("emotion_probabilities", {})
        positive_speech  = (
            speech_emotion.get("Neutral", 0) +
            speech_emotion.get("Calm", 0) +
            speech_emotion.get("Happy", 0)
        ) * 100
        emotional_stability = round(
            0.6 * facial_stability + 0.4 * positive_speech, 1
        )

        # ── Communication Skill (speech + NLP) ────────────────────────────
        speech_comm = speech_result.get("communication_score", 50.0)
        nlp_score   = nlp_result.get("nlp_score", 50.0)
        communication_skill = round(
            0.4 * speech_comm + 0.6 * nlp_score, 1
        )

        # ── Confidence Score (facial + speech) ────────────────────────────
        # Confidence inferred from low anxiety + good energy + low pauses
        facial_agg = facial_result.get("aggregate", {})
        low_anxiety = 100 - (
            facial_agg.get("Angry", 0) +
            facial_agg.get("Fear", 0) +
            facial_agg.get("Sad", 0)
        ) * 100

        prosodic    = speech_result.get("prosodic_features", {})
        energy      = min(100, prosodic.get("energy_mean", 0.05) * 1000)
        pause_penalty = prosodic.get("pause_ratio", 0.5) * 50
        confidence_score = round(
            0.4 * low_anxiety + 0.4 * energy - pause_penalty + 20, 1  # +20 baseline
        )
        confidence_score = max(0, min(100, confidence_score))

        # ── Overall Suitability ────────────────────────────────────────────
        overall_score = round(
            self.cfg.facial_weight * emotional_stability +
            self.cfg.speech_weight * communication_skill +
            self.cfg.nlp_weight    * nlp_score,
            1
        )

        # ── Competency Summary ─────────────────────────────────────────────
        competencies = nlp_result.get("competencies_detected", {})
        strong_competencies = [k for k, v in competencies.items() if v]

        return {
            "confidence_score":        confidence_score,
            "communication_skill":     communication_skill,
            "emotional_stability":     emotional_stability,
            "overall_suitability":     overall_score,
            "strong_competencies":     strong_competencies,
            "modality_scores": {
                "facial_analysis":  round(facial_stability, 1),
                "speech_analysis":  round(speech_comm, 1),
                "language_analysis": round(nlp_score, 1)
            },
            "recommendation": _get_recommendation(overall_score)
        }


def _get_recommendation(score: float) -> str:
    if score >= 80:
        return "Strong Candidate — Highly Recommended"
    elif score >= 65:
        return "Good Candidate — Recommended with Minor Reservations"
    elif score >= 50:
        return "Average Candidate — Further Assessment Suggested"
    else:
        return "Below Average — Additional Screening Required"
