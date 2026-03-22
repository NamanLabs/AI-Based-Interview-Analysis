# models/nlp_model.py
"""
NLP Analysis using BERT for:
  1. Sentiment analysis (positive / neutral / negative communication tone)
  2. Communication quality scoring (vocabulary richness, coherence proxy)
  3. Keyword & competency extraction (leadership, problem-solving, teamwork, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from collections import Counter
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional
from transformers import (
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
    pipeline
)

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import nlp_cfg, MODELS_DIR


# ── BERT Sentiment Classifier ─────────────────────────────────────────────────

class BERTSentimentModel(nn.Module):
    """
    Fine-tuned BERT for 3-class sentiment:
      0 = Negative, 1 = Neutral, 2 = Positive
    """
    LABELS = ["Negative", "Neutral", "Positive"]

    def __init__(self, cfg=nlp_cfg):
        super().__init__()
        self.bert = BertModel.from_pretrained(cfg.model_name)
        hidden = self.bert.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        return self.classifier(cls_output)


# ── NLP Analyzer ─────────────────────────────────────────────────────────────

class NLPAnalyzer:
    """
    Main NLP analysis pipeline for interview transcripts.
    Uses BERT for sentiment + heuristic scoring for communication quality.
    """

    # Competency keyword banks
    COMPETENCY_KEYWORDS = {
        "Leadership":        ["led", "managed", "directed", "supervised", "coordinated", "mentored", "motivated"],
        "Problem Solving":   ["solved", "analyzed", "identified", "resolved", "debugged", "optimized", "improved"],
        "Teamwork":          ["collaborated", "team", "together", "supported", "helped", "partnered", "cooperated"],
        "Communication":     ["presented", "explained", "communicated", "reported", "documented", "articulated"],
        "Technical Skills":  ["implemented", "developed", "built", "designed", "programmed", "deployed", "tested"],
        "Adaptability":      ["adapted", "flexible", "adjusted", "learned", "changed", "pivoted", "handled"],
        "Initiative":        ["initiated", "proposed", "started", "created", "launched", "suggested", "innovated"],
    }

    # Filler words to penalize
    FILLER_WORDS = {"um", "uh", "like", "you know", "basically", "literally", "kinda", "sorta"}

    def __init__(self, weights_path: Optional[str] = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg    = nlp_cfg

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(nlp_cfg.model_name)

        # Sentiment model
        self.sentiment_model = BERTSentimentModel(nlp_cfg).to(self.device)
        if weights_path and Path(weights_path).exists():
            self.sentiment_model.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
            logger.info(f"NLP model loaded from {weights_path}")
        else:
            logger.warning("No NLP weights loaded — using pretrained BERT features.")
        self.sentiment_model.eval()

    def analyze(self, transcript: str) -> Dict:
        """
        Full NLP analysis of a transcript string.
        Returns sentiment, communication scores, and competency flags.
        """
        if not transcript or len(transcript.strip()) < 10:
            return {"error": "Transcript too short or empty."}

        transcript = transcript.strip()
        sentences  = self._split_sentences(transcript)

        # Per-sentence sentiment
        sentiments = [self._predict_sentiment(s) for s in sentences if len(s) > 5]

        # Aggregate sentiment
        agg_sentiment = self._aggregate_sentiments(sentiments)

        # Competency detection
        competencies = self._detect_competencies(transcript)

        # Communication quality metrics
        quality = self._communication_quality(transcript, sentences)

        # Overall NLP score (0-100)
        nlp_score = self._compute_nlp_score(agg_sentiment, quality, competencies)

        return {
            "transcript_length": len(transcript.split()),
            "sentence_count": len(sentences),
            "sentiment": agg_sentiment,
            "competencies_detected": competencies,
            "communication_quality": quality,
            "nlp_score": nlp_score
        }

    def _predict_sentiment(self, text: str) -> Dict:
        """Predict sentiment for a single sentence."""
        encoding = self.tokenizer(
            text,
            max_length=self.cfg.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.sentiment_model(input_ids, attention_mask)
            probs  = F.softmax(logits, dim=-1).squeeze().cpu().tolist()

        label = BERTSentimentModel.LABELS[np.argmax(probs)]
        return {
            "label": label,
            "probabilities": dict(zip(BERTSentimentModel.LABELS, probs))
        }

    def _aggregate_sentiments(self, sentiments: List[Dict]) -> Dict:
        if not sentiments:
            return {"dominant": "Neutral", "distribution": {}}
        counter = Counter(s["label"] for s in sentiments)
        total   = len(sentiments)
        dist    = {k: round(v / total, 3) for k, v in counter.items()}
        return {
            "dominant": counter.most_common(1)[0][0],
            "distribution": dist,
            "positive_ratio": round(dist.get("Positive", 0), 3)
        }

    def _detect_competencies(self, text: str) -> Dict[str, bool]:
        text_lower = text.lower()
        return {
            comp: any(kw in text_lower for kw in keywords)
            for comp, keywords in self.COMPETENCY_KEYWORDS.items()
        }

    def _communication_quality(self, text: str, sentences: List[str]) -> Dict:
        words = text.lower().split()
        total_words  = len(words)
        unique_words = len(set(words))

        # Type-token ratio (vocabulary richness)
        ttr = unique_words / total_words if total_words > 0 else 0

        # Avg sentence length
        avg_sent_len = total_words / len(sentences) if sentences else 0

        # Filler word count
        filler_count = sum(text.lower().count(fw) for fw in self.FILLER_WORDS)
        filler_ratio = filler_count / total_words if total_words > 0 else 0

        return {
            "total_words": total_words,
            "unique_words": unique_words,
            "vocabulary_richness": round(ttr, 3),
            "avg_sentence_length": round(avg_sent_len, 1),
            "filler_word_count": filler_count,
            "filler_ratio": round(filler_ratio, 3)
        }

    def _compute_nlp_score(
        self, sentiment: Dict, quality: Dict, competencies: Dict
    ) -> float:
        score = 0.0

        # 1. Positive sentiment tone (30 pts)
        pos_ratio = sentiment.get("positive_ratio", 0)
        score += min(30, pos_ratio * 50)

        # 2. Vocabulary richness (20 pts) — ideal TTR ~0.4–0.7
        ttr = quality.get("vocabulary_richness", 0)
        score += min(20, ttr * 30)

        # 3. Competencies demonstrated (30 pts — up to 7 competencies)
        comp_count = sum(1 for v in competencies.values() if v)
        score += min(30, comp_count * 5)

        # 4. Low filler ratio (10 pts)
        filler = quality.get("filler_ratio", 1)
        score += max(0, 10 - filler * 50)

        # 5. Sentence length (10 pts) — ideal 10–20 words
        avg_len = quality.get("avg_sentence_length", 0)
        if 10 <= avg_len <= 20:
            score += 10
        elif avg_len > 0:
            score += max(0, 10 - abs(avg_len - 15) * 0.5)

        return round(min(100, score), 1)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


# ── Training Helper ────────────────────────────────────────────────────────────

def train_nlp_model(train_loader, val_loader, cfg=nlp_cfg):
    """
    Fine-tune BERT for sentiment. Use SST-2 or a custom interview dataset.
    DataLoader yields: (input_ids, attention_mask, labels)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = BERTSentimentModel(cfg).to(device)

    # Freeze all BERT layers except last 4
    for name, param in model.bert.named_parameters():
        if not any(f"layer.{i}" in name for i in range(8, 12)):
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5, weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()
    save_path = MODELS_DIR / "nlp_sentiment_best.pth"
    best_val_acc = 0.0

    for epoch in range(cfg.epochs if hasattr(cfg, 'epochs') else 10):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = (
                input_ids.to(device), attention_mask.to(device), labels.to(device)
            )
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        val_correct, val_total = 0, 0
        model.eval()
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids, attention_mask, labels = (
                    input_ids.to(device), attention_mask.to(device), labels.to(device)
                )
                preds = model(input_ids, attention_mask).argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
        val_acc = val_correct / val_total

        logger.info(f"[NLP] Epoch {epoch+1} | "
                    f"Loss: {total_loss/len(train_loader):.4f} | "
                    f"Val Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    return model
