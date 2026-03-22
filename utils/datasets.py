# utils/datasets.py
"""
PyTorch Dataset classes for training all three modality models.

Supported Datasets:
  - FacialDataset:  FER-2013 / AffectNet structure
  - SpeechDataset:  RAVDESS / CREMA-D / TESS
  - NLPDataset:     Custom CSV of (text, sentiment_label)
"""

import os
import torch
import numpy as np
import pandas as pd
import librosa
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from loguru import logger
from typing import Tuple, Optional, List

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import facial_cfg, audio_cfg, nlp_cfg
from transformers import BertTokenizer


# ── Facial Dataset ─────────────────────────────────────────────────────────────

class FacialDataset(Dataset):
    """
    Expects directory structure:
      root/
        angry/   *.jpg
        happy/   *.jpg
        ...
    
    Compatible with FER-2013 / AffectNet folder layout.
    Download FER-2013: https://www.kaggle.com/datasets/msambare/fer2013
    """

    TRANSFORMS = {
        "train": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    def __init__(self, root_dir: str, split: str = "train"):
        self.root = Path(root_dir)
        self.split = split
        self.transform = self.TRANSFORMS[split]
        self.samples: List[Tuple[Path, int]] = []

        label_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        self.label_map = {d.name.lower(): i for i, d in enumerate(label_dirs)}

        for label_dir in label_dirs:
            label_idx = self.label_map[label_dir.name.lower()]
            for img_path in label_dir.glob("*.[jp][pn]g"):
                self.samples.append((img_path, label_idx))

        logger.info(f"FacialDataset [{split}]: {len(self.samples)} samples, "
                    f"{len(self.label_map)} classes: {list(self.label_map.keys())}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, label


# ── Speech Dataset ──────────────────────────────────────────────────────────

class SpeechDataset(Dataset):
    """
    Compatible with RAVDESS naming convention:
      03-01-01-01-01-01-01.wav
      Modality-Vocal-Emotion-Intensity-Statement-Repetition-Actor.wav
    
    Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful
    
    Download RAVDESS: https://zenodo.org/record/1188976
    Also compatible with CREMA-D / TESS with minor label_fn adjustments.
    """

    RAVDESS_EMOTION_MAP = {
        "01": 0,  # neutral
        "02": 1,  # calm
        "03": 2,  # happy
        "04": 3,  # sad
        "05": 4,  # angry
        "06": 5,  # fearful
    }

    def __init__(self, root_dir: str, fixed_length: int = 300):
        self.root = Path(root_dir)
        self.fixed_length = fixed_length
        self.samples: List[Tuple[Path, int]] = []

        for wav_path in self.root.rglob("*.wav"):
            label = self._get_label(wav_path)
            if label is not None:
                self.samples.append((wav_path, label))

        logger.info(f"SpeechDataset: {len(self.samples)} audio files loaded.")

    def _get_label(self, wav_path: Path) -> Optional[int]:
        """Parse emotion label from RAVDESS filename."""
        try:
            parts = wav_path.stem.split("-")
            emotion_code = parts[2]
            return self.RAVDESS_EMOTION_MAP.get(emotion_code)
        except (IndexError, KeyError):
            return None

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]
        waveform, sr = librosa.load(str(wav_path), sr=audio_cfg.sample_rate, mono=True)

        mel = librosa.feature.melspectrogram(
            y=waveform, sr=sr,
            n_mels=audio_cfg.n_mels,
            hop_length=audio_cfg.hop_length
        )
        mel = librosa.power_to_db(mel, ref=np.max)

        # Pad/truncate to fixed_length
        T = mel.shape[1]
        if T < self.fixed_length:
            mel = np.pad(mel, ((0,0),(0, self.fixed_length - T)), mode="constant")
        else:
            mel = mel[:, :self.fixed_length]

        tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, 128, T)
        return tensor, label


# ── NLP Dataset ──────────────────────────────────────────────────────────────

class NLPDataset(Dataset):
    """
    CSV format: text,label
    label: 0=Negative, 1=Neutral, 2=Positive
    
    You can build this from SST-2 or custom interview data.
    """

    def __init__(self, csv_path: str, max_length: int = None):
        self.df         = pd.read_csv(csv_path)
        self.max_length = max_length or nlp_cfg.max_length
        self.tokenizer  = BertTokenizer.from_pretrained(nlp_cfg.model_name)

        assert "text" in self.df.columns and "label" in self.df.columns, \
            "CSV must have 'text' and 'label' columns"
        logger.info(f"NLPDataset: {len(self.df)} samples from {csv_path}")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        enc  = self.tokenizer(
            str(row["text"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return (
            enc["input_ids"].squeeze(),
            enc["attention_mask"].squeeze(),
            torch.tensor(int(row["label"]), dtype=torch.long)
        )


# ── DataLoader Factory ────────────────────────────────────────────────────────

def get_facial_loaders(root: str, batch_size: int = 32, val_split: float = 0.2):
    train_ds = FacialDataset(root, "train")
    val_ds   = FacialDataset(root, "val")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    )


def get_speech_loaders(root: str, batch_size: int = 32, val_split: float = 0.2):
    full_ds = SpeechDataset(root)
    val_size   = int(len(full_ds) * val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    )


def get_nlp_loaders(csv_path: str, batch_size: int = 16, val_split: float = 0.2):
    full_ds = NLPDataset(csv_path)
    val_size   = int(len(full_ds) * val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    )
