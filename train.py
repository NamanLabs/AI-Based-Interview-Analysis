# train.py
"""
Unified training script for all three modality models.

Usage:
    # Train all models
    python train.py --all

    # Train specific model
    python train.py --facial  --data data/raw/facial
    python train.py --speech  --data data/raw/speech/ravdess
    python train.py --nlp     --data data/raw/nlp/transcripts.csv
"""

import argparse
from loguru import logger
from pathlib import Path


def train_facial(data_dir: str):
    from utils.datasets import get_facial_loaders
    from models.facial_emotion_model import train_facial_model
    logger.info(f"Training Facial Emotion Model on: {data_dir}")
    train_loader, val_loader = get_facial_loaders(data_dir)
    model = train_facial_model(train_loader, val_loader)
    logger.info("✓ Facial model training complete.")
    return model


def train_speech(data_dir: str):
    from utils.datasets import get_speech_loaders
    from models.speech_emotion_model import train_speech_model
    logger.info(f"Training Speech Emotion Model on: {data_dir}")
    train_loader, val_loader = get_speech_loaders(data_dir)
    model = train_speech_model(train_loader, val_loader)
    logger.info("✓ Speech model training complete.")
    return model


def train_nlp(csv_path: str):
    from utils.datasets import get_nlp_loaders
    from models.nlp_model import train_nlp_model
    logger.info(f"Training NLP Sentiment Model on: {csv_path}")
    train_loader, val_loader = get_nlp_loaders(csv_path)
    model = train_nlp_model(train_loader, val_loader)
    logger.info("✓ NLP model training complete.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Interview Analyzer Models")
    parser.add_argument("--all",    action="store_true", help="Train all models")
    parser.add_argument("--facial", action="store_true", help="Train facial model")
    parser.add_argument("--speech", action="store_true", help="Train speech model")
    parser.add_argument("--nlp",    action="store_true", help="Train NLP model")
    parser.add_argument("--data",   type=str, default="data/raw", help="Data directory or CSV path")
    args = parser.parse_args()

    if args.all or args.facial:
        facial_data = str(Path(args.data) / "facial") if args.all else args.data
        if Path(facial_data).exists():
            train_facial(facial_data)
        else:
            logger.warning(f"Facial data not found at {facial_data}. Skipping.")

    if args.all or args.speech:
        speech_data = str(Path(args.data) / "speech") if args.all else args.data
        if Path(speech_data).exists():
            train_speech(speech_data)
        else:
            logger.warning(f"Speech data not found at {speech_data}. Skipping.")

    if args.all or args.nlp:
        nlp_data = str(Path(args.data) / "transcripts.csv") if args.all else args.data
        if Path(nlp_data).exists():
            train_nlp(nlp_data)
        else:
            logger.warning(f"NLP data not found at {nlp_data}. Skipping.")

    logger.info("All requested training jobs complete.")
