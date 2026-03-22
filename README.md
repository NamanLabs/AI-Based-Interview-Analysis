# 🎙️ Multimodal Deep Learning Framework for AI-Based Interview Analysis

**Naman Kumar | Reg: 23FE10CAI00331 | MUJ Deep Learning Lab (AIM3230)**  
Guided by Dr. Pankaj Upadhyay

---

## 📁 Project Structure

```
interview_analyzer/
├── config/
│   └── config.py              ← All hyperparameters & paths
├── models/
│   ├── facial_emotion_model.py ← EfficientNet-B0 / ViT → 7 emotions
│   ├── speech_emotion_model.py ← CNN + Bi-GRU → 6 speech emotions
│   ├── nlp_model.py            ← BERT sentiment + competency detection
│   └── fusion_model.py         ← Cross-modal attention + scoring
├── utils/
│   ├── video_processor.py     ← Frame extraction, face detection
│   ├── audio_processor.py     ← MFCC/Mel features, transcription
│   └── datasets.py            ← Dataset classes (FER/RAVDESS/CSV)
├── templates/
│   └── index.html             ← Web UI dashboard
├── data/
│   ├── raw/                   ← Your training datasets go here
│   └── processed/             ← Preprocessed cached features
├── saved_models/              ← Trained .pth weight files (auto-created)
├── pipeline.py                ← End-to-end inference orchestrator
├── app.py                     ← Flask REST API
├── train.py                   ← Training CLI
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
cd interview_analyzer
pip install -r requirements.txt

# Install ffmpeg (required for audio extraction)
# Ubuntu/Debian:
sudo apt install ffmpeg
# macOS:
brew install ffmpeg

# Optional: Install Whisper for best transcription quality
pip install openai-whisper
```

### 2. Run the Web App (without training — uses heuristic scoring)

```bash
python app.py
# Open http://localhost:5000
```

### 3. Command-Line Analysis

```bash
python pipeline.py path/to/interview.mp4 --save
```

---

## 🏋️ Training the Models

### Step 1: Get Datasets

| Model | Dataset | Download |
|-------|---------|----------|
| Facial | FER-2013 | [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) |
| Speech | RAVDESS | [Zenodo](https://zenodo.org/record/1188976) |
| NLP | SST-2 / Custom | [HuggingFace](https://huggingface.co/datasets/sst2) |

### Step 2: Organize Data

```
data/raw/
├── facial/
│   ├── angry/  *.jpg
│   ├── happy/  *.jpg
│   └── ...
├── speech/
│   └── Actor_01/*.wav   (RAVDESS format)
└── transcripts.csv      (text, label columns)
```

### Step 3: Train

```bash
# Train all models
python train.py --all --data data/raw

# Train individual models
python train.py --facial --data data/raw/facial
python train.py --speech --data data/raw/speech
python train.py --nlp    --data data/raw/transcripts.csv
```

---

## 🌐 API Reference

### `POST /analyze`
Upload a video file for full multimodal analysis.
```bash
curl -X POST -F "video=@interview.mp4" http://localhost:5000/analyze
```

### `POST /analyze/text`
Analyze a text transcript only.
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"transcript": "I led a team of 5 engineers..."}' \
  http://localhost:5000/analyze/text
```

### `GET /results/<job_id>`
Retrieve a cached result.

---

## 📊 Output Scores

| Score | Description | Range |
|-------|-------------|-------|
| `overall_suitability` | Weighted fusion of all modalities | 0–100 |
| `confidence_score` | Facial + speech energy analysis | 0–100 |
| `communication_skill` | Speech clarity + NLP quality | 0–100 |
| `emotional_stability` | Facial + speech emotion consistency | 0–100 |

---

## 🏗️ Architecture Summary

```
Interview Video
      │
      ├─── Frame Extraction ──→ Face Detection ──→ EfficientNet-B0 ──→ Facial Embedding (256-d)
      │
      ├─── Audio Extraction ──→ Mel Spectrogram ──→ CNN+BiGRU ──────→ Speech Embedding (512-d)
      │                    └──→ Whisper STT ──────→ BERT ───────────→ Text Embedding (768-d)
      │
      └─── Cross-Modal Attention Fusion ──→ Gated MLP ──→ Suitability Score (0-100)
```

---

## 🔧 Configuration

All settings are in `config/config.py`. Key tuneable parameters:

```python
facial_cfg.backbone = "vit_small_patch16_224"  # Switch to ViT
fusion_cfg.fusion_method = "attention"          # attention | concat | weighted
score_cfg.nlp_weight = 0.40                     # Increase NLP weight
```

---

## 📦 Tech Stack

- **Deep Learning**: PyTorch, timm (ViT/EfficientNet), HuggingFace Transformers (BERT)
- **Vision**: OpenCV, torchvision
- **Audio**: Librosa, torchaudio, OpenAI Whisper
- **Backend**: Flask, Flask-CORS
- **Visualization**: Plotly, Seaborn
- **Deployment**: Git, venv, ffmpeg
