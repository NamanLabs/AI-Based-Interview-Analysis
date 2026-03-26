# pipeline.py
"""
Main Interview Analysis Pipeline.

Usage:
    from pipeline import InterviewAnalysisPipeline
    
    pipeline = InterviewAnalysisPipeline()
    results  = pipeline.analyze("path/to/interview.mp4")
    print(results)
"""

import os
import json
import tempfile
from pathlib import Path
from loguru import logger
from typing import Dict, Optional

import sys
sys.path.append(str(Path(__file__).parent))

from utils.video_processor import VideoProcessor
from utils.audio_processor import AudioProcessor
from models.facial_emotion_model import FacialEmotionAnalyzer
from models.speech_emotion_model import SpeechEmotionAnalyzer
from models.nlp_model import NLPAnalyzer
from models.fusion_model import InterviewScorer
from config.config import MODELS_DIR


class InterviewAnalysisPipeline:
    """
    End-to-end pipeline:
      Video → [Frames + Audio] → [Facial | Speech | NLP] → Fusion → Scores
    """

    def __init__(
        self,
        facial_weights: Optional[str] = None,
        speech_weights: Optional[str] = None,
        nlp_weights: Optional[str] = None,
        device: Optional[str] = None
    ):
        logger.info("Initializing Interview Analysis Pipeline...")

        # Auto-detect saved weights
        facial_weights = facial_weights or str(MODELS_DIR / "facial_emotion_best.pth")
        speech_weights = speech_weights or str(MODELS_DIR / "speech_emotion_best.pth")
        nlp_weights    = nlp_weights    or str(MODELS_DIR / "nlp_sentiment_best.pth")

        self.facial_analyzer = FacialEmotionAnalyzer(facial_weights, device)
        self.speech_analyzer = SpeechEmotionAnalyzer(speech_weights, device)
        self.nlp_analyzer    = NLPAnalyzer(nlp_weights, device)
        self.scorer          = InterviewScorer()

        logger.info("Pipeline ready.")

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def analyze(self, video_path: str, save_report: bool = False) -> Dict:
     
        logger.info(f"▶ Starting analysis: {video_path}")

        # ── Step 1: Video Processing ──────────────────────────────────────
        logger.info("Step 1/5: Processing video...")
        video_proc = VideoProcessor(video_path)
        face_sequence = video_proc.get_face_sequence()   # [(ts, face_crop), ...]

        if not face_sequence:
            return {"error": "No faces detected in video. Please ensure the candidate is visible."}

        # ── Step 2: Audio Extraction ──────────────────────────────────────
        logger.info("Step 2/5: Extracting audio...")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.wav")
            try:
                audio_path = video_proc.extract_audio(audio_path)
                audio_proc = AudioProcessor(audio_path)
                audio_available = True
            except RuntimeError as e:
                logger.warning(f"Audio extraction failed: {e}. Skipping audio analysis.")
                audio_available = False

            # ── Step 3: Facial Analysis ───────────────────────────────────
            logger.info("Step 3/5: Running facial emotion analysis...")
            face_crops = [crop for _, crop in face_sequence]
            facial_result = self.facial_analyzer.analyze_sequence(face_crops)

            # ── Step 4: Speech + NLP Analysis ────────────────────────────
            if audio_available:
                logger.info("Step 4/5: Running speech emotion analysis...")
                speech_result = self.speech_analyzer.analyze(audio_proc)

                logger.info("Step 4b/5: Transcribing speech...")
                transcript = audio_proc.transcribe()

                logger.info("Step 4c/5: Running NLP analysis...")
                nlp_result = self.nlp_analyzer.analyze(transcript)
            else:
                speech_result = self._empty_speech_result()
                nlp_result    = self._empty_nlp_result()
                transcript    = ""

            # ── Step 5: Fusion & Scoring ──────────────────────────────────
            logger.info("Step 5/5: Computing final scores...")
            final_scores = self.scorer.compute_scores(
                facial_result, speech_result, nlp_result
            )

        # ── Compile Full Report ───────────────────────────────────────────
        report = {
            "video_path":        video_path,
            "duration_seconds":  video_proc.duration,
            "faces_analyzed":    facial_result.get("total_frames_analyzed", 0),
            "transcript":        transcript,

            "facial_analysis":   facial_result,
            "speech_analysis":   speech_result,
            "nlp_analysis":      nlp_result,
            "final_scores":      final_scores
        }

        if save_report:
            report_path = Path(video_path).with_suffix(".json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {report_path}")

        logger.info(f"✓ Analysis complete. Overall score: {final_scores['overall_suitability']}/100")
        return report

    # ── Text-Only Mode (no video) ─────────────────────────────────────────────

    def analyze_transcript(self, transcript: str) -> Dict:
        """
        Analyze a plain-text transcript (no video required).
        Returns NLP scores only.
        """
        nlp_result  = self.nlp_analyzer.analyze(transcript)
        empty_face  = self._empty_facial_result()
        empty_speech = self._empty_speech_result()
        scores = self.scorer.compute_scores(empty_face, empty_speech, nlp_result)
        return {"nlp_analysis": nlp_result, "scores": scores}

    # ── Fallback empty results ────────────────────────────────────────────────

    @staticmethod
    def _empty_facial_result() -> Dict:
        return {
            "per_frame": [],
            "aggregate": {},
            "dominant_emotion": "Neutral",
            "emotional_stability_score": 50.0,
            "total_frames_analyzed": 0
        }

    @staticmethod
    def _empty_speech_result() -> Dict:
        return {
            "emotion_probabilities": {},
            "dominant_emotion": "Neutral",
            "prosodic_features": {},
            "communication_score": 50.0
        }

    @staticmethod
    def _empty_nlp_result() -> Dict:
        return {
            "transcript_length": 0,
            "sentence_count": 0,
            "sentiment": {"dominant": "Neutral", "positive_ratio": 0.5},
            "competencies_detected": {},
            "communication_quality": {},
            "nlp_score": 50.0
        }


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interview Analysis Pipeline")
    parser.add_argument("video", help="Path to interview video file")
    parser.add_argument("--save", action="store_true", help="Save JSON report")
    args = parser.parse_args()

    pipe   = InterviewAnalysisPipeline()
    result = pipe.analyze(args.video, save_report=args.save)

    # Pretty print final scores
    scores = result.get("final_scores", {})
    print("\n" + "="*50)
    print("  INTERVIEW ANALYSIS RESULTS")
    print("="*50)
    for key, val in scores.items():
        if not isinstance(val, dict):
            print(f"  {key:<30} {val}")
    print("="*50)
