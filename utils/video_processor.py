# utils/video_processor.py
"""
Handles video ingestion:
  - Frame extraction at configurable FPS
  - Face detection & alignment using OpenCV + Haar / DNN
  - Audio stream extraction
"""

import cv2
import numpy as np
import subprocess
import os
from pathlib import Path
from loguru import logger
from typing import Generator, Tuple, List, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import video_cfg, audio_cfg


class VideoProcessor:
    """Extract frames, faces, and audio from an interview video file."""

    # Haar cascade for fast face detection
    _HAAR = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        logger.info(f"Loaded video: {self.video_path.name} | "
                    f"{self.duration:.1f}s | {self.fps:.0f}fps")

    # ── Frame Extraction ──────────────────────────────────────────────────────

    def extract_frames(self, sample_fps: int = None) -> Generator[Tuple[float, np.ndarray], None, None]:
        """
        Yield (timestamp_sec, BGR_frame) at the target sample_fps.
        """
        sample_fps = sample_fps or video_cfg.frame_rate
        interval = max(1, int(self.fps / sample_fps))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                timestamp = frame_idx / self.fps
                yield timestamp, frame
            frame_idx += 1

    # ── Face Detection & Alignment ────────────────────────────────────────────

    def detect_and_crop_face(
        self, frame: np.ndarray, target_size: Tuple[int, int] = None
    ) -> Optional[np.ndarray]:
        """
        Detect the largest face in a frame, crop & resize it.
        Returns None if no face is found.
        """
        target_size = target_size or video_cfg.face_size
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._HAAR.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None
        # Pick the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # Add 20% padding
        pad_x, pad_y = int(w * 0.2), int(h * 0.2)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)
        face = frame[y1:y2, x1:x2]
        return cv2.resize(face, target_size)

    def get_face_sequence(self) -> List[Tuple[float, np.ndarray]]:
        """
        Returns list of (timestamp, face_crop) for all frames with a detected face.
        """
        faces = []
        for ts, frame in self.extract_frames():
            face = self.detect_and_crop_face(frame)
            if face is not None:
                faces.append((ts, face))
        logger.info(f"Extracted {len(faces)} face crops from video.")
        return faces

    # ── Audio Extraction ──────────────────────────────────────────────────────

    def extract_audio(self, output_path: str = None) -> str:
        """
        Extract audio track to a .wav file using ffmpeg.
        Returns path to the saved audio file.
        """
        if output_path is None:
            output_path = str(self.video_path.with_suffix(".wav"))
        cmd = [
            "ffmpeg", "-y", "-i", str(self.video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(audio_cfg.sample_rate),
            "-ac", "1", output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg error: {result.stderr}")
            raise RuntimeError("Audio extraction failed. Is ffmpeg installed?")
        logger.info(f"Audio extracted to: {output_path}")
        return output_path

    def __del__(self):
        if hasattr(self, "cap"):
            self.cap.release()
