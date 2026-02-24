"""
Emotion Detection Module

Combines VADER sentiment analysis with HuggingFace transformer-based
emotion classification for granular, intensity-aware emotion detection.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

# VADER for sentiment scoring
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# HuggingFace for granular emotion classification
try:
    from transformers import pipeline as hf_pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Emotion categories  (must support at least 3 — we support 7+)
# ---------------------------------------------------------------------------
EMOTION_CATEGORIES = [
    "joy", "sadness", "anger", "surprise", "fear", "disgust", "neutral"
]

# Map HuggingFace model labels → our canonical labels
HF_LABEL_MAP = {
    "joy":      "joy",
    "sadness":  "sadness",
    "anger":    "anger",
    "surprise": "surprise",
    "fear":     "fear",
    "disgust":  "disgust",
    "neutral":  "neutral",
}

# Map VADER-only polarity to a coarse emotion when HF is unavailable
VADER_POLARITY_MAP = {
    "positive": "joy",
    "negative": "sadness",
    "neutral":  "neutral",
}


@dataclass
class EmotionResult:
    """Result of emotion analysis on a piece of text."""
    text: str
    primary_emotion: str          # e.g. "joy", "anger", "neutral"
    intensity: float              # 0.0 – 1.0
    granular_label: str           # more nuanced label from HF model
    vader_scores: Dict[str, float] = field(default_factory=dict)
    hf_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "primary_emotion": self.primary_emotion,
            "intensity": round(self.intensity, 3),
            "granular_label": self.granular_label,
            "vader_scores": {k: round(v, 3) for k, v in self.vader_scores.items()},
            "hf_scores": {k: round(v, 3) for k, v in self.hf_scores.items()},
        }


class EmotionDetector:
    """
    Two-stage emotion detector:
      1. VADER — fast compound / pos / neg / neu scores + intensity
      2. HuggingFace distilroberta — granular emotion labels
    Falls back to VADER-only mode if transformers is unavailable.
    """

    def __init__(self, use_hf: bool = True):
        # Always initialise VADER
        self.vader = SentimentIntensityAnalyzer()

        # Optionally load HuggingFace emotion classifier
        self.hf_classifier = None
        if use_hf and HF_AVAILABLE:
            try:
                # Suppress symlink warnings on Windows
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
                self.hf_classifier = hf_pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None,           # return all labels with scores
                    truncation=True,
                )
                print("✓ HuggingFace emotion model loaded successfully")
            except Exception as exc:
                print(f"⚠ HuggingFace model could not be loaded ({exc}). "
                      "Falling back to VADER-only mode.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, text: str) -> EmotionResult:
        """Analyse *text* and return an EmotionResult."""
        vader_scores = self._vader_analyse(text)
        intensity = self._compute_intensity(vader_scores)

        if self.hf_classifier is not None:
            hf_scores, granular_label = self._hf_analyse(text)
            primary_emotion = HF_LABEL_MAP.get(granular_label, granular_label)
        else:
            hf_scores = {}
            polarity = self._vader_polarity(vader_scores)
            primary_emotion = VADER_POLARITY_MAP[polarity]
            granular_label = primary_emotion

        return EmotionResult(
            text=text,
            primary_emotion=primary_emotion,
            intensity=intensity,
            granular_label=granular_label,
            vader_scores=vader_scores,
            hf_scores=hf_scores,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _vader_analyse(self, text: str) -> Dict[str, float]:
        return self.vader.polarity_scores(text)

    @staticmethod
    def _vader_polarity(scores: Dict[str, float]) -> str:
        compound = scores["compound"]
        if compound >= 0.05:
            return "positive"
        elif compound <= -0.05:
            return "negative"
        return "neutral"

    @staticmethod
    def _compute_intensity(vader_scores: Dict[str, float]) -> float:
        """Map VADER compound score (−1 … +1) to intensity (0 … 1)."""
        return min(abs(vader_scores["compound"]) * 1.2, 1.0)

    def _hf_analyse(self, text: str):
        """Return (scores_dict, top_label) from HuggingFace model."""
        results = self.hf_classifier(text)[0]  # list of {label, score}
        scores = {r["label"]: r["score"] for r in results}
        top_label = max(scores, key=scores.get)
        return scores, top_label
