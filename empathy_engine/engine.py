"""
Empathy Engine — Orchestrator

Main pipeline: text → emotion detection → voice mapping → TTS → audio file.
"""

import time
from typing import Optional

from empathy_engine.emotion_detector import EmotionDetector
from empathy_engine.voice_mapper import VoiceMapper
from empathy_engine.tts_engine import TTSEngine


class EmpathyEngine:
    """
    The Empathy Engine orchestrates the full text-to-empathetic-speech pipeline.
    """

    def __init__(self, tts_backend: str = "auto", use_hf: bool = True):
        """
        Parameters
        ----------
        tts_backend : str
            TTS backend to use: "pyttsx3" (offline) or "gtts" (online).
        use_hf : bool
            Whether to use HuggingFace model for granular emotion detection.
        """
        print("🔧 Initializing Empathy Engine...")
        self.detector = EmotionDetector(use_hf=use_hf)
        self.mapper = VoiceMapper()
        self.tts = TTSEngine(backend=tts_backend)
        print("✅ Empathy Engine ready!\n")

    def process(
        self,
        text: str,
        output_path: Optional[str] = None,
    ) -> dict:
        """
        Full pipeline: analyse text → modulate voice → generate audio.

        Returns
        -------
        dict with keys:
            emotion   – EmotionResult.to_dict()
            voice     – VoiceParams.to_dict()
            mapping   – human-readable explanation
            audio     – absolute path to output audio file
            time_ms   – processing time in milliseconds
        """
        t0 = time.time()

        # 1. Detect emotion
        emotion = self.detector.detect(text)

        # 2. Map emotion → voice parameters
        voice_params = self.mapper.map(emotion.primary_emotion, emotion.intensity)
        mapping_explanation = self.mapper.explain(
            emotion.primary_emotion, emotion.intensity
        )

        # 3. Synthesise speech
        audio_path = self.tts.synthesize(text, voice_params, output_path)

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        return {
            "emotion": emotion.to_dict(),
            "voice": voice_params.to_dict(),
            "mapping": mapping_explanation,
            "audio": audio_path,
            "time_ms": elapsed_ms,
        }
