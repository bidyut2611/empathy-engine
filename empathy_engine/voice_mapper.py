"""
Voice Mapper Module

Maps detected emotions to vocal parameter adjustments (rate, pitch, volume)
with linear intensity scaling. Supports granular emotions.
"""

from dataclasses import dataclass


@dataclass
class VoiceParams:
    """Voice parameter adjustments relative to baseline."""
    rate: int          # words-per-minute (pyttsx3 default ~200)
    pitch: float       # pitch multiplier (1.0 = baseline)
    volume: float      # 0.0 – 1.0

    def to_dict(self) -> dict:
        return {
            "rate": self.rate,
            "pitch": round(self.pitch, 2),
            "volume": round(self.volume, 2),
        }


# ---------------------------------------------------------------------------
# Baseline values
# ---------------------------------------------------------------------------
BASELINE_RATE   = 200   # wpm
BASELINE_PITCH  = 1.0
BASELINE_VOLUME = 0.85

# ---------------------------------------------------------------------------
# Per-emotion adjustments at MAXIMUM intensity (intensity=1.0)
# Values are (delta_rate, delta_pitch_mult, delta_volume)
# delta_rate:          +/- wpm from baseline
# delta_pitch_mult:    multiplied onto baseline pitch
# delta_volume:        added onto baseline volume
# ---------------------------------------------------------------------------
EMOTION_PROFILE = {
    # Emotion         rate_delta  pitch_mult  vol_delta
    "joy":           ( +60,       1.35,       +0.15 ),
    "sadness":       ( -50,       0.78,       -0.15 ),
    "anger":         ( +40,       1.22,       +0.15 ),
    "surprise":      ( +70,       1.45,       +0.10 ),
    "fear":          ( +50,       1.18,       -0.10 ),
    "disgust":       ( -20,       0.88,       -0.05 ),
    "neutral":       (   0,       1.00,        0.00 ),

    # Extended / nuanced emotions (bonus)
    "inquisitive":   ( +15,       1.20,       +0.05 ),
    "concerned":     ( -15,       0.92,       -0.05 ),
    "excited":       ( +65,       1.40,       +0.15 ),
    "calm":          ( -30,       0.90,       -0.10 ),
}


class VoiceMapper:
    """
    Convert an emotion label + intensity into concrete VoiceParams.

    All adjustments are linearly interpolated between baseline (intensity=0)
    and the full profile value (intensity=1).
    """

    def __init__(
        self,
        baseline_rate: int = BASELINE_RATE,
        baseline_pitch: float = BASELINE_PITCH,
        baseline_volume: float = BASELINE_VOLUME,
    ):
        self.baseline_rate = baseline_rate
        self.baseline_pitch = baseline_pitch
        self.baseline_volume = baseline_volume

    def map(self, emotion: str, intensity: float) -> VoiceParams:
        """
        Return VoiceParams for the given *emotion* and *intensity* (0–1).

        Unknown emotions are treated as neutral.
        """
        intensity = max(0.0, min(1.0, intensity))  # clamp
        profile = EMOTION_PROFILE.get(emotion, EMOTION_PROFILE["neutral"])

        rate_delta, pitch_mult, vol_delta = profile

        # Linear interpolation from baseline to full profile
        rate = int(self.baseline_rate + rate_delta * intensity)
        pitch = self.baseline_pitch + (pitch_mult - self.baseline_pitch) * intensity
        volume = max(0.1, min(1.0,
                    self.baseline_volume + vol_delta * intensity))

        return VoiceParams(rate=rate, pitch=pitch, volume=volume)

    def explain(self, emotion: str, intensity: float) -> str:
        """Return a human-readable explanation of the mapping."""
        params = self.map(emotion, intensity)
        profile = EMOTION_PROFILE.get(emotion, EMOTION_PROFILE["neutral"])
        rate_delta, pitch_mult, vol_delta = profile

        lines = [
            f"Emotion:   {emotion}",
            f"Intensity: {intensity:.1%}",
            f"",
            f"Voice Parameters (intensity-scaled):",
            f"  Rate:   {params.rate} wpm  (baseline {self.baseline_rate}, "
            f"delta {rate_delta:+d} × {intensity:.2f})",
            f"  Pitch:  {params.pitch:.2f}×  (baseline {self.baseline_pitch}, "
            f"target {pitch_mult}× × {intensity:.2f})",
            f"  Volume: {params.volume:.2f}  (baseline {self.baseline_volume}, "
            f"delta {vol_delta:+.2f} × {intensity:.2f})",
        ]
        return "\n".join(lines)
