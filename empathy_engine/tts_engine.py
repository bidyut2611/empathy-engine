"""
TTS Engine Module

Synthesises speech from text with vocal parameter modulation.
Supports multiple backends:
  1. macOS 'say' command (native, reliable, supports rate/voice)
  2. pyttsx3 (cross-platform offline)
  3. gTTS (online, Google)
"""

import os
import sys
import uuid
import subprocess
import tempfile
import platform
from pathlib import Path

from empathy_engine.voice_mapper import VoiceParams


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _is_macos() -> bool:
    return platform.system() == "Darwin"


class TTSEngine:
    """
    Text-to-Speech engine that applies VoiceParams before synthesis.

    Backends (in reliability order on macOS):
      • macos_say  – macOS native 'say' command (most reliable on Mac)
      • pyttsx3    – cross-platform offline
      • gtts       – online (Google), with pydub post-processing
    """

    def __init__(self, backend: str = "auto"):
        """
        Parameters
        ----------
        backend : str
            "auto", "pyttsx3", "gtts", or "macos_say"
            "auto" selects macos_say on macOS, pyttsx3 elsewhere.
        """
        if backend == "auto":
            self.backend = "macos_say" if _is_macos() else "pyttsx3"
        else:
            self.backend = backend.lower()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        params: VoiceParams,
        output_path: str | None = None,
    ) -> str:
        """
        Synthesise *text* with the given *params* and save to *output_path*.
        Returns the absolute path to the generated audio file.
        """
        if output_path is None:
            ext = ".mp3" if self.backend == "gtts" else ".wav"
            output_path = str(OUTPUT_DIR / f"empathy_{uuid.uuid4().hex[:8]}{ext}")

        if self.backend == "macos_say":
            return self._synthesize_macos_say(text, params, output_path)
        elif self.backend == "pyttsx3":
            return self._synthesize_pyttsx3(text, params, output_path)
        elif self.backend == "gtts":
            return self._synthesize_gtts(text, params, output_path)
        else:
            raise ValueError(f"Unknown TTS backend: {self.backend}")

    # ------------------------------------------------------------------
    # macOS 'say' backend  (most reliable on macOS)
    # ------------------------------------------------------------------
    def _synthesize_macos_say(
        self, text: str, params: VoiceParams, output_path: str
    ) -> str:
        """
        Use macOS native 'say' command.
        Supports: -r (rate in wpm), -o (output file), -v (voice)
        Generates AIFF then converts to WAV for browser compatibility.
        """
        # Ensure .wav extension for browser compatibility
        if output_path.endswith(".aiff"):
            output_path = output_path[:-5] + ".wav"
        elif not output_path.endswith((".wav", ".mp3")):
            output_path += ".wav"

        # say outputs AIFF natively — we'll convert to WAV after
        aiff_path = output_path.rsplit(".", 1)[0] + ".aiff"

        # Build command
        cmd = ["say"]

        # Rate (words per minute) — directly supported
        cmd.extend(["-r", str(params.rate)])

        # Voice selection based on pitch:
        #   Higher pitch → use a female voice (Samantha)
        #   Lower pitch → use a male voice (Alex or Daniel)
        #   Normal pitch → default Samantha
        if params.pitch < 0.9:
            cmd.extend(["-v", "Daniel"])   # deeper male voice
        elif params.pitch > 1.2:
            cmd.extend(["-v", "Samantha"]) # higher female voice
        else:
            cmd.extend(["-v", "Samantha"]) # default

        # Output file (AIFF first)
        cmd.extend(["-o", aiff_path])

        # Text
        cmd.append(text)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"'say' command failed: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError("macOS 'say' command not found. Use --engine pyttsx3 or --engine gtts.")

        # Convert AIFF → WAV using macOS afconvert (built-in)
        try:
            convert_result = subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16", aiff_path, output_path],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if convert_result.returncode == 0:
                # Remove the intermediate AIFF file
                try:
                    os.remove(aiff_path)
                except OSError:
                    pass
            else:
                # If conversion fails, just use the AIFF file
                output_path = aiff_path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # afconvert not available — fallback to AIFF
            output_path = aiff_path

        # Post-process volume with pydub if needed
        if abs(params.volume - 0.85) > 0.05:
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(output_path)
                volume_db = (params.volume - 0.85) * 30  # map to dB change
                audio = audio + volume_db
                fmt = "wav" if output_path.endswith(".wav") else "aiff"
                audio.export(output_path, format=fmt)
            except (ImportError, Exception):
                pass  # Skip volume adjustment if pydub not available

        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # pyttsx3 backend
    # ------------------------------------------------------------------
    def _synthesize_pyttsx3(
        self, text: str, params: VoiceParams, output_path: str
    ) -> str:
        """
        Use pyttsx3 in a subprocess to avoid event loop hanging issues.
        """
        if not output_path.endswith((".wav", ".mp3", ".aiff")):
            output_path += ".aiff"

        output_path = os.path.abspath(output_path)

        # Run pyttsx3 in a subprocess to avoid macOS runLoop hang
        script = f"""
import pyttsx3
import sys

engine = pyttsx3.init()
engine.setProperty('rate', {params.rate})
engine.setProperty('volume', {params.volume})

try:
    voices = engine.getProperty('voices')
    if voices:
        engine.setProperty('voice', voices[0].id)
except Exception:
    pass

engine.save_to_file({repr(text)}, {repr(output_path)})
engine.runAndWait()
engine.stop()
sys.exit(0)
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"pyttsx3 failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            # Fallback to macos_say if on macOS
            if _is_macos():
                return self._synthesize_macos_say(text, params, output_path)
            raise RuntimeError("pyttsx3 timed out during synthesis.")

        return output_path

    # ------------------------------------------------------------------
    # gTTS backend  (online, with pydub speed adjustment)
    # ------------------------------------------------------------------
    def _synthesize_gtts(
        self, text: str, params: VoiceParams, output_path: str
    ) -> str:
        from gtts import gTTS

        # gTTS produces an mp3
        if not output_path.endswith(".mp3"):
            output_path = output_path.rsplit(".", 1)[0] + ".mp3"

        tmp_path = os.path.join(tempfile.gettempdir(), f"gtts_{uuid.uuid4().hex[:8]}.mp3")
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(tmp_path)

        # Post-process with pydub for speed / volume adjustment
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_mp3(tmp_path)

            # Speed adjustment: map rate (wpm) to a playback speed factor
            baseline_rate = 200
            speed_factor = params.rate / baseline_rate
            speed_factor = max(0.5, min(2.0, speed_factor))

            if abs(speed_factor - 1.0) > 0.05:
                new_frame_rate = int(audio.frame_rate * speed_factor)
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": new_frame_rate
                }).set_frame_rate(audio.frame_rate)

            # Volume adjustment
            volume_db = (params.volume - 0.85) * 20
            audio = audio + volume_db

            audio.export(output_path, format="mp3")
        except ImportError:
            import shutil
            shutil.move(tmp_path, output_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        return os.path.abspath(output_path)
