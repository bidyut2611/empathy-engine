#!/usr/bin/env python3
"""
Empathy Engine — CLI Interface

Usage:
    python cli.py "I am so happy today!"
    python cli.py "This is terrible." --output angry.wav
    python cli.py "Hello world" --engine gtts --no-hf
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from empathy_engine.engine import EmpathyEngine


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------
class C:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    END     = "\033[0m"


EMOTION_COLORS = {
    "joy":       C.GREEN,
    "sadness":   C.BLUE,
    "anger":     C.RED,
    "surprise":  C.YELLOW,
    "fear":      C.YELLOW,
    "disgust":   C.RED,
    "neutral":   C.CYAN,
}


def print_banner():
    print(f"""
{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════╗
║         🎭  THE EMPATHY ENGINE  🎭                  ║
║      Giving AI a Human Voice                         ║
╚══════════════════════════════════════════════════════╝{C.END}
""")


def print_result(result: dict):
    emotion = result["emotion"]
    voice = result["voice"]
    color = EMOTION_COLORS.get(emotion["primary_emotion"], C.CYAN)

    print(f"{C.BOLD}📝 Input Text:{C.END}")
    print(f"   \"{emotion['text']}\"\n")

    print(f"{C.BOLD}🔍 Emotion Analysis:{C.END}")
    print(f"   Primary Emotion: {color}{C.BOLD}{emotion['primary_emotion'].upper()}{C.END}")
    print(f"   Granular Label:  {color}{emotion['granular_label']}{C.END}")
    print(f"   Intensity:       {_intensity_bar(emotion['intensity'])} {emotion['intensity']:.1%}")

    if emotion.get("vader_scores"):
        v = emotion["vader_scores"]
        print(f"\n   VADER Scores:")
        print(f"     Compound: {v.get('compound', 0):+.3f}  |  "
              f"Pos: {v.get('pos', 0):.3f}  |  "
              f"Neg: {v.get('neg', 0):.3f}  |  "
              f"Neu: {v.get('neu', 0):.3f}")

    if emotion.get("hf_scores"):
        print(f"\n   HuggingFace Emotion Scores:")
        sorted_scores = sorted(emotion["hf_scores"].items(), key=lambda x: -x[1])
        for label, score in sorted_scores:
            bar = "█" * int(score * 20)
            highlight = C.BOLD if label == emotion["granular_label"] else ""
            print(f"     {highlight}{label:>10s}: {bar} {score:.3f}{C.END}")

    print(f"\n{C.BOLD}🎤 Voice Parameters:{C.END}")
    print(f"   Rate:   {voice['rate']} wpm")
    print(f"   Pitch:  {voice['pitch']:.2f}×")
    print(f"   Volume: {voice['volume']:.2f}")

    print(f"\n{C.BOLD}🔊 Audio Output:{C.END}")
    print(f"   {C.GREEN}{result['audio']}{C.END}")
    print(f"\n   ⏱  Processed in {result['time_ms']:.0f} ms")
    print()


def _intensity_bar(intensity: float) -> str:
    filled = int(intensity * 20)
    return f"[{'█' * filled}{'░' * (20 - filled)}]"


def main():
    parser = argparse.ArgumentParser(
        description="The Empathy Engine — Text to Empathetic Speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "I'm so happy today!"
  python cli.py "This is frustrating." --output frustrated.wav
  python cli.py "Good morning." --engine gtts
  python cli.py "Hello world" --no-hf
        """,
    )
    parser.add_argument("text", help="Text to synthesize with emotion")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output audio file path (default: auto-generated)",
    )
    parser.add_argument(
        "--engine", "-e",
        choices=["pyttsx3", "gtts", "auto"],
        default="auto",
        help="TTS engine to use (default: auto — macOS 'say' on Mac, pyttsx3 elsewhere)",
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Disable HuggingFace model (use VADER-only mode)",
    )

    args = parser.parse_args()

    print_banner()

    # Initialise engine
    engine = EmpathyEngine(
        tts_backend=args.engine,
        use_hf=not args.no_hf,
    )

    # Process
    result = engine.process(args.text, output_path=args.output)

    # Display results
    print_result(result)


if __name__ == "__main__":
    main()
