# 🎭 The Empathy Engine — Giving AI a Human Voice

A Python service that **dynamically modulates vocal characteristics** of synthesized speech based on the **detected emotion** of the source text. Moving beyond monotonic TTS delivery to achieve expressive, human-like audio output.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
  - [CLI](#cli)
  - [Web Interface](#web-interface)
- [Emotion-to-Voice Mapping](#emotion-to-voice-mapping)
- [Design Choices](#design-choices)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

---

## Overview

Standard Text-to-Speech systems produce functional but robotic output — they lack prosody, emotional range, and the subtle vocal cues that build rapport. **The Empathy Engine** bridges this gap by:

1. **Detecting emotion** from input text using dual-engine analysis (VADER + HuggingFace transformers)
2. **Mapping emotions** to vocal parameter adjustments with intensity-aware scaling
3. **Synthesizing speech** with modulated rate, pitch, and volume to sound genuinely expressive

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    THE EMPATHY ENGINE                    │
│                                                         │
│  Input Text                                             │
│      │                                                  │
│      ▼                                                  │
│  ┌──────────────────────┐                               │
│  │  EMOTION DETECTOR    │                               │
│  │  ├─ VADER Sentiment  │──► Compound Score ──► Intensity│
│  │  └─ HuggingFace      │──► 7 Emotion Labels          │
│  │     DistilRoBERTa    │                               │
│  └──────────┬───────────┘                               │
│             │                                           │
│             ▼                                           │
│  ┌──────────────────────┐                               │
│  │  VOICE MAPPER        │                               │
│  │  emotion + intensity │──► VoiceParams(rate, pitch,   │
│  │  → linear scaling    │                    volume)    │
│  └──────────┬───────────┘                               │
│             │                                           │
│             ▼                                           │
│  ┌──────────────────────┐                               │
│  │  TTS ENGINE          │                               │
│  │  ├─ pyttsx3 (offline)│──► .wav audio file            │
│  │  └─ gTTS   (online)  │──► .mp3 audio file            │
│  └──────────────────────┘                               │
└─────────────────────────────────────────────────────────┘
```

---

## Features

### ✅ Core (Must-Haves)

| # | Requirement | Implementation |
|---|-------------|---------------|
| 1 | **Text Input** | CLI prompt + Flask API endpoint |
| 2 | **Emotion Detection** | 7 categories: joy, sadness, anger, surprise, fear, disgust, neutral |
| 3 | **Vocal Parameter Modulation** | Rate (wpm), Pitch (multiplier), Volume (0–1) |
| 4 | **Emotion-to-Voice Mapping** | Documented, demonstrable mapping table with clear logic |
| 5 | **Audio Output** | Generates playable `.wav` / `.mp3` files |

### 🌟 Bonus (Stretch Goals)

| Feature | Implementation |
|---------|---------------|
| **Granular Emotions** | 7+ emotions via HuggingFace `j-hartmann/emotion-english-distilroberta-base` |
| **Intensity Scaling** | All vocal adjustments scale linearly with emotion intensity (0.0–1.0) |
| **Web Interface** | Flask app with text input, emotion visualization, and embedded audio player |
| **SSML-Ready Architecture** | Modular design ready for SSML integration |

---

## Setup & Installation

### Prerequisites

- **Python 3.9+**
- **pip** (Python package manager)
- **ffmpeg** (optional, only needed for gTTS post-processing with pydub)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/empathy-engine.git
cd empathy-engine

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (for VADER)
python -c "import nltk; nltk.download('vader_lexicon')"
```

> **Note:** The HuggingFace model (~300 MB) will be downloaded automatically on first run. Use `--no-hf` flag to skip this and use VADER-only mode for faster startup.

---

## Usage

### CLI

```bash
# Basic usage — positive emotion
python cli.py "I'm so excited about this new opportunity!"

# Negative emotion with custom output file
python cli.py "This is really frustrating and disappointing." --output frustrated.wav

# Use gTTS (online) instead of pyttsx3
python cli.py "Hello, how are you today?" --engine gtts

# Fast mode — VADER only, no HuggingFace model
python cli.py "Great job everyone!" --no-hf
```

**CLI Output Example:**

```
╔══════════════════════════════════════════════════════╗
║         🎭  THE EMPATHY ENGINE  🎭                  ║
║      Giving AI a Human Voice                         ║
╚══════════════════════════════════════════════════════╝

📝 Input Text:
   "I'm so excited about this new opportunity!"

🔍 Emotion Analysis:
   Primary Emotion: JOY
   Granular Label:  joy
   Intensity:       [████████████████░░░░] 82.3%

🎤 Voice Parameters:
   Rate:   249 wpm
   Pitch:  1.29×
   Volume: 0.97

🔊 Audio Output:
   /path/to/output/empathy_a1b2c3d4.wav
```

### Web Interface

```bash
# Start the Flask server
python web/app.py

# Open in browser
# http://localhost:5000
```

The web interface provides:
- 📝 Text area for input
- 🎤 "Speak with Emotion" button
- 🔍 Emotion analysis with visual breakdown (emotion badge, intensity bar, score chart)
- 🎤 Voice parameter display (rate, pitch, volume with baseline comparison)
- 🔊 Embedded audio player for instant playback

---

## Emotion-to-Voice Mapping

The mapping logic is the heart of the Empathy Engine. Each emotion has a **profile** defining maximum adjustments for rate, pitch, and volume. These are **linearly scaled** by the detected intensity.

### Mapping Table (at maximum intensity)

| Emotion | Rate (wpm) | Pitch | Volume | Rationale |
|---------|-----------|-------|--------|-----------|
| 😊 **Joy** | 260 (+30%) | 1.35× | 1.00 | Happy speech is faster, higher-pitched, and louder |
| 😢 **Sadness** | 150 (-25%) | 0.78× | 0.70 | Sad speech is slow, low-pitched, and quiet |
| 😠 **Anger** | 240 (+20%) | 1.22× | 1.00 | Angry speech is forceful — faster, higher, and louder |
| 😲 **Surprise** | 270 (+35%) | 1.45× | 0.95 | Surprise produces rapid, highly-pitched exclamations |
| 😨 **Fear** | 250 (+25%) | 1.18× | 0.75 | Fear produces fast but quiet, slightly higher speech |
| 🤢 **Disgust** | 180 (-10%) | 0.88× | 0.80 | Disgust slows speech with a lower, restrained tone |
| 😐 **Neutral** | 200 (base) | 1.00× | 0.85 | Normal conversational delivery |

### Intensity Scaling Formula

For any parameter `P` with baseline value `B` and emotion delta `D`:

```
P = B + D × intensity
```

Where `intensity ∈ [0.0, 1.0]` is derived from the VADER compound score:

```
intensity = min(|compound| × 1.2, 1.0)
```

**Example:** "This is good" (compound=0.44) → intensity=0.53 → moderate adjustment.
"This is the best news ever!" (compound=0.87) → intensity=1.0 → full adjustment.

---

## Design Choices

### Why Dual Emotion Detection?

- **VADER** excels at sentiment intensity scoring with its lexicon+rule approach, providing a reliable compound score for intensity calculation
- **HuggingFace DistilRoBERTa** provides granular 7-class emotion classification that VADER cannot, enabling nuanced voice modulation
- Combining both gives us the best of both worlds: reliable intensity + granular labels

### Why pyttsx3 as Default?

- **Offline** — no API keys, no internet required, no costs
- **Cross-platform** — works on macOS (NSSpeechSynthesizer), Windows (SAPI5), Linux (espeak)
- **Native parameter control** — rate and volume are directly adjustable via the API
- **gTTS alternative** available for higher-quality output when online

### Why Linear Intensity Scaling?

- **Predictable** — judges can verify the math
- **Intuitive** — stronger emotion → proportionally stronger modulation
- **Avoidance of over-modulation** — values are clamped to safe ranges

---

## Project Structure

```
empathy-engine/
├── empathy_engine/              # Core Python package
│   ├── __init__.py
│   ├── emotion_detector.py      # VADER + HuggingFace emotion analysis
│   ├── voice_mapper.py          # Emotion → VoiceParams with intensity scaling
│   ├── tts_engine.py            # pyttsx3 + gTTS synthesis
│   └── engine.py                # Main orchestrator pipeline
├── web/                         # Flask web application
│   ├── app.py                   # Flask routes & API
│   ├── templates/
│   │   └── index.html           # Web UI
│   └── static/
│       └── style.css            # Dark-themed styling
├── output/                      # Generated audio files (auto-created)
├── cli.py                       # CLI entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.9+ |
| **Sentiment Analysis** | VADER (vaderSentiment) |
| **Emotion Classification** | HuggingFace Transformers (DistilRoBERTa) |
| **TTS (Offline)** | pyttsx3 |
| **TTS (Online)** | gTTS (Google Text-to-Speech) |
| **Audio Processing** | pydub |
| **Web Framework** | Flask |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
