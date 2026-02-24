#!/usr/bin/env python3
"""
Empathy Engine — Flask Web Interface

Run:  python web/app.py
Open: http://localhost:5000
"""

import os
import sys

# Add project root to path so we can import empathy_engine
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, request, jsonify, send_from_directory
from empathy_engine.engine import EmpathyEngine

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Output directory for generated audio
AUDIO_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Lazy-initialise engine (avoid slow startup on import)
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = EmpathyEngine(tts_backend="auto", use_hf=True)
    return _engine


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve the main web UI."""
    return render_template("index.html")


@app.route("/synthesize", methods=["POST"])
def synthesize():
    """Accept text, run the empathy engine, return JSON results."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please provide some text."}), 400

    # Optional: choose engine
    backend = data.get("engine", "pyttsx3")
    engine = get_engine()

    # Override backend if requested
    engine.tts.backend = backend

    try:
        result = engine.process(text)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    # Make audio path relative for serving
    audio_filename = os.path.basename(result["audio"])
    result["audio_url"] = f"/audio/{audio_filename}"

    return jsonify(result)


@app.route("/audio/<filename>")
def serve_audio(filename):
    """Serve generated audio files."""
    return send_from_directory(AUDIO_DIR, filename)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🌐 Starting Empathy Engine Web Interface...")
    print("   Open http://localhost:5001 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5001)
