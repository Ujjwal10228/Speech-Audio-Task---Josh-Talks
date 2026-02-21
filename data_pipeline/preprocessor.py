"""
Preprocessing utilities: audio normalisation and text cleaning for Hindi ASR.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from config import AUDIO_DIR, TRANSCRIPTION_DIR, SAMPLE_RATE, MAX_AUDIO_DURATION, MIN_AUDIO_DURATION


# ── Text cleaning ──────────────────────────────────────────────────────────────

# Characters to strip from Hindi transcriptions
_PUNCT = re.compile(r'[।॥\.,!?;:\"\'\(\)\[\]{}<>|\\/@#$%^&*+=~`]')
_MULTI_SPACE = re.compile(r'\s+')
_DIGITS = re.compile(r'[0-9]+')          # keep Devanagari numerals (०-९), remove ASCII

def clean_text(text: str) -> str:
    """
    Normalise a Hindi transcription segment for ASR training.

    Steps: Unicode NFC, strip ASCII punctuation, collapse whitespace.
    English words written in Devanagari are preserved as-is (per guidelines).
    """
    text = unicodedata.normalize("NFC", text)
    text = _PUNCT.sub(" ", text)
    text = _DIGITS.sub("", text)        # remove inline digits (timestamps, numbering artefacts)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


# ── Audio processing ───────────────────────────────────────────────────────────

def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load and resample audio to 16 kHz mono."""
    wav, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return wav, SAMPLE_RATE


def is_valid_duration(wav: np.ndarray, sr: int) -> bool:
    duration = len(wav) / sr
    return MIN_AUDIO_DURATION <= duration <= MAX_AUDIO_DURATION


def preprocess_audio(path: str | Path, out_path: str | Path | None = None) -> np.ndarray | None:
    """
    Load, validate, and peak-normalise a WAV file.

    Returns the float32 waveform array or None if the clip is out-of-bounds.
    Saves to *out_path* (same path if None).
    """
    wav, sr = load_audio(path)
    if not is_valid_duration(wav, sr):
        return None
    # Peak normalise to [-1, 1]
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav / peak
    if out_path is None:
        out_path = path
    sf.write(str(out_path), wav, sr, subtype="PCM_16")
    return wav


def build_training_manifest(df, out_csv: str | Path = "data/train_manifest.csv"):
    """
    Build a flat CSV with columns: audio_path, text, duration
    from the downloaded index DataFrame and transcription JSONs.

    Each segment in a transcription becomes one row.
    """
    import pandas as pd
    from data_pipeline.downloader import load_transcription

    rows = []
    for _, row in df.iterrows():
        rid = str(row["recording_id"])
        audio_path = AUDIO_DIR / f"{rid}.wav"
        if not audio_path.exists():
            continue
        segments = load_transcription(rid)
        for seg in segments:
            text = clean_text(seg.get("text", ""))
            if not text:
                continue
            duration = seg.get("end_time", 0) - seg.get("start_time", 0)
            if duration < MIN_AUDIO_DURATION or duration > MAX_AUDIO_DURATION:
                continue
            rows.append({
                "recording_id": rid,
                "audio_path": str(audio_path),
                "start_time": seg.get("start_time"),
                "end_time": seg.get("end_time"),
                "text": text,
                "duration": duration,
            })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Manifest saved: {len(manifest)} segments → {out_csv}")
    return manifest


if __name__ == "__main__":
    # Quick smoke-test
    sample = "नमस्ते, यह एक परीक्षण है।  1234"
    print(clean_text(sample))
