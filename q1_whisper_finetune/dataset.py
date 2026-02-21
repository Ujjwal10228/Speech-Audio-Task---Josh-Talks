"""
HuggingFace Dataset wrapper for the Josh Talks Hindi ASR data.
Each example: {'audio': {'array': np.ndarray, 'sampling_rate': int}, 'sentence': str}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from datasets import Dataset, Audio

from config import SAMPLE_RATE


def _load_segment(row: dict) -> dict | None:
    """
    Load one audio segment.  If start/end times are given, slice the waveform;
    otherwise return the full file.
    """
    try:
        wav, sr = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
        start = row.get("start_time")
        end = row.get("end_time")
        if start is not None and end is not None:
            s = int(float(start) * sr)
            e = int(float(end) * sr)
            wav = wav[s:e]
        if len(wav) == 0:
            return None
        return {"array": wav.astype(np.float32), "sampling_rate": sr}
    except Exception as exc:
        print(f"[WARN] Could not load {row['audio_path']}: {exc}")
        return None


def build_hf_dataset(manifest_csv: str | Path, test_size: float = 0.1, seed: int = 42) -> dict:
    """
    Build train/test HuggingFace Datasets from the manifest CSV.

    Returns {'train': Dataset, 'test': Dataset}.
    """
    df = pd.read_csv(manifest_csv)
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    examples = []
    for _, row in df.iterrows():
        audio = _load_segment(row.to_dict())
        if audio is None:
            continue
        examples.append({"audio": audio, "sentence": row["text"]})

    ds = Dataset.from_list(examples)
    split = ds.train_test_split(test_size=test_size, seed=seed)
    return {"train": split["train"], "test": split["test"]}


def load_fleurs_hindi_test():
    """Load the FLEURS Hindi test split from HuggingFace."""
    from datasets import load_dataset
    try:
        fleurs = load_dataset("google/fleurs", "hi_in", split="test")
    except (RuntimeError, ValueError):
        fleurs = load_dataset(
            "google/fleurs", "hi_in", split="test",
            revision="refs/convert/parquet",
        )
    if "transcription" in fleurs.column_names:
        fleurs = fleurs.rename_column("transcription", "sentence")
    elif "raw_transcription" in fleurs.column_names:
        fleurs = fleurs.rename_column("raw_transcription", "sentence")
    return fleurs
