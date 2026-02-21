"""
Clip audio segments from full recordings using pydub.

Why pydub: clean API for millisecond-precise slicing without re-encoding the
full file; exports to WAV directly.
"""
from __future__ import annotations

from pathlib import Path

from pydub import AudioSegment

from config import AUDIO_DIR, CLIPPED_AUDIO_DIR


def clip_segment(
    recording_id: str,
    start_sec: float,
    end_sec: float,
    out_filename: str | None = None,
) -> Path | None:
    """
    Slice [start_sec, end_sec] from the full recording WAV.

    Returns the output Path or None if the source file is missing.
    """
    src = AUDIO_DIR / f"{recording_id}.wav"
    if not src.exists():
        print(f"[WARN] Audio not found: {src}")
        return None

    audio = AudioSegment.from_wav(str(src))
    start_ms = int(start_sec * 1000)
    end_ms   = int(end_sec   * 1000)
    clip = audio[start_ms:end_ms]

    if out_filename is None:
        out_filename = f"{recording_id}_{start_ms}_{end_ms}.wav"
    out_path = CLIPPED_AUDIO_DIR / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clip.export(str(out_path), format="wav")
    return out_path


def clip_all_hits(hits_df) -> list[Path | None]:
    """
    Clip audio for every row in a disfluency hits DataFrame.

    Expected columns: recording_id, start_time, end_time, clip_filename.
    Returns list of output paths.
    """
    paths = []
    for _, row in hits_df.iterrows():
        out = clip_segment(
            recording_id=str(row["recording_id"]),
            start_sec=float(row["start_time"]),
            end_sec=float(row["end_time"]),
            out_filename=str(row["clip_filename"]),
        )
        paths.append(out)
    return paths
