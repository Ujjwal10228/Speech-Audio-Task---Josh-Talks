"""
End-to-end disfluency pipeline:
  1. Iterate over all recordings in the index CSV.
  2. Load transcription segments → detect disfluencies.
  3. Clip the corresponding audio segment.
  4. Write a structured CSV with one row per disfluency occurrence.

Output CSV schema
-----------------
recording_id | segment_index | start_time | end_time | segment_text |
disfluency_type | matched_token | clip_filename | clip_path
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import DISFLUENCY_OUTPUT_CSV, CLIPPED_AUDIO_DIR, CLIPPED_AUDIO_BASE_URL
from data_pipeline.downloader import load_transcription
from q2_disfluency.detector import analyse_recording
from q2_disfluency.audio_clipper import clip_segment


def run_pipeline(index_csv: str | Path) -> pd.DataFrame:
    """
    Process all recordings and return the disfluency DataFrame.

    *index_csv* – path to the dataset index with columns: user_id, recording_id, ...
    """
    df_index = pd.read_csv(index_csv)
    rows = []

    for _, rec in tqdm(df_index.iterrows(), total=len(df_index), desc="Detecting disfluencies"):
        rid = str(rec["recording_id"])
        segments = load_transcription(rid)
        if not segments:
            continue

        hits = analyse_recording(segments)
        for hit in hits:
            clip_fname = f"{rid}_{int(hit.start_time * 1000)}_{int(hit.end_time * 1000)}.wav"
            clip_path  = clip_segment(
                recording_id=rid,
                start_sec=hit.start_time,
                end_sec=hit.end_time,
                out_filename=clip_fname,
            )
            # PS deliverable: segmented_audio_link (URL after upload, or local path)
            segmented_link = ""
            if clip_path:
                segmented_link = (CLIPPED_AUDIO_BASE_URL.rstrip("/") + "/" + clip_fname) if CLIPPED_AUDIO_BASE_URL else str(clip_path)
            rows.append({
                "recording_id":          rid,
                "disfluency_type":       hit.types[0],
                "start_time":            round(hit.start_time, 3),
                "end_time":              round(hit.end_time, 3),
                "segmented_audio_link":  segmented_link,
                "segment_index":         hit.segment_index,
                "segment_text":          hit.text,
                "matched_token":         hit.matched_token,
                "clip_filename":         clip_fname,
                "clip_path":             str(clip_path) if clip_path else "",
            })

    result = pd.DataFrame(rows)
    result.to_csv(DISFLUENCY_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nDisfluency dataset saved: {len(result)} rows → {DISFLUENCY_OUTPUT_CSV}")
    _print_summary(result)
    return result


def _print_summary(df: pd.DataFrame):
    print("\n── Disfluency Type Distribution ──")
    print(df["disfluency_type"].value_counts().to_string())
    print(f"\nUnique recordings affected: {df['recording_id'].nunique()}")
    print(f"Total segments with disfluency: {len(df)}")


def demo() -> pd.DataFrame:
    """
    Demonstrate disfluency detection on synthetic Hindi transcription segments
    (no audio download needed).
    """
    sample_segments = [
        {"text": "मैं मैं बहुत खुश हूँ", "start_time": 0.0, "end_time": 2.5},
        {"text": "उह हम्म तो मतलब यह बात है", "start_time": 2.5, "end_time": 5.0},
        {"text": "आज मौसम बहुत अच्छा है", "start_time": 5.0, "end_time": 7.5},
        {"text": "वो वो बस ऐसे ही बोल रहा था", "start_time": 7.5, "end_time": 10.0},
        {"text": "मैं… मतलब हम कल जाएंगे", "start_time": 10.0, "end_time": 12.5},
        {"text": "हाँऽऽऽ ठीक है ना", "start_time": 12.5, "end_time": 14.0},
        {"text": "यानी लाइक बेसिकली यह सही है", "start_time": 14.0, "end_time": 16.5},
        {"text": "uh um so like basically this is it", "start_time": 16.5, "end_time": 19.0},
        {"text": "अरे ओह मैंने देखा नहीं", "start_time": 19.0, "end_time": 21.0},
        {"text": "यह बहुत ज़रूरी काम है", "start_time": 21.0, "end_time": 23.0},
    ]

    sample_recordings = {
        "demo_rec_001": sample_segments[:5],
        "demo_rec_002": sample_segments[5:],
    }

    rows = []
    for rid, segments in sample_recordings.items():
        hits = analyse_recording(segments)
        for hit in hits:
            clip_fname = f"{rid}_{int(hit.start_time * 1000)}_{int(hit.end_time * 1000)}.wav"
            rows.append({
                "recording_id":          rid,
                "disfluency_type":       hit.types[0],
                "start_time":            round(hit.start_time, 3),
                "end_time":              round(hit.end_time, 3),
                "segmented_audio_link":  "(upload clip and add link)",
                "segment_index":         hit.segment_index,
                "segment_text":          hit.text,
                "matched_token":         hit.matched_token,
                "clip_filename":         clip_fname,
                "clip_path":             "",
            })

    result = pd.DataFrame(rows)
    result.to_csv(DISFLUENCY_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nDisfluency dataset saved: {len(result)} rows → {DISFLUENCY_OUTPUT_CSV}")
    _print_summary(result)
    return result


if __name__ == "__main__":
    run_pipeline("data/index.csv")
