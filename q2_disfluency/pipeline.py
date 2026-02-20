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
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import DISFLUENCY_OUTPUT_CSV, CLIPPED_AUDIO_DIR
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
            rows.append({
                "recording_id":    rid,
                "segment_index":   hit.segment_index,
                "start_time":      round(hit.start_time, 3),
                "end_time":        round(hit.end_time, 3),
                "segment_text":    hit.text,
                "disfluency_type": hit.types[0],
                "matched_token":   hit.matched_token,
                "clip_filename":   clip_fname,
                "clip_path":       str(clip_path) if clip_path else "",
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


if __name__ == "__main__":
    run_pipeline("data/index.csv")
